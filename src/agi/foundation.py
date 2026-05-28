"""Frozen-LLM foundation wrapper for the AGI architecture.

Wraps a pretrained instruct-tuned LLM (Qwen2-0.5B-Instruct by
default) as a *frozen* stable substrate. Phase 1.0 does not
update any foundation parameters — the foundation's job is to
provide:

1. **Stable key vectors** via mean-pooled last-hidden-state.
   Used to index :class:`XRayEpisodicMemory`. Stable because
   the foundation is frozen, so the same text always produces
   the same key.
2. **Generation** for assistant responses. Augmented with
   retrieved facts when the memory has relevant entries.

Device + dtype are auto-picked unless the caller overrides:
- CUDA available → ``cuda`` + ``fp16`` (matches Colab L4 plan).
- Otherwise     → ``cpu`` + ``fp32`` (safer; fp16 on CPU is
  fragile across torch versions).
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _pick_default_device_dtype() -> tuple[str, torch.dtype]:
    """CUDA + fp16 when available; otherwise CPU + fp32."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    # MPS is sometimes flaky for HF generation paths on small
    # models; default to CPU for the deterministic path. Users
    # can pass device="mps" explicitly to override.
    return "cpu", torch.float32


class FrozenFoundation:
    """Wraps a pretrained LLM as the stable foundation.

    All parameters are frozen and ``model.eval()`` is set at
    construction time. The wrapper exposes a stable-key
    ``get_key()`` and a ``generate()`` for responses.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B-Instruct",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        default_device, default_dtype = _pick_default_device_dtype()
        self.device = device if device is not None else default_device
        if dtype is None:
            # Dtype default tracks device unless caller overrides.
            self.dtype = (
                torch.float16 if self.device == "cuda" else torch.float32
            )
        else:
            self.dtype = dtype
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # transformers 4.57+ deprecated ``torch_dtype`` in favour
        # of the unified ``dtype`` kwarg; we use the new spelling
        # to keep the load warning-free on current versions.
        load_kwargs: dict = {"dtype": self.dtype}
        if self.device == "cuda":
            # device_map="cuda" handles cross-device placement;
            # on CPU we move explicitly after load to keep
            # behaviour predictable across transformers versions.
            load_kwargs["device_map"] = self.device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        # Lazily resolved once; depends on the loaded model's config.
        self._key_dim: Optional[int] = None

    @property
    def key_dim(self) -> int:
        """Hidden dimension of the loaded model — used by
        :class:`XRayEpisodicMemory` to size its key buffer."""
        if self._key_dim is None:
            self._key_dim = int(self.model.config.hidden_size)
        return self._key_dim

    @torch.no_grad()
    def get_key(self, text: str) -> torch.Tensor:
        """Compute a stable key vector for ``text`` via mean-
        pooled last-hidden-state.

        The pooling is mask-aware so padding tokens don't dilute
        the mean. The returned tensor is float32 on CPU regardless
        of the foundation's compute dtype — keys are used by the
        memory layer and we want consistent precision there.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # (1, T, H)
        mask = inputs["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # Move to CPU + float32 for storage-side comparison stability.
        return pooled.squeeze(0).to(torch.float32).cpu()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response continuation for ``prompt``.

        Only the *new* tokens are returned, not the prompt itself.
        ``temperature=0`` switches to greedy decoding (deterministic).
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False
        outputs = self.model.generate(**inputs, **gen_kwargs)
        # Slice the prompt tokens off and decode only the new tail.
        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
