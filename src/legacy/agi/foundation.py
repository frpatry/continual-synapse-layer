"""Frozen-LLM foundation wrapper for the AGI architecture.

Wraps a pretrained instruct-tuned LLM (Qwen2.5-1.5B-Instruct by
default) as a *frozen* stable substrate. Phase 1.0 used the
smaller Qwen2-0.5B-Instruct; Phase 1.2 upgraded to Qwen2.5-1.5B
after the Phase 1.1 demo showed Qwen-0.5B couldn't reliably
follow the JSON-only extraction instruction (it confabulated a
full fact schema on every input). Qwen2.5-1.5B is large enough
to follow few-shot extraction prompts cleanly while still fitting
in fp16 on a T4/L4 (the model is ~3 GB on disk, ~1.5 GB on GPU).
Override with ``model_name=...`` to revert to 0.5B for
comparison runs. The foundation's job is to provide:

1. **Stable key vectors** via mean-pooled last-hidden-state.
   Used to index :class:`XRayEpisodicMemory`. Stable because
   the foundation is frozen, so the same text always produces
   the same key.
2. **Generation** for assistant responses. Augmented with
   retrieved facts when the memory has relevant entries.
3. **Per-token generation signals** (Phase 2b) — entropy and
   attention-to-fact-spans — captured by
   :meth:`generate_with_signals` and consumed by the
   metacognitive layer.

Device + dtype are auto-picked unless the caller overrides:
- CUDA available → ``cuda`` + ``fp16`` (matches Colab L4 plan).
- Otherwise     → ``cpu`` + ``fp32`` (safer; fp16 on CPU is
  fragile across torch versions).

The model is loaded with ``attn_implementation="eager"`` so
that ``output_attentions=True`` returns real attention tensors.
The default SDPA backend can silently return ``None`` for
attentions on some transformers versions; eager is uniformly
supported and the perf cost is small for the model sizes we
use (≤ 1.5B).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _pick_default_device_dtype() -> tuple[str, torch.dtype]:
    """CUDA + fp16 when available; otherwise CPU + fp32."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    # MPS is sometimes flaky for HF generation paths on small
    # models; default to CPU for the deterministic path. Users
    # can pass device="mps" explicitly to override.
    return "cpu", torch.float32


@dataclass
class GenerationInfo:
    """Per-generation signal record returned by
    :meth:`FrozenFoundation.generate_with_signals`.

    Attributes:
        response_text: Decoded response (prompt stripped, special
            tokens skipped) — the same string the legacy
            :meth:`FrozenFoundation.generate` would return.
        generated_token_ids: IDs of the newly-generated tokens
            (no prompt tokens).
        response_length_tokens: ``len(generated_token_ids)``,
            cached for convenience.
        mean_token_entropy: Mean Shannon entropy (nats) of the
            per-step generation distribution, averaged over
            generated tokens. Bounded by ``log(vocab_size)``.
        max_token_entropy: Maximum per-step entropy. Useful as a
            "the model was confused on at least one token" signal.
        attention_to_facts_mean: Mean (over generated steps) of
            the attention mass each newly-generated token paid
            to ``fact_token_ranges``. ``0.0`` when no ranges are
            provided or when all ranges fall outside the prompt.
        attention_to_facts_max: Per-step max of the same signal.
        generation_time_seconds: Wall time of the call, useful
            for benchmarking the eager-attention overhead.
    """

    response_text: str
    generated_token_ids: List[int]
    response_length_tokens: int
    mean_token_entropy: float
    max_token_entropy: float
    attention_to_facts_mean: float
    attention_to_facts_max: float
    generation_time_seconds: float

    def to_dict(self) -> dict:
        """JSON-serialisable rendering for logging."""
        return {
            "response_text": self.response_text,
            "generated_token_ids": list(self.generated_token_ids),
            "response_length_tokens": int(self.response_length_tokens),
            "mean_token_entropy": float(self.mean_token_entropy),
            "max_token_entropy": float(self.max_token_entropy),
            "attention_to_facts_mean": float(self.attention_to_facts_mean),
            "attention_to_facts_max": float(self.attention_to_facts_max),
            "generation_time_seconds": float(self.generation_time_seconds),
        }


class FrozenFoundation:
    """Wraps a pretrained LLM as the stable foundation.

    All parameters are frozen and ``model.eval()`` is set at
    construction time. The wrapper exposes a stable-key
    ``get_key()``, a plain-text ``generate()``, and a
    signal-rich ``generate_with_signals()``.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
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
        #
        # ``attn_implementation="eager"`` is critical for
        # ``generate_with_signals`` — the SDPA / FlashAttention
        # backends short-circuit attention-weight materialisation
        # and ``output_attentions=True`` either warns + falls back
        # silently or returns ``None`` (varies by HF version).
        # Eager attention is uniformly supported and the wallclock
        # cost on a ≤ 1.5B model is modest.
        load_kwargs: dict = {
            "dtype": self.dtype,
            "attn_implementation": "eager",
        }
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

    # ------------------------------------------------------------------
    # Phase 2b — generation with internal signals
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_with_signals(
        self,
        prompt: str,
        fact_token_ranges: Optional[List[Tuple[int, int]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> GenerationInfo:
        """Generate and capture per-token generation signals.

        Returns a :class:`GenerationInfo` containing the response
        plus two aggregate signals the metacognitive layer reads:

        - **Per-token entropy**: how confident the model was at
          each generated step (lower is more confident; bounded
          by ``log(vocab_size)``).
        - **Attention-to-facts**: how much attention each newly
          generated token paid to ``fact_token_ranges`` in the
          prompt — a proxy for "did the model lean on the
          injected facts or on its own priors?".

        ``fact_token_ranges`` is a list of ``(start, end)`` index
        pairs into the *tokenised* prompt (half-open intervals,
        in input-ID space). When ``None`` or empty,
        ``attention_to_facts_*`` are returned as ``0.0`` and the
        attention bookkeeping is skipped (faster path).

        Per-step attention is averaged across layers and heads;
        the slice we use is the last query row of each step's
        attention tensor — i.e. the attention the *just-generated*
        token paid to all previous positions, including the fact
        spans.
        """
        t0 = time.perf_counter()

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])

        want_attentions = bool(fact_token_ranges)

        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "output_scores": True,
            "output_attentions": want_attentions,
            "return_dict_in_generate": True,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False
        out = self.model.generate(**inputs, **gen_kwargs)

        # ``sequences`` includes the prompt; slice off the prompt
        # portion to get the newly-generated tail.
        full_seq = out.sequences[0]
        new_token_ids = full_seq[prompt_len:].tolist()
        response_text = self.tokenizer.decode(
            new_token_ids, skip_special_tokens=True,
        )

        # ----- per-step entropy -----
        # ``scores`` is a tuple of length n_generated, each entry
        # of shape (batch, vocab). We compute entropy directly
        # from log-softmax to keep numerical stability (avoids
        # log(softmax(x)) which can underflow).
        scores: tuple = out.scores or ()
        if not scores:
            mean_entropy = 0.0
            max_entropy = 0.0
        else:
            entropies: List[float] = []
            for step_logits in scores:
                # (1, vocab) -> (vocab,)
                step_logits_1d = step_logits[0]
                log_probs = F.log_softmax(
                    step_logits_1d.to(torch.float32), dim=-1,
                )
                probs = log_probs.exp()
                # Use -(p * log p) summed over vocab; clamp NaNs
                # from p == 0 (log p → -inf) via masked sum.
                ent_term = probs * log_probs
                ent_term = torch.where(
                    probs > 0, ent_term, torch.zeros_like(ent_term),
                )
                entropy = float(-(ent_term.sum()).item())
                entropies.append(entropy)
            mean_entropy = float(sum(entropies) / len(entropies))
            max_entropy = float(max(entropies))

        # ----- attention-to-facts -----
        att_mean = 0.0
        att_max = 0.0
        if want_attentions and out.attentions:
            # ``out.attentions`` is a tuple of length n_generated.
            # Each entry is a tuple of length n_layers of tensors
            # of shape (batch, num_heads, q_len, k_len). For
            # step 0 q_len == prompt_len; subsequent steps with
            # KV-cache typically have q_len == 1. We always take
            # the *last* query row — the row corresponding to the
            # token being generated.
            per_step_attention: List[float] = []
            # Pre-clamp ranges to prompt bounds.
            ranges = _clamp_ranges(fact_token_ranges, prompt_len)
            if ranges:
                for layer_tuple in out.attentions:
                    # Sum attention to fact positions across layers
                    # and heads — running aggregation to avoid
                    # holding the full attention tuple alongside
                    # downstream tensors.
                    per_step_total = 0.0
                    n_layer = 0
                    for layer_att in layer_tuple:
                        # (1, num_heads, q_len, k_len)
                        last_row = layer_att[0, :, -1, :]  # (heads, k)
                        # Average across heads.
                        head_avg = last_row.mean(dim=0)  # (k,)
                        # Sum mass on fact positions (skip out-of-
                        # range indices that may appear with KV
                        # cache shrinking k_len in odd HF builds).
                        k_len = int(head_avg.shape[0])
                        step_mass = 0.0
                        for start, end in ranges:
                            s = max(0, min(start, k_len))
                            e = max(0, min(end, k_len))
                            if e > s:
                                step_mass += float(
                                    head_avg[s:e].sum().item()
                                )
                        per_step_total += step_mass
                        n_layer += 1
                    if n_layer > 0:
                        per_step_attention.append(
                            per_step_total / n_layer
                        )
            if per_step_attention:
                att_mean = float(
                    sum(per_step_attention) / len(per_step_attention)
                )
                att_max = float(max(per_step_attention))

        elapsed = time.perf_counter() - t0
        return GenerationInfo(
            response_text=response_text,
            generated_token_ids=new_token_ids,
            response_length_tokens=len(new_token_ids),
            mean_token_entropy=mean_entropy,
            max_token_entropy=max_entropy,
            attention_to_facts_mean=att_mean,
            attention_to_facts_max=att_max,
            generation_time_seconds=elapsed,
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _clamp_ranges(
    ranges: List[Tuple[int, int]] | None, upper: int,
) -> List[Tuple[int, int]]:
    """Drop / clamp ``(start, end)`` pairs to ``[0, upper)``.

    Out-of-bounds inputs are silently dropped rather than raising
    — callers may not know the exact prompt length until after
    tokenisation, and we'd rather return ``att=0`` than crash.
    """
    if not ranges:
        return []
    out: List[Tuple[int, int]] = []
    for start, end in ranges:
        s = max(0, int(start))
        e = min(int(upper), int(end))
        if e > s:
            out.append((s, e))
    return out
