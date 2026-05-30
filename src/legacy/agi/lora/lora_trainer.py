"""LoRA training loop for Phase 2e.

Designed to run on a Colab T4/L4 GPU. The heavy imports
(``peft``, ``transformers.AutoModelForCausalLM``) are deliberately
*lazy* — module import is cheap and unit tests don't need peft
locally. Setup + training only touch peft inside their function
bodies.

The dataset / collation logic IS imported eagerly because it's
trivially unit-testable on CPU with a small tokenizer.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, TYPE_CHECKING

import torch
from torch.utils.data import Dataset


if TYPE_CHECKING:  # only for typing — no runtime import
    from transformers import PreTrainedTokenizerBase


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

@dataclass
class LoRAConfig:
    """Hyperparameters for one LoRA distillation run.

    Defaults sized for Qwen2.5-1.5B on Colab T4 (16 GB VRAM):
      * rank-8 LoRA on ``q_proj`` + ``v_proj`` only (≤ 3M trainable params)
      * batch_size 4 × gradient_accumulation_steps 2 = effective 8
      * max_seq_length 256 (training data is short)
      * 3 epochs, AdamW lr 1e-4
    """

    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"],
    )

    learning_rate: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 3
    max_seq_length: int = 256
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 50
    log_every_steps: int = 50

    seed: int = 42
    device: str = "cuda"


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

class DistillationDataset(Dataset):
    """Wraps a JSONL file of ``{"prompt": ..., "target": ...}``
    records into tokenised ``(input_ids, labels)`` samples.

    The ``labels`` tensor masks the prompt portion with ``-100``
    so cross-entropy is computed *only* over the target tokens
    (standard SFT-on-completion practice). Avoids the model
    learning to repeat the prompt and saves loss-bandwidth for
    the response.
    """

    PROMPT_MASK_ID: int = -100

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: "PreTrainedTokenizerBase",
        max_length: int = 256,
        records: Optional[Sequence[dict]] = None,
    ) -> None:
        """``records`` lets tests pass in-memory dicts without
        writing a JSONL file. When set, ``jsonl_path`` is
        ignored (caller can pass ``Path()`` placeholder)."""
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        if records is not None:
            self.records = list(records)
        else:
            self.records = []
            with Path(jsonl_path).open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        prompt = str(rec["prompt"])
        target = str(rec["target"])

        # Tokenize prompt + target separately so we know where to
        # mask the labels.
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=False,
        )["input_ids"]
        target_ids = self.tokenizer(
            target, add_special_tokens=False,
        )["input_ids"]
        # Append EOS so the model learns to stop.
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            target_ids = list(target_ids) + [eos]

        input_ids = list(prompt_ids) + list(target_ids)
        labels = (
            [self.PROMPT_MASK_ID] * len(prompt_ids)
            + list(target_ids)
        )

        # Truncate to max_length from the LEFT (keeps the target
        # intact at the right edge where generation happens).
        if len(input_ids) > self.max_length:
            keep = self.max_length
            input_ids = input_ids[-keep:]
            labels = labels[-keep:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def pad_collate(
    batch: List[dict],
    pad_token_id: int,
) -> dict:
    """Right-pad a batch of ``{input_ids, labels}`` to the longest
    sequence in the batch. Pad positions get ``-100`` in labels."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids = torch.full(
        (len(batch), max_len), pad_token_id, dtype=torch.long,
    )
    labels = torch.full(
        (len(batch), max_len),
        DistillationDataset.PROMPT_MASK_ID,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (len(batch), max_len), dtype=torch.long,
    )
    for i, item in enumerate(batch):
        n = item["input_ids"].shape[0]
        input_ids[i, :n] = item["input_ids"]
        labels[i, :n] = item["labels"]
        attention_mask[i, :n] = 1
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


# ----------------------------------------------------------------------
# Setup + training (lazy peft imports)
# ----------------------------------------------------------------------

def setup_lora_model(config: LoRAConfig):
    """Load base Qwen + attach a fresh LoRA adapter.

    Returns ``(model, tokenizer)``. Imports ``peft`` and
    ``transformers`` lazily so this module loads cleanly without
    those deps installed.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Phase 2e: fp32 on CUDA (was fp16). Qwen-1.5B LoRA on
    # T4 (Turing — no bf16 support) with naked fp16 attention
    # in eager mode reliably overflows to NaN in softmax on the
    # very first forward pass.
    #
    # IMPORTANT — explicit ``.to(torch.float32)`` after load:
    # Qwen2.5-1.5B-Instruct's ``config.json`` ships
    # ``torch_dtype="float16"``. On some transformers/peft
    # version combinations the ``dtype=`` kwarg to
    # ``from_pretrained`` is silently overridden by the config's
    # own ``torch_dtype``, so the model comes back as fp16 even
    # when we asked for fp32. The explicit cast below is
    # belt-and-suspenders: regardless of what the loader gave
    # us, the model + LoRA train in fp32 from here on.
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        dtype=torch.float32,
        torch_dtype=torch.float32,  # alias for older transformers
        attn_implementation="eager",
    )
    model = model.to(torch.float32)

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer


def evaluate_lora(model, val_loader, device: str) -> float:
    """Compute mean per-sample validation loss."""
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            bs = input_ids.shape[0]
            total += float(out.loss.item()) * bs
            n += bs
    model.train()
    return total / max(n, 1)


def train_lora(
    config: LoRAConfig,
    train_data_path: Path,
    val_data_path: Path,
    output_dir: Path,
) -> List[dict]:
    """End-to-end LoRA training loop.

    Returns a history list of ``{step, train_loss, val_loss}``
    dicts (one entry per ``log_every_steps`` checkpoint). The
    trained adapter + tokenizer are saved to ``output_dir`` on
    completion.

    All heavy imports happen inside this function — keep it as
    the single GPU-bound entry point so the rest of the module
    stays importable on CPU.
    """
    from torch.utils.data import DataLoader

    torch.manual_seed(config.seed)

    model, tokenizer = setup_lora_model(config)
    model.to(config.device)

    train_ds = DistillationDataset(
        train_data_path, tokenizer, config.max_seq_length,
    )
    val_ds = DistillationDataset(
        val_data_path, tokenizer, config.max_seq_length,
    )
    pad_id = tokenizer.pad_token_id

    def _collate(batch: List[dict]) -> dict:
        return pad_collate(batch, pad_token_id=pad_id)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        shuffle=True, collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False, collate_fn=_collate,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
    )
    n_train_steps = math.ceil(
        len(train_loader) * config.num_epochs
        / max(config.gradient_accumulation_steps, 1)
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            1.0, (step + 1) / max(config.warmup_steps, 1),
        ),
    )

    history: List[dict] = []
    step = 0
    model.train()
    for epoch in range(config.num_epochs):
        for batch in train_loader:
            input_ids = batch["input_ids"].to(config.device)
            labels = batch["labels"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out.loss / max(config.gradient_accumulation_steps, 1)
            loss.backward()
            if (step + 1) % max(config.gradient_accumulation_steps, 1) == 0:
                # Clip gradient norm to 1.0 — cheap insurance
                # against any remaining numerical spikes (added
                # alongside the Phase 2e fp32 switch).
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if step % config.log_every_steps == 0:
                val_loss = evaluate_lora(model, val_loader, config.device)
                entry = {
                    "step": int(step),
                    "epoch": int(epoch),
                    "train_loss": float(out.loss.item()),
                    "val_loss": float(val_loss),
                }
                history.append(entry)
                print(
                    f"step={step:5d} epoch={epoch} "
                    f"train_loss={out.loss.item():.4f} "
                    f"val_loss={val_loss:.4f}"
                )
            step += 1

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))

    return history


__all__ = [
    "DistillationDataset",
    "LoRAConfig",
    "evaluate_lora",
    "pad_collate",
    "setup_lora_model",
    "train_lora",
]
