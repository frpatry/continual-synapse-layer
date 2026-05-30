"""Tests for the LoRA trainer.

These tests do NOT load real Qwen and do NOT require ``peft`` to
be installed locally — they exercise the dataset / collate
logic, the config defaults, and the lazy import surface. The
actual training loop runs in Colab; correctness of the lazy
imports is verified by checking ``setup_lora_model`` raises a
clean ``ImportError`` when peft is absent.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch

from agi.lora.lora_trainer import (
    DistillationDataset,
    LoRAConfig,
    pad_collate,
)


# ---------- LoRAConfig ----------

def test_lora_config_defaults_are_reasonable():
    cfg = LoRAConfig()
    assert cfg.base_model_name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.lora_rank == 8
    assert cfg.lora_alpha == 16
    assert cfg.target_modules == ["q_proj", "v_proj"]
    assert cfg.batch_size >= 1
    assert cfg.num_epochs >= 1
    assert cfg.max_seq_length >= 64


def test_lora_config_target_modules_default_is_fresh_list():
    """The default_factory must produce a fresh list per instance —
    otherwise two configs would share the same list and mutation
    would leak across instances."""
    a = LoRAConfig()
    b = LoRAConfig()
    a.target_modules.append("o_proj")
    assert "o_proj" not in b.target_modules


# ---------- DistillationDataset ----------

class _FakeTokenizer:
    """Whitespace tokenizer with deterministic small vocab."""

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self.eos_token_id = 0
        self.pad_token_id = 1
        # Reserve IDs 0 and 1.
        self._next_id = 2

    def _id(self, tok: str) -> int:
        if tok not in self._vocab:
            self._vocab[tok] = self._next_id
            self._next_id += 1
        return self._vocab[tok]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self._id(t) for t in text.split()]}


def test_dataset_loads_jsonl_and_returns_tensors(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text(
        json.dumps({"prompt": "say hello", "target": "hello world"}) + "\n"
        + json.dumps({"prompt": "name?", "target": "François"}) + "\n"
    )
    ds = DistillationDataset(path, _FakeTokenizer(), max_length=32)
    assert len(ds) == 2
    item = ds[0]
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)
    assert item["input_ids"].dim() == 1
    assert item["input_ids"].shape == item["labels"].shape


def test_dataset_masks_prompt_in_labels():
    """The CE loss should be computed only over the target tokens —
    prompt positions get -100 in labels."""
    tok = _FakeTokenizer()
    ds = DistillationDataset(
        Path(),  # ignored when records=
        tok,
        max_length=32,
        records=[{"prompt": "Q ?", "target": "A B"}],
    )
    item = ds[0]
    labels = item["labels"]
    input_ids = item["input_ids"]
    # First 2 tokens (the prompt "Q ?") must be masked.
    assert (labels[:2] == DistillationDataset.PROMPT_MASK_ID).all()
    # The remaining (target tokens + EOS) must be non-masked.
    assert (labels[2:] != DistillationDataset.PROMPT_MASK_ID).all()


def test_dataset_appends_eos_to_target():
    tok = _FakeTokenizer()
    ds = DistillationDataset(
        Path(), tok, max_length=32,
        records=[{"prompt": "Q", "target": "A"}],
    )
    item = ds[0]
    # Prompt "Q" (1 token) + target "A" (1 token) + EOS = 3.
    assert item["input_ids"].shape[0] == 3
    assert int(item["input_ids"][-1].item()) == tok.eos_token_id


def test_dataset_truncates_from_the_left():
    """When the combined length exceeds max_length, we keep the
    RIGHT side (target) intact."""
    tok = _FakeTokenizer()
    ds = DistillationDataset(
        Path(), tok, max_length=4,
        records=[{"prompt": "a b c d e f g h", "target": "x y"}],
    )
    item = ds[0]
    # Max 4 tokens total; the EOS (added to target) is at the end.
    assert item["input_ids"].shape[0] == 4
    assert int(item["input_ids"][-1].item()) == tok.eos_token_id


# ---------- pad_collate ----------

def test_pad_collate_pads_to_longest_in_batch():
    batch = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([1, 2, 3]),
        },
        {
            "input_ids": torch.tensor([5]),
            "labels": torch.tensor([5]),
        },
    ]
    out = pad_collate(batch, pad_token_id=99)
    assert out["input_ids"].shape == (2, 3)
    assert out["labels"].shape == (2, 3)
    assert out["attention_mask"].shape == (2, 3)
    # Padding token in slot 1 of the second sample.
    assert int(out["input_ids"][1, 1].item()) == 99
    # Padding label is -100 (masked).
    assert int(out["labels"][1, 1].item()) == DistillationDataset.PROMPT_MASK_ID
    # Attention mask 1 where real, 0 where padded.
    assert out["attention_mask"][1].tolist() == [1, 0, 0]


# ---------- Lazy peft import ----------

def test_setup_lora_model_lazy_imports_peft():
    """When peft isn't installed, ``setup_lora_model`` should raise
    a clean ``ImportError`` mentioning peft — not crash at module
    import time."""
    from agi.lora.lora_trainer import setup_lora_model  # noqa: PLC0415
    # If peft IS installed, this test is a no-op (the call would
    # still work). The point is: importing the trainer module
    # didn't fail above.
    try:
        importlib.import_module("peft")
        peft_available = True
    except ImportError:
        peft_available = False

    if peft_available:
        pytest.skip("peft is installed — the lazy-import path can't be exercised")
    cfg = LoRAConfig(device="cpu")
    with pytest.raises(ImportError):
        setup_lora_model(cfg)
