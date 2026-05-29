"""Generate the LoRA distillation training dataset locally.

Runs Qwen + metacog teacher pipeline on a synthetic query mix
to produce ``(prompt, target)`` JSONL pairs. The output files
are gitignored — they're regenerated per-machine and uploaded to
Colab for Phase 2e LoRA training.

Run from the repo root::

    python scripts/generate_lora_training_data.py
    python scripts/generate_lora_training_data.py --n-per-category 25  # quick smoke

Output structure (default ``--output-dir data/lora``):

    data/lora/train.jsonl    ~80% of generated examples
    data/lora/val.jsonl      ~20% of generated examples
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from agi.foundation import FrozenFoundation  # noqa: E402
from agi.lora.distillation import TeacherPipeline  # noqa: E402
from agi.lora.training_data import (  # noqa: E402
    generate_distillation_dataset,
    generate_training_queries,
)
from agi.metacognition.orchestrator import MetacognitiveOrchestrator  # noqa: E402
from agi.metacognition.templates import ResponseTemplates  # noqa: E402
from agi.metacognition.training import load_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the Phase 2e LoRA distillation dataset by "
            "running the teacher pipeline on a synthetic query mix."
        )
    )
    parser.add_argument(
        "--n-per-category", type=int, default=250,
        help="Queries per category (4 categories ⇒ 4× = total).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/lora"),
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("data/metacog/checkpoints"),
    )
    parser.add_argument(
        "--template-key", type=str, default="ignorance_polite_fr",
        help="Template the teacher uses when admitting ignorance.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Teacher generation temperature (0 = greedy).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=96,
        help="Cap on teacher response length.",
    )
    args = parser.parse_args()

    t_load = time.perf_counter()
    print("Loading Qwen2.5-1.5B foundation ...")
    foundation = FrozenFoundation()
    print(f"  Loaded in {time.perf_counter() - t_load:.1f}s")

    print("Loading metacog checkpoints ...")
    pre_layer = load_checkpoint(
        args.checkpoint_dir / "pre_layer.pt", mode="pre",
    )
    post_layer = load_checkpoint(
        args.checkpoint_dir / "post_layer.pt", mode="post",
    )
    templates = ResponseTemplates()
    orchestrator = MetacognitiveOrchestrator(
        pre_layer=pre_layer,
        post_layer=post_layer,
        templates=templates,
    )
    teacher = TeacherPipeline(
        foundation, orchestrator, templates,
        template_key=args.template_key,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(
        f"Generating queries: {args.n_per_category}/category "
        f"× 4 categories = {args.n_per_category * 4} total ..."
    )
    queries = generate_training_queries(
        n_per_category=args.n_per_category, seed=args.seed,
    )

    n_val = int(len(queries) * args.val_ratio)
    val_queries = queries[:n_val]
    train_queries = queries[n_val:]
    print(
        f"Split: {len(train_queries)} train / {len(val_queries)} val"
    )

    t_train = time.perf_counter()
    print("Running teacher pipeline on train queries ...")
    n_train = generate_distillation_dataset(
        train_queries, teacher, foundation,
        args.output_dir / "train.jsonl",
    )
    print(
        f"  {n_train} train examples in "
        f"{time.perf_counter() - t_train:.1f}s"
    )

    t_val = time.perf_counter()
    print("Running teacher pipeline on val queries ...")
    n_val_written = generate_distillation_dataset(
        val_queries, teacher, foundation,
        args.output_dir / "val.jsonl",
    )
    print(
        f"  {n_val_written} val examples in "
        f"{time.perf_counter() - t_val:.1f}s"
    )

    print()
    print(f"Done. Train: {n_train}, Val: {n_val_written}")
    print(
        f"Files: {args.output_dir / 'train.jsonl'} + "
        f"{args.output_dir / 'val.jsonl'}"
    )
    print(
        "Upload both files to Colab and run "
        "colab/phase_2e_lora_training.ipynb."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
