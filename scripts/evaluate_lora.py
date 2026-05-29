"""Compare base Qwen vs LoRA-adapted Qwen on the Phase 2d.2
100-case validation set.

For each case we generate a response from BOTH models (no metacog
scaffolding) and run the resulting (response, features) through
the trained POST metacog layer to estimate hallucination rates.

Run from the repo root::

    python scripts/evaluate_lora.py \\
        --lora-adapter data/lora/adapter \\
        --output results/agi/phase_2e_comparison_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from agi.foundation import FrozenFoundation  # noqa: E402
from agi.memory.precision import serialize_facts  # noqa: E402
from agi.memory.xray_episodic import XRayEpisodicMemory  # noqa: E402
from agi.metacognition.features import (  # noqa: E402
    assemble_feature_vector,
    extract_alignment_features,
    extract_generation_features,
    extract_memory_features,
    extract_query_features,
)
from agi.metacognition.training import load_checkpoint  # noqa: E402

from experiments.agi.phase_2_validation.test_cases import ALL_CASES  # noqa: E402


def _build_memory(foundation, facts: list[dict]) -> XRayEpisodicMemory:
    memory = XRayEpisodicMemory(
        key_dim=foundation.key_dim,
        retrieval_threshold=0.3,
        foundation=foundation,
    )
    for fact in facts:
        text = serialize_facts(fact)
        key = foundation.get_key(text)
        memory.add_entry(key, fact)
    return memory


def _build_prompt(query: str, retrieval) -> str:
    """Raw prompt — no admission scaffolding. Mirrors what the
    LoRA student saw during training."""
    if not retrieval:
        return f"Question: {query}\n\nRéponse:"
    facts_str = "\n".join(
        f"- {serialize_facts(entry.facts)}" for entry, _sim in retrieval
    )
    return f"Contexte connu:\n{facts_str}\n\nQuestion: {query}\n\nRéponse:"


def _evaluate_one(
    case,
    foundation: FrozenFoundation,
    text_generator,  # callable: prompt → response text
    post_layer,
) -> dict:
    memory = _build_memory(foundation, case.memory_facts)
    query_key = foundation.get_key(case.query)
    retrieval = memory.retrieve(query_key, top_k=5)

    prompt = _build_prompt(case.query, retrieval)
    t0 = time.perf_counter()
    response = text_generator(prompt)
    dt = time.perf_counter() - t0

    memory_feats = extract_memory_features(retrieval)
    query_feats = extract_query_features(case.query, foundation)
    # No gen_info from the LoRA path → generation features are zero.
    gen_feats = extract_generation_features(None)
    align_feats = extract_alignment_features(
        response=response, facts=retrieval, foundation=foundation,
    )
    combined = {**memory_feats, **query_feats, **gen_feats, **align_feats}
    tensor = assemble_feature_vector(combined, mode="post")
    post_state = post_layer.predict(tensor, raw_features=combined)

    return {
        "case_id": case.case_id,
        "query": case.query,
        "expected_status": case.expected_status,
        "post_predicted_status": post_state.epistemic_status,
        "post_recommended_action": post_state.recommended_action,
        "response": response,
        "gen_seconds": float(dt),
    }


def _summarize(results: list[dict], label: str) -> dict:
    by_status = Counter(r["post_predicted_status"] for r in results)
    by_action = Counter(r["post_recommended_action"] for r in results)
    hallucination_rate = by_status["hallucinated"] / max(len(results), 1)
    admit_rate = by_action["admit_ignorance"] / max(len(results), 1)
    return {
        "label": label,
        "n": len(results),
        "predicted_status_distribution": dict(by_status),
        "recommended_action_distribution": dict(by_action),
        "hallucination_rate": float(hallucination_rate),
        "admit_ignorance_rate": float(admit_rate),
    }


def _markdown_report(
    summary_base: dict, summary_lora: dict, output_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Phase 2e — Base vs LoRA Comparison")
    lines.append("")
    lines.append(
        "100 hand-crafted Phase 2d.2 validation cases, raw "
        "prompts (no metacog scaffolding), POST metacog layer "
        "judges each response."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("| metric | base Qwen | LoRA Qwen | Δ |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| hallucination rate | "
        f"{summary_base['hallucination_rate']:.3f} | "
        f"{summary_lora['hallucination_rate']:.3f} | "
        f"{summary_lora['hallucination_rate'] - summary_base['hallucination_rate']:+.3f} |"
    )
    lines.append(
        f"| admit_ignorance rate | "
        f"{summary_base['admit_ignorance_rate']:.3f} | "
        f"{summary_lora['admit_ignorance_rate']:.3f} | "
        f"{summary_lora['admit_ignorance_rate'] - summary_base['admit_ignorance_rate']:+.3f} |"
    )
    lines.append("")
    lines.append("## Predicted-status distributions")
    lines.append("")
    lines.append("| status | base count | LoRA count |")
    lines.append("|---|---:|---:|")
    statuses = set(summary_base["predicted_status_distribution"]) | set(
        summary_lora["predicted_status_distribution"]
    )
    for s in sorted(statuses):
        lines.append(
            f"| {s} | "
            f"{summary_base['predicted_status_distribution'].get(s, 0)} | "
            f"{summary_lora['predicted_status_distribution'].get(s, 0)} |"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lora-adapter", type=Path,
        default=Path("data/lora/adapter"),
        help="Path to the trained LoRA adapter directory.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("data/metacog/checkpoints"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/agi/phase_2e_comparison_report.md"),
    )
    parser.add_argument(
        "--raw-jsonl", type=Path,
        default=Path("results/agi/phase_2e_comparison_raw.jsonl"),
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=96,
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
    )
    parser.add_argument(
        "--cases-limit", type=int, default=None,
    )
    args = parser.parse_args()

    if not args.lora_adapter.exists():
        print(
            f"ERROR: LoRA adapter not found at {args.lora_adapter}. "
            "Train it in colab/phase_2e_lora_training.ipynb first.",
            file=sys.stderr,
        )
        return 1

    print("Loading base foundation ...")
    foundation = FrozenFoundation()
    post_layer = load_checkpoint(
        args.checkpoint_dir / "post_layer.pt", mode="post",
    )

    # The LoRA adapter is loaded onto the same base via peft.
    # Import is lazy — only this script needs peft locally.
    from peft import PeftModel  # noqa: PLC0415

    base_model = foundation.model
    lora_model = PeftModel.from_pretrained(base_model, str(args.lora_adapter))
    lora_model.eval()

    tokenizer = foundation.tokenizer

    def _gen(model):
        def _generate(prompt: str) -> str:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024,
            ).to(foundation.device)
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                pad_token_id=tokenizer.eos_token_id,
            )
            return tokenizer.decode(
                out[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
        return _generate

    cases = list(ALL_CASES)
    if args.cases_limit:
        cases = cases[: args.cases_limit]

    base_results: list[dict] = []
    lora_results: list[dict] = []
    args.raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.raw_jsonl.open("w") as raw:
        for i, case in enumerate(cases, 1):
            print(f"[{i:3d}/{len(cases)}] {case.case_id} ...")
            base_r = _evaluate_one(case, foundation, _gen(base_model), post_layer)
            lora_r = _evaluate_one(case, foundation, _gen(lora_model), post_layer)
            base_r["arm"] = "base"
            lora_r["arm"] = "lora"
            base_results.append(base_r)
            lora_results.append(lora_r)
            raw.write(json.dumps(base_r, ensure_ascii=False) + "\n")
            raw.write(json.dumps(lora_r, ensure_ascii=False) + "\n")

    summary_base = _summarize(base_results, "base")
    summary_lora = _summarize(lora_results, "lora")
    _markdown_report(summary_base, summary_lora, args.output)

    print()
    print("=== Summary ===")
    print(f"BASE: {summary_base}")
    print(f"LORA: {summary_lora}")
    print(f"\nReport → {args.output}")
    print(f"Raw    → {args.raw_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
