"""CLI runner for Phase 2d.2 real-Qwen validation.

Run from the repo root::

    python -m experiments.agi.phase_2_validation.run_validation
    python -m experiments.agi.phase_2_validation.run_validation --cases-subset known
    python -m experiments.agi.phase_2_validation.run_validation --cases-limit 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from agi.foundation import FrozenFoundation  # noqa: E402
from agi.metacognition.training import load_checkpoint  # noqa: E402

# Relative imports work when invoked as a module; we also support a
# direct ``python <path>`` invocation by patching sys.path above.
try:  # ``python -m experiments...``
    from .pipeline import run_validation_case
    from .analysis import generate_report
    from .test_cases import ALL_CASES
except ImportError:  # ``python experiments/.../run_validation.py``
    sys.path.insert(0, str(_REPO_ROOT))
    from experiments.agi.phase_2_validation.pipeline import run_validation_case  # noqa: E402
    from experiments.agi.phase_2_validation.analysis import generate_report  # noqa: E402
    from experiments.agi.phase_2_validation.test_cases import ALL_CASES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Phase 2d.2 real-Qwen generalisation test. "
            "Writes a markdown report to "
            "results/agi/phase_2_validation_report.md by default."
        )
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("data/metacog/checkpoints"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/agi/phase_2_validation_report.md"),
    )
    parser.add_argument(
        "--results-jsonl", type=Path,
        default=Path("results/agi/phase_2_validation_raw.jsonl"),
        help=(
            "Where to dump per-case raw results (one JSON object "
            "per line). Useful for downstream re-analysis without "
            "rerunning Qwen."
        ),
    )
    parser.add_argument(
        "--cases-subset", type=str, default=None,
        help="Filter case_ids by prefix (e.g. 'known', 'halluc_00').",
    )
    parser.add_argument(
        "--cases-limit", type=int, default=None,
        help="Cap on the number of cases (after subset filtering).",
    )
    args = parser.parse_args()

    print("Loading Qwen2.5-1.5B-Instruct foundation...")
    t_load = time.perf_counter()
    foundation = FrozenFoundation()
    print(f"Loaded in {time.perf_counter() - t_load:.1f}s.")

    print("Loading metacog PRE + POST checkpoints...")
    pre_layer = load_checkpoint(
        args.checkpoint_dir / "pre_layer.pt", mode="pre",
    )
    post_layer = load_checkpoint(
        args.checkpoint_dir / "post_layer.pt", mode="post",
    )

    cases = list(ALL_CASES)
    if args.cases_subset:
        cases = [c for c in cases if c.case_id.startswith(args.cases_subset)]
    if args.cases_limit:
        cases = cases[: args.cases_limit]
    print(f"Running {len(cases)} cases.\n")

    results: list[dict] = []
    args.results_jsonl.parent.mkdir(parents=True, exist_ok=True)
    raw_handle = args.results_jsonl.open("w")
    t_total = time.perf_counter()
    for i, case in enumerate(cases, start=1):
        t_case = time.perf_counter()
        print(
            f"[{i:3d}/{len(cases)}] {case.case_id}  "
            f"({case.expected_status:>13}) ... ",
            end="",
            flush=True,
        )
        try:
            result = run_validation_case(
                case, foundation, pre_layer, post_layer,
            )
            dt = time.perf_counter() - t_case
            print(
                f"{dt:5.1f}s  "
                f"pre→{result['pre_predicted_status']:<13} "
                f"post→{result['post_predicted_status']:<13}"
            )
        except Exception as exc:  # noqa: BLE001 — keep going
            result = {
                "case_id": case.case_id,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"ERROR: {exc}")
        results.append(result)
        raw_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        raw_handle.flush()
    raw_handle.close()

    wall = time.perf_counter() - t_total
    print(f"\nTotal wall: {wall:.1f}s")

    generate_report(results, args.output, wall_seconds=wall)
    print(f"Report → {args.output}")
    print(f"Raw    → {args.results_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
