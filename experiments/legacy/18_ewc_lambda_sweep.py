"""Experiment 18 — EWC λ sweep on Permuted-MNIST 15-task.

Phase 1 of a two-phase EWC tuning protocol:
- Phase 1 (this script): scan λ ∈ {1, 10, 100, 500, 1000, 5000} at
  5 seeds each. Identify the λ that maximises mean ACC.
- Phase 2 (separate experiment): re-run that λ at 10 seeds for a
  fair head-to-head against cs_gated_cosine (also at n=10) from
  exp 17.

Why a sweep is needed: exp 17 used λ=1000 because that's the value
exp 06 / exp 02 had been using on Split-MNIST. Permuted-MNIST has
a very different gradient regime (15 tasks vs 5, much more inter-
task interference, much larger Fisher norms), so λ=1000 may be
either far too strong (over-regularises and prevents learning new
tasks) or far too weak (under-protects old tasks). The sweep lets
us know for sure before claiming cs_gated_cosine beats EWC.

Methods: one per λ, named ``ewc_lam_<value>``, all otherwise
identical (Fisher sample size = 500, 1 epoch/task, dropout=0.5,
shared 10-class head). Same training hyperparameters and benchmark
construction as exp 17's Permuted-MNIST half.

Per-method JSON checkpointing under
``results/logs/ewc_lambda_sweep/<ts>_18_ewc_lambda_sweep_T15.json``
with atomic .tmp + os.replace. Each completed λ updates the
checkpoint so a kill mid-sweep preserves prior λs' results.

Reports mean ± std ACC and FGT per λ. Identifies the
argmax-by-mean-ACC value and prints it explicitly at the end
of the run for easy parse-out.

Run from the repo root::

    python experiments/18_ewc_lambda_sweep.py --num-tasks 15 \\
        --seeds 0 1 2 3 4 --lambdas 1 10 100 500 1000 5000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.baselines.ewc import EWC  # noqa: E402
from continual_synapse.baselines.naive_finetune import MLPClassifier, MLPConfig  # noqa: E402
from continual_synapse.evaluation.benchmarks import PermutedMNIST  # noqa: E402
from continual_synapse.evaluation.multi_seed import MultiSeedRun, run_multi_seed  # noqa: E402
from continual_synapse.evaluation.reporting import compute_metrics  # noqa: E402
from continual_synapse.evaluation.runner import ContinualRunner, set_seed  # noqa: E402
from continual_synapse.evaluation.statistics import (  # noqa: E402
    format_summary_table,
    summarise_method,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--num-tasks", type=int, default=15)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument(
        "--lambdas", type=float, nargs="+",
        default=[1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0],
        help="λ values to sweep. Each becomes one 'method' named "
             "ewc_lam_<value> in the checkpoint.",
    )
    # ---- Training hyperparameters (mirror exp 17 Permuted-MNIST) ----
    p.add_argument("--epochs-per-task", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-hidden-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--ewc-fisher-samples", type=int, default=500)
    p.add_argument("--permutation-seed", type=int, default=42)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda", "mps"],
    )
    p.add_argument(
        "--cache-dir", default=str(_REPO_ROOT / "data" / "hf_cache")
    )
    p.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "results" / "logs" / "ewc_lambda_sweep"),
    )
    return p.parse_args()


def _lam_method_name(lam: float) -> str:
    """Format ``lam`` into a method-name suffix that's safe for filenames."""
    if float(lam).is_integer():
        return f"ewc_lam_{int(lam)}"
    return f"ewc_lam_{lam:g}"


def _build_mlp(args, num_classes: int) -> MLPClassifier:
    return MLPClassifier(
        MLPConfig(
            input_dim=784,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_hidden_layers=args.num_hidden_layers,
            dropout=args.dropout,
        )
    )


def _build_factories(
    args, num_classes: int
) -> tuple[dict[str, Callable], dict[str, list[dict]], list[str]]:
    diagnostics: dict[str, list[dict]] = {}
    method_names: list[str] = []
    factories: dict[str, Callable] = {}

    runner_kwargs = dict(
        optimizer_factory=lambda p: torch.optim.SGD(
            p, lr=args.lr, momentum=args.momentum
        ),
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        eval_batch_size=256,
        device=args.device,
        record_zero_shot=True,
    )

    for lam in args.lambdas:
        name = _lam_method_name(lam)
        method_names.append(name)
        diagnostics[name] = []

        def make_factory(lam_value: float, method_key: str) -> Callable:
            def factory(seed: int):
                set_seed(seed)
                model = _build_mlp(args, num_classes)
                e = EWC(
                    lam=lam_value,
                    fisher_sample_size=args.ewc_fisher_samples,
                    device=args.device,
                )
                per_task_diag: list[dict] = []

                def on_task_end(i, task, m):
                    e.consolidate(m, task.train)
                    per_task_diag.append(
                        {
                            "task_index": int(i),
                            "num_consolidated_tasks": int(
                                e.num_consolidated_tasks
                            ),
                        }
                    )

                runner = ContinualRunner(
                    seed=seed,
                    regulariser=e.penalty,
                    on_task_end=on_task_end,
                    **runner_kwargs,
                )
                diagnostics[method_key].append(
                    {"seed": seed, "per_task": per_task_diag}
                )
                return model, runner

            return factory

        factories[name] = make_factory(float(lam), name)

    return factories, diagnostics, method_names


def _multi_seed_to_jsonable(run: MultiSeedRun) -> dict:
    out: dict = {"method": run.method, "seeds": run.seeds, "results": []}
    for r in run.results:
        summary = compute_metrics(r)
        out["results"].append(
            {
                "benchmark": r.benchmark,
                "task_names": r.task_names,
                "accuracy_matrix": [
                    [None if math.isnan(v) else float(v) for v in row]
                    for row in r.accuracy_matrix
                ],
                "random_baseline": r.random_baseline.tolist(),
                "metrics": asdict(summary),
            }
        )
    return out


def _build_payload(
    args: argparse.Namespace,
    ts: int,
    runs: list[MultiSeedRun],
    summaries: list,
    diagnostics: dict[str, list[dict]],
    method_times: dict[str, float],
    *,
    is_partial: bool,
    methods_completed: list[str],
    methods_requested: list[str],
    lambdas: list[float],
) -> dict[str, Any]:
    return {
        "experiment": "18_ewc_lambda_sweep",
        "timestamp": ts,
        "config": vars(args),
        "lambdas": list(lambdas),
        "is_partial": is_partial,
        "methods_completed": list(methods_completed),
        "methods_requested": list(methods_requested),
        "methods": [_multi_seed_to_jsonable(r) for r in runs],
        "summaries": [
            {
                "method": s.method,
                "n_seeds": s.n_seeds,
                "metric_means": s.metric_means,
                "metric_stds": s.metric_stds,
                "per_seed_metrics": s.per_seed_metrics,
            }
            for s in summaries
        ],
        "diagnostics": diagnostics,
        "method_times_seconds": method_times,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    if len(args.seeds) < 2:
        raise SystemExit("multi-seed experiment needs at least 2 seeds")
    if len(args.lambdas) < 1:
        raise SystemExit("--lambdas must include at least one value")

    print(
        f"Loading PermutedMNIST with {args.num_tasks} tasks "
        f"(permutation seed={args.permutation_seed}, dropout={args.dropout})..."
    )
    bench = PermutedMNIST.from_huggingface(
        num_tasks=args.num_tasks,
        seed=args.permutation_seed,
        cache_dir=args.cache_dir,
    )
    factories, diagnostics, method_names = _build_factories(
        args, num_classes=bench.num_classes_per_task
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = output_dir / f"{ts}_18_ewc_lambda_sweep_T{args.num_tasks}.json"
    print(f"Checkpoint path: {path}", flush=True)
    print(
        f"Sweeping λ ∈ {sorted(args.lambdas)} at "
        f"{len(args.seeds)} seeds each.",
        flush=True,
    )

    runs: list[MultiSeedRun] = []
    summaries: list = []
    method_times: dict[str, float] = {}
    methods_completed: list[str] = []

    for method in method_names:
        t0 = time.time()

        def _seed_done(m, i, n, seed, result):
            s = compute_metrics(result)
            elapsed = time.time() - t0
            print(
                f"    {m} seed {i + 1}/{n} (seed={seed}, "
                f"{elapsed:.0f}s into method): "
                f"acc={s.average_accuracy:.3f}, "
                f"fgt={s.average_forgetting:+.3f}",
                flush=True,
            )

        run = run_multi_seed(
            method,
            factories[method],
            bench,
            seeds=args.seeds,
            progress=lambda m, i, n: print(
                f"  {m}: seed {i + 1}/{n}", flush=True
            ),
            on_seed_complete=_seed_done,
        )
        elapsed = time.time() - t0
        method_times[method] = elapsed
        runs.append(run)
        summaries.append(summarise_method(run))
        methods_completed.append(method)
        print(f"{method} finished in {elapsed:.1f}s", flush=True)

        is_partial = len(methods_completed) < len(method_names)
        payload = _build_payload(
            args, ts, runs, summaries, diagnostics, method_times,
            is_partial=is_partial,
            methods_completed=methods_completed,
            methods_requested=method_names,
            lambdas=args.lambdas,
        )
        _atomic_write_json(path, payload)
        tag = "partial" if is_partial else "final"
        print(
            f"  Checkpoint written ({tag}, "
            f"{len(methods_completed)}/{len(method_names)} λ values): {path}",
            flush=True,
        )

    # ---------- End-of-run summary ----------
    print()
    print(
        f"Per-λ mean ± std (n={len(args.seeds)} seeds, "
        f"PERMUTED-MNIST {args.num_tasks} tasks, dropout={args.dropout}):"
    )
    print(format_summary_table(summaries))

    # ---------- Identify the best λ by mean ACC ----------
    if summaries:
        best = max(
            summaries,
            key=lambda s: s.metric_means.get("average_accuracy", float("-inf")),
        )
        best_lam_value = None
        for lam in args.lambdas:
            if _lam_method_name(lam) == best.method:
                best_lam_value = lam
                break
        print()
        print("=" * 70)
        print("BEST λ (by mean ACC):")
        print(f"  method: {best.method}")
        if best_lam_value is not None:
            print(f"  λ:      {best_lam_value:g}")
        print(
            f"  ACC:    {best.metric_means.get('average_accuracy', float('nan')):.4f} "
            f"± {best.metric_stds.get('average_accuracy', float('nan')):.4f}"
        )
        print(
            f"  FGT:    {best.metric_means.get('average_forgetting', float('nan')):+.4f} "
            f"± {best.metric_stds.get('average_forgetting', float('nan')):.4f}"
        )
        print("=" * 70)

    print(f"Saved run log to {path}")


if __name__ == "__main__":
    main()
