"""Experiment 29 — Pretrain the frozen contrastive encoder (Option B).

Self-supervised pretraining of a permutation-invariant MNIST
encoder. The objective: any two random permutations of the same
MNIST image should produce nearby features; different images
should produce distinct features. This trains the encoder by
construction to be invariant to the transform that defines the
Permuted-MNIST continual benchmark.

After training, the encoder is saved to ``--output-path`` and
loaded as a frozen keying function for the dual-substrate eval
(:mod:`experiments/28_episodic_dual_substrate_eval`). The
projection head is discarded — only the encoder's penultimate
output is used downstream.

Run from the repo root::

    python experiments/29_pretrain_contrastive_encoder.py --epochs 50

The script ends with three sanity-check assertions:

- Linear-probe accuracy on MNIST test ≥ ``--probe-floor`` (default
  0.90; MNIST is easy so 0.95+ is the expected ballpark)
- Same-digit cross-permutation similarity ≥ ``--same-floor``
  (default 0.50) — verifies the contrastive objective actually
  pulled augmented views together
- (same-digit minus different-digit) gap ≥ ``--gap-floor`` (default
  0.20) — verifies the encoder discriminates between digit
  classes, not just that all features collapsed to one point

Any failure raises ``AssertionError`` at the end. The checkpoint
is still written to disk before the assertions run, so a failing
encoder is available for inspection.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.episodic.contrastive_encoder import (  # noqa: E402
    ContrastiveEncoder,
    apply_permutation,
    info_nce_loss,
    random_permutation,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--feature-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--projection-dim", type=int, default=64)
    p.add_argument(
        "--output-path", type=Path,
        default=_REPO_ROOT / "results" / "pretrained" / "contrastive_encoder.pt",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--cache-dir", default=str(_REPO_ROOT / "data" / "hf_cache")
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda", "mps"],
    )
    # ---- Probe + sanity-check parameters ----
    p.add_argument(
        "--probe-epochs", type=int, default=20,
        help="Epochs to train the linear probe on top of the frozen "
             "encoder for the sanity check.",
    )
    p.add_argument(
        "--probe-floor", type=float, default=0.90,
        help="Minimum linear-probe accuracy required to consider the "
             "encoder usable. MNIST is easy enough that 0.95+ is "
             "achievable; the floor is set lower to allow some "
             "tuning slack.",
    )
    p.add_argument(
        "--same-floor", type=float, default=0.50,
        help="Minimum mean cosine similarity for same-digit cross-"
             "permutation pairs.",
    )
    p.add_argument(
        "--gap-floor", type=float, default=0.20,
        help="Minimum margin between same-digit and different-digit "
             "mean similarities. Guards against representation "
             "collapse where everything maps to one cluster.",
    )
    return p.parse_args()


# ---------- data loading ----------


def _load_mnist_tensors(
    cache_dir: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (train_x, train_y, test_x, test_y) as float tensors.

    Images are flattened to ``(N, 784)`` in ``[0, 1]``. Labels are
    int64. Uses the same HuggingFace ``ylecun/mnist`` source that
    :class:`SplitMNIST` does, so we share the dataset cache.
    """
    from datasets import load_dataset  # type: ignore[import-untyped]
    import numpy as np

    ds = load_dataset("ylecun/mnist", cache_dir=cache_dir)

    def _split(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        split = ds[name]
        images = np.stack(
            [np.asarray(im, dtype=np.uint8) for im in split["image"]]
        )
        labels = np.asarray(split["label"], dtype=np.int64)
        x = torch.from_numpy(images).to(torch.float32).view(-1, 784) / 255.0
        y = torch.from_numpy(labels)
        return x, y

    train_x, train_y = _split("train")
    test_x, test_y = _split("test")
    return train_x, train_y, test_x, test_y


# ---------- pretraining loop ----------


def _train(
    encoder: ContrastiveEncoder,
    train_x: torch.Tensor,
    args: argparse.Namespace,
) -> list[float]:
    """Train the contrastive encoder. Returns per-epoch mean loss."""
    encoder.train()
    optim = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    loader = DataLoader(
        TensorDataset(train_x),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    epoch_losses: list[float] = []
    for epoch in range(args.epochs):
        t0 = time.time()
        running = 0.0
        n_batches = 0
        for (x,) in loader:
            x = x.to(args.device)
            # Two independent permutations per batch (SimCLR
            # convention: same perm applied to all rows of a batch,
            # but the two augmented views see different perms).
            perms = random_permutation(
                dim=x.shape[-1], n=2, device=args.device,
            )
            x1 = apply_permutation(x, perms[0])
            x2 = apply_permutation(x, perms[1])
            _, z1 = encoder(x1)
            _, z2 = encoder(x2)
            loss = info_nce_loss(z1, z2, temperature=args.temperature)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += float(loss.item())
            n_batches += 1
        mean_loss = running / max(1, n_batches)
        epoch_losses.append(mean_loss)
        print(
            f"  epoch {epoch + 1:3d}/{args.epochs}  "
            f"loss={mean_loss:.4f}  "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )
    return epoch_losses


# ---------- sanity checks ----------


def _linear_probe_accuracy(
    encoder: ContrastiveEncoder,
    train_x: torch.Tensor, train_y: torch.Tensor,
    test_x: torch.Tensor, test_y: torch.Tensor,
    args: argparse.Namespace,
) -> float:
    """Train a single linear layer on top of frozen encoder features.

    The encoder's weights stay fixed; only the linear classifier
    learns. After training, evaluate on the held-out test set and
    return the accuracy. This is the standard SimCLR-style probe:
    if features are well-shaped, even a linear head should hit
    high accuracy on MNIST.
    """
    encoder.eval()
    # Extract features once (no encoder update).
    with torch.no_grad():
        train_feats = _extract_features(encoder, train_x, args)
        test_feats = _extract_features(encoder, test_x, args)
    classifier = nn.Linear(
        encoder.feature_dim, int(train_y.max().item()) + 1,
    ).to(args.device)
    optim = torch.optim.Adam(classifier.parameters(), lr=1e-2)
    loader = DataLoader(
        TensorDataset(train_feats, train_y),
        batch_size=args.batch_size, shuffle=True,
    )
    for epoch in range(args.probe_epochs):
        for fx, fy in loader:
            fx = fx.to(args.device)
            fy = fy.to(args.device)
            logits = classifier(fx)
            loss = F.cross_entropy(logits, fy)
            optim.zero_grad()
            loss.backward()
            optim.step()
    with torch.no_grad():
        test_logits = classifier(test_feats.to(args.device))
        preds = test_logits.argmax(dim=-1).cpu()
        return float((preds == test_y).float().mean().item())


def _extract_features(
    encoder: ContrastiveEncoder, x: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Run encoder.encode in chunks of args.batch_size."""
    encoder.eval()
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, x.shape[0], args.batch_size):
            batch = x[i : i + args.batch_size].to(args.device)
            chunks.append(encoder.encode(batch).cpu())
    return torch.cat(chunks, dim=0)


def _cross_permutation_similarity(
    encoder: ContrastiveEncoder,
    test_x: torch.Tensor, test_y: torch.Tensor,
    args: argparse.Namespace,
    n_pairs_per_class: int = 200,
    n_diff_pairs: int = 2000,
) -> tuple[float, float]:
    """Measure how well the encoder distinguishes same-digit pairs
    from different-digit pairs under two random permutations.

    Procedure:

    - Same-digit: for each class ``c``, sample two images from class
      ``c``, apply two independent permutations, encode both, and
      record their cosine similarity. Repeat ``n_pairs_per_class``
      times per class; average across all classes.
    - Different-digit: sample two images from different classes,
      apply two random permutations, encode, record. Repeat
      ``n_diff_pairs`` times.

    Returns ``(mean_same, mean_diff)``. If the contrastive objective
    learned the right thing, ``mean_same`` should be high (≥ 0.5)
    and the gap ``mean_same - mean_diff`` should be substantial
    (≥ 0.2).
    """
    encoder.eval()
    torch.manual_seed(0)  # deterministic similarity numbers
    num_classes = int(test_y.max().item()) + 1
    by_class = {c: torch.where(test_y == c)[0] for c in range(num_classes)}

    def _encode_pair(idx_a: int, idx_b: int) -> tuple[float]:
        xa = test_x[idx_a].unsqueeze(0).to(args.device)
        xb = test_x[idx_b].unsqueeze(0).to(args.device)
        perms = random_permutation(dim=xa.shape[-1], n=2, device=args.device)
        fa = encoder.encode(apply_permutation(xa, perms[0]))
        fb = encoder.encode(apply_permutation(xb, perms[1]))
        return float(
            F.cosine_similarity(
                F.normalize(fa, dim=-1),
                F.normalize(fb, dim=-1),
                dim=-1,
            ).item()
        )

    same_sims: list[float] = []
    with torch.no_grad():
        for c in range(num_classes):
            idxs = by_class[c]
            if idxs.numel() < 2:
                continue
            for _ in range(n_pairs_per_class):
                pair = idxs[torch.randperm(idxs.numel())[:2]]
                same_sims.append(_encode_pair(int(pair[0]), int(pair[1])))

        diff_sims: list[float] = []
        for _ in range(n_diff_pairs):
            ca, cb = torch.randperm(num_classes)[:2]
            ia = by_class[int(ca)][torch.randint(0, by_class[int(ca)].numel(), (1,))]
            ib = by_class[int(cb)][torch.randint(0, by_class[int(cb)].numel(), (1,))]
            diff_sims.append(_encode_pair(int(ia), int(ib)))

    mean_same = float(sum(same_sims) / max(1, len(same_sims)))
    mean_diff = float(sum(diff_sims) / max(1, len(diff_sims)))
    return mean_same, mean_diff


# ---------- main ----------


def main() -> None:
    args = parse_args()
    args.output_path = Path(args.output_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    print(
        f"Contrastive pretraining (Option B):\n"
        f"  epochs={args.epochs}  batch_size={args.batch_size}  "
        f"lr={args.lr}  temperature={args.temperature}\n"
        f"  feature_dim={args.feature_dim}  "
        f"hidden_dim={args.hidden_dim}  "
        f"projection_dim={args.projection_dim}\n"
        f"  device={args.device}  output={args.output_path}",
        flush=True,
    )

    train_x, train_y, test_x, test_y = _load_mnist_tensors(args.cache_dir)
    print(
        f"  loaded MNIST: train={tuple(train_x.shape)}  "
        f"test={tuple(test_x.shape)}",
        flush=True,
    )

    encoder = ContrastiveEncoder(
        input_dim=train_x.shape[-1],
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        projection_dim=args.projection_dim,
    ).to(args.device)

    print("\n=== Pretraining ===", flush=True)
    epoch_losses = _train(encoder, train_x, args)

    # Save the checkpoint BEFORE the sanity checks, so a failing
    # encoder is on disk for offline inspection.
    payload = {
        "state_dict": encoder.state_dict(),
        "config": encoder.config,
        "epoch_losses": epoch_losses,
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
    }
    torch.save(payload, args.output_path)
    print(f"\nWrote encoder checkpoint to {args.output_path}", flush=True)

    print("\n=== Sanity checks ===", flush=True)
    probe_acc = _linear_probe_accuracy(
        encoder, train_x, train_y, test_x, test_y, args,
    )
    print(
        f"  linear-probe MNIST test accuracy: {probe_acc:.4f}  "
        f"(floor {args.probe_floor:.2f})",
        flush=True,
    )

    mean_same, mean_diff = _cross_permutation_similarity(
        encoder, test_x, test_y, args,
    )
    gap = mean_same - mean_diff
    print(
        f"  same-digit cross-perm sim: {mean_same:.4f}  "
        f"(floor {args.same_floor:.2f})",
        flush=True,
    )
    print(
        f"  different-digit cross-perm sim: {mean_diff:.4f}",
        flush=True,
    )
    print(
        f"  gap (same − different): {gap:.4f}  "
        f"(floor {args.gap_floor:.2f})",
        flush=True,
    )

    # Hard assertions — the checkpoint is already saved, so failure
    # surfaces clearly but doesn't lose the artifact.
    failures: list[str] = []
    if probe_acc < args.probe_floor:
        failures.append(
            f"probe_acc {probe_acc:.4f} < floor {args.probe_floor:.2f}"
        )
    if mean_same < args.same_floor:
        failures.append(
            f"same-digit similarity {mean_same:.4f} < floor "
            f"{args.same_floor:.2f}"
        )
    if gap < args.gap_floor:
        failures.append(
            f"same-vs-different gap {gap:.4f} < floor {args.gap_floor:.2f}"
        )
    if failures:
        raise AssertionError(
            "Pretrained encoder failed sanity checks:\n  "
            + "\n  ".join(failures)
            + f"\nThe checkpoint is still saved at {args.output_path} "
            f"for inspection; do not deploy it without re-running."
        )
    print("\n✓ Encoder passes all sanity checks.", flush=True)


if __name__ == "__main__":
    main()
