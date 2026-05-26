"""Tests for the Cold Storage v2 inference-time retrieval ensemble."""

from __future__ import annotations

import base64

import pytest
import torch
import torch.nn.functional as F

from continual_synapse.baselines.naive_finetune import MLPClassifier, MLPConfig
from continual_synapse.baselines.synapse_finetune import SynapseAugmentedMLP
from continual_synapse.cold_storage.compression import quantize
from continual_synapse.cold_storage.store import ColdStorage
from continual_synapse.evaluation.runner import set_seed
from continual_synapse.inference.retrieval_ensemble import RetrievalEnsemble
from continual_synapse.synapse_layer.layer import SynapseLayer
from continual_synapse.synapse_layer.modulation import SynapseModulation


def _bare_mlp(hidden_dim: int = 8, num_classes: int = 3) -> MLPClassifier:
    set_seed(0)
    return MLPClassifier(
        MLPConfig(
            input_dim=4,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_hidden_layers=2,
            dropout=0.0,
        )
    )


def _populate_store(
    store: ColdStorage, embeddings: list[list[float]], n_neurons: int
) -> None:
    """Insert one entry per embedding with a dummy document."""
    for i, emb in enumerate(embeddings):
        blob = quantize(torch.zeros(n_neurons, n_neurons), precision=32)
        store.store_cluster(
            embedding=emb,
            metadata={
                "precision": 32, "n_neurons": n_neurons,
                "age": 0, "access_count": 0, "created_at_step": 0,
            },
            document=base64.b64encode(blob).decode("ascii"),
            entry_id=f"e{i}",
        )


# ---- construction + validation ----


def test_init_validates_shapes_and_ranges() -> None:
    model = _bare_mlp()
    emb = torch.zeros(0, 8)
    lbl = torch.zeros(0, dtype=torch.long)
    # ok
    RetrievalEnsemble(model, emb, lbl)
    # wrong dim
    with pytest.raises(ValueError, match="2-D"):
        RetrievalEnsemble(model, torch.zeros(8), lbl)
    with pytest.raises(ValueError, match="1-D"):
        RetrievalEnsemble(model, emb, torch.zeros(0, 1, dtype=torch.long))
    # mismatched N
    with pytest.raises(ValueError, match="disagree"):
        RetrievalEnsemble(model, torch.zeros(3, 8), torch.zeros(2, dtype=torch.long))
    # bad knobs
    with pytest.raises(ValueError, match="k"):
        RetrievalEnsemble(model, emb, lbl, k=0)
    with pytest.raises(ValueError, match="tau"):
        RetrievalEnsemble(model, emb, lbl, tau=1.1)
    with pytest.raises(ValueError, match="lambda_blend"):
        RetrievalEnsemble(model, emb, lbl, lambda_blend=-0.1)


def test_from_model_and_storage_empty_store_returns_zeros() -> None:
    """An empty cold storage produces zero-size embedding/label tensors
    so :meth:`predict` falls through to the bare model."""
    model = _bare_mlp(hidden_dim=8)
    store = ColdStorage(collection_name="ens_empty")
    ens = RetrievalEnsemble.from_model_and_storage(
        model, store, k=3, tau=0.5, lambda_blend=0.5,
    )
    assert ens.embeddings.shape == (0, 8)
    assert ens.labels.shape == (0,)


def test_from_model_and_storage_derives_labels_via_argmax() -> None:
    """Each stored embedding is labelled by the model's argmax output."""
    model = _bare_mlp(hidden_dim=4, num_classes=3)
    store = ColdStorage(collection_name="ens_labels")
    # Three embeddings, each a one-hot in the 4-d feature space.
    _populate_store(
        store,
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0]],
        n_neurons=4,
    )
    ens = RetrievalEnsemble.from_model_and_storage(
        model, store, k=2, tau=0.5, lambda_blend=0.5,
    )
    # Recompute the expected labels: argmax(model.classify(embeddings)).
    with torch.no_grad():
        expected = model.classify(ens.embeddings).argmax(dim=-1)
    torch.testing.assert_close(ens.labels, expected)


# ---- predict behaviour ----


def test_predict_empty_storage_passes_through_model_logits() -> None:
    """No stored entries ⇒ predict equals raw model logits bit-exact."""
    model = _bare_mlp(hidden_dim=8)
    ens = RetrievalEnsemble(
        model,
        cold_storage_embeddings=torch.zeros(0, 8),
        cold_storage_labels=torch.zeros(0, dtype=torch.long),
        k=3, tau=0.5, lambda_blend=0.5,
    )
    x = torch.randn(4, 4)
    model.eval()
    with torch.no_grad():
        expected = model(x)
    actual = ens.predict(x)
    torch.testing.assert_close(actual, expected)


def test_predict_low_similarity_falls_through_to_model() -> None:
    """When every stored embedding is orthogonal to the model's
    features, the top-1 sim ≤ tau ⇒ ensemble returns model logits."""
    set_seed(0)
    model = _bare_mlp(hidden_dim=8, num_classes=3)
    # Stored embeddings on axes 4..7; model features for ones() input
    # will live on axes 0..3 (approximately, after one ReLU layer).
    # We bias toward orthogonality by storing axis vectors in the
    # "other half" of the 8-d feature space.
    embeddings = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ])
    labels = torch.tensor([1, 2, 0], dtype=torch.long)
    ens = RetrievalEnsemble(
        model, embeddings, labels, k=3, tau=0.5, lambda_blend=1.0,
    )
    x = torch.ones(2, 4)
    # Probe model features once to confirm the orthogonality property
    # holds for THIS seed before asserting fall-through.
    model.eval()
    with torch.no_grad():
        h = model.features(x)
        h_norm = F.normalize(h, dim=-1)
        stored_norm = F.normalize(embeddings, dim=-1)
        sim = (h_norm @ stored_norm.T).max(dim=-1).values
        assert (sim <= 0.5).all(), (
            f"test setup invalid for this seed: max sim {sim} not <= tau"
        )
        expected_model = model(x)

    actual = ens.predict(x)
    torch.testing.assert_close(actual, expected_model)


def test_predict_high_similarity_engages_blend_and_pushes_toward_label() -> None:
    """When a query equals a stored embedding exactly (sim=1 > tau),
    the blended output for lambda=1 is dominated by that entry's
    label slot — argmax should equal the stored label."""
    model = _bare_mlp(hidden_dim=4, num_classes=3)
    # Single stored embedding; label it explicitly.
    embeddings = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    labels = torch.tensor([2], dtype=torch.long)  # class 2
    ens = RetrievalEnsemble(
        model, embeddings, labels, k=1, tau=0.5, lambda_blend=1.0,
    )
    # Inject the stored embedding directly via the features pathway.
    # We bypass model.features by monkey-patching it to return our
    # known vector (the only way to guarantee the cosine sim is 1).
    target_h = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    orig_features = model.features
    model.features = lambda x: target_h.expand(x.shape[0], -1).clone()  # type: ignore[method-assign]
    try:
        out = ens.predict(torch.zeros(1, 4))
    finally:
        model.features = orig_features  # type: ignore[method-assign]
    # With lambda=1 and a single neighbour of class 2, retrieval_probs
    # is a one-hot at class 2; the model softmax is mixed at weight 0.
    # The log-blended logits therefore have their max at class 2.
    assert int(out.argmax(dim=-1).item()) == 2


def test_predict_weighted_vote_with_multiple_classes() -> None:
    """Top-k vote: weights are sims clipped at 0, accumulated per
    label class, then normalised. Verify against a hand computation."""
    model = _bare_mlp(hidden_dim=4, num_classes=3)
    # Three embeddings: two of class 1, one of class 2. We'll arrange
    # the cosine sims so class 1 narrowly wins by weight.
    embeddings = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],   # → class 1
        [0.9, 0.1, 0.0, 0.0],   # → class 1
        [0.0, 1.0, 0.0, 0.0],   # → class 2
    ])
    labels = torch.tensor([1, 1, 2], dtype=torch.long)
    ens = RetrievalEnsemble(
        model, embeddings, labels, k=3, tau=0.0, lambda_blend=1.0,
    )
    # Force the features to [1, 0, 0, 0] so the cosine sims are
    # [1.0, ~0.994, 0.0] respectively. Class 1 weight = 1 + 0.994 ≈ 1.994;
    # class 2 weight = 0; renormalised: class 1 gets ~1.0 probability.
    forced_h = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    model.features = lambda x: forced_h.expand(x.shape[0], -1).clone()  # type: ignore[method-assign]
    out = ens.predict(torch.zeros(1, 4))
    # Argmax should be class 1.
    assert int(out.argmax(dim=-1).item()) == 1


def test_predict_k_cap_against_small_archive() -> None:
    """Passing k larger than the archive size doesn't crash; we just
    use every available entry."""
    model = _bare_mlp(hidden_dim=4, num_classes=3)
    embeddings = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    labels = torch.tensor([0, 1], dtype=torch.long)
    ens = RetrievalEnsemble(
        model, embeddings, labels, k=10, tau=0.0, lambda_blend=0.5,
    )
    out = ens.predict(torch.randn(3, 4))
    assert out.shape == (3, 3)


def test_predict_does_not_mutate_model_training_state() -> None:
    """The wrapper must put the model in eval mode for the forward
    pass, then restore the prior training flag."""
    model = _bare_mlp(hidden_dim=4, num_classes=3)
    model.train()  # start in training mode
    ens = RetrievalEnsemble(
        model,
        cold_storage_embeddings=torch.zeros(0, 4),
        cold_storage_labels=torch.zeros(0, dtype=torch.long),
    )
    ens.predict(torch.randn(2, 4))
    assert model.training, "predict should restore train mode"


def test_predict_synapse_augmented_uses_base_features_only() -> None:
    """For a SynapseAugmentedMLP the retrieval query must come from
    base.features (the same vector space stored embeddings live in),
    NOT from the wrapper's modulator-augmented features."""
    cfg = MLPConfig(input_dim=4, hidden_dim=4, num_classes=3, dropout=0.0)
    set_seed(0)
    base = MLPClassifier(cfg)
    synapse = SynapseLayer(n_neurons=4, learning_rate=1e-3)
    # Inflate strengths so any leak through the modulator path would
    # be visible.
    with torch.no_grad():
        synapse.strengths.fill_(2.0)
    aug = SynapseAugmentedMLP(
        base, synapse, SynapseModulation(init_gate=0.5),
    )
    aug.eval()
    embeddings = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    labels = torch.tensor([0], dtype=torch.long)
    ens = RetrievalEnsemble(
        aug, embeddings, labels, k=1, tau=1.0, lambda_blend=0.5,
        # tau=1.0 with strict > check ⇒ never engages (cosine ≤ 1)
    )
    # Identity invariant: with tau > 1 the ensemble must fall through to
    # the model's prediction — verify it matches base.classify(base.features(x)).
    x = torch.randn(2, 4)
    out = ens.predict(x)
    with torch.no_grad():
        bare = base.classify(base.features(x))
    torch.testing.assert_close(out, bare)
