"""Tests for reconstructive retrieval from cold storage."""

from __future__ import annotations

import base64

import pytest
import torch

from continual_synapse.cold_storage.compression import quantize
from continual_synapse.cold_storage.store import ColdStorage
from continual_synapse.consolidation.reconstruction import (
    fetch_entries_for_query,
    reconstruct_strengths,
)


def _store_strengths(
    store: ColdStorage,
    embedding: list[float],
    strengths: torch.Tensor,
    precision: int = 32,
    entry_id: str | None = None,
) -> str:
    blob = quantize(strengths, precision=precision)
    doc = base64.b64encode(blob).decode("ascii")
    meta = {
        "precision": precision,
        "n_neurons": int(strengths.shape[0]),
        "age": 0,
        "access_count": 0,
        "created_at_step": 0,
    }
    return store.store_cluster(
        embedding=embedding,
        metadata=meta,
        document=doc,
        entry_id=entry_id,
    )


def test_empty_store_returns_zeros() -> None:
    store = ColdStorage(collection_name="recon_empty")
    out = reconstruct_strengths(
        store, torch.zeros(3), k=4, n_neurons=3
    )
    assert torch.all(out == 0)
    assert out.shape == (3, 3)


def test_single_entry_recovered_at_full_precision() -> None:
    store = ColdStorage(collection_name="recon_single")
    s = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    _store_strengths(store, embedding=[1.0, 0.0], strengths=s, entry_id="x")
    out = reconstruct_strengths(
        store, torch.tensor([1.0, 0.0]), k=4, n_neurons=2
    )
    # One match → weight is 1 / (1 + d) / 1 = 1; matrix recovered exactly.
    torch.testing.assert_close(out, s)


def test_context_dependent_retrieval_returns_different_patterns() -> None:
    """Two entries with different keys; queries near each return different
    reconstructions. This is the property that should resolve the
    Phase-3.5 universal-correction conflict."""
    store = ColdStorage(collection_name="recon_context")
    s_a = torch.full((3, 3), 1.0)
    s_b = torch.full((3, 3), -1.0)
    _store_strengths(store, embedding=[1.0, 0.0, 0.0], strengths=s_a, entry_id="a")
    _store_strengths(store, embedding=[0.0, 1.0, 0.0], strengths=s_b, entry_id="b")

    # Query close to A → reconstruction strongly weighted toward s_a.
    out_a = reconstruct_strengths(
        store, torch.tensor([0.9, 0.1, 0.0]), k=2, n_neurons=3
    )
    out_b = reconstruct_strengths(
        store, torch.tensor([0.1, 0.9, 0.0]), k=2, n_neurons=3
    )
    # Different contexts give clearly different outputs.
    assert (out_a > 0).any() and (out_a.mean() > 0)
    assert (out_b < 0).any() and (out_b.mean() < 0)


def test_weighting_uniform_averages_equally() -> None:
    store = ColdStorage(collection_name="recon_uniform")
    s_a = torch.full((2, 2), 2.0)
    s_b = torch.full((2, 2), 4.0)
    _store_strengths(store, embedding=[1.0, 0.0], strengths=s_a)
    _store_strengths(store, embedding=[10.0, 10.0], strengths=s_b)
    # Query is far from both; under uniform weighting each gets weight 1/2.
    out = reconstruct_strengths(
        store,
        torch.tensor([5.0, 5.0]),
        k=2,
        n_neurons=2,
        weighting="uniform",
    )
    torch.testing.assert_close(out, (s_a + s_b) / 2.0)


def test_bump_access_count_updates_metadata() -> None:
    store = ColdStorage(collection_name="recon_bump")
    _store_strengths(
        store, embedding=[1.0], strengths=torch.zeros(1, 1), entry_id="bumpy"
    )
    reconstruct_strengths(
        store, torch.tensor([1.0]), k=1, n_neurons=1
    )
    e = store.get_by_id("bumpy")
    assert e.metadata["access_count"] == 1
    # Second retrieval bumps to 2.
    reconstruct_strengths(
        store, torch.tensor([1.0]), k=1, n_neurons=1
    )
    e = store.get_by_id("bumpy")
    assert e.metadata["access_count"] == 2


def test_bump_access_count_can_be_disabled() -> None:
    store = ColdStorage(collection_name="recon_no_bump")
    _store_strengths(
        store, embedding=[1.0], strengths=torch.zeros(1, 1), entry_id="quiet"
    )
    reconstruct_strengths(
        store,
        torch.tensor([1.0]),
        k=1,
        n_neurons=1,
        bump_access_count=False,
    )
    e = store.get_by_id("quiet")
    assert e.metadata["access_count"] == 0


def test_low_precision_round_trip_is_approximate() -> None:
    """A 4-bit-archived entry returns a close but not exact matrix."""
    g = torch.Generator().manual_seed(0)
    store = ColdStorage(collection_name="recon_4bit")
    s = torch.randn(3, 3, generator=g)
    _store_strengths(
        store, embedding=[1.0, 0.0, 0.0], strengths=s, precision=4
    )
    out = reconstruct_strengths(
        store, torch.tensor([1.0, 0.0, 0.0]), k=1, n_neurons=3
    )
    # Not exact, but close — error should be well under the matrix range.
    err = (out - s).abs().max().item()
    assert err < 0.5
    assert not torch.allclose(out, s)


def test_zero_k_returns_zeros() -> None:
    store = ColdStorage(collection_name="recon_zero_k")
    _store_strengths(store, embedding=[1.0], strengths=torch.ones(1, 1))
    out = reconstruct_strengths(
        store, torch.tensor([1.0]), k=0, n_neurons=1
    )
    assert torch.all(out == 0)


def test_rejects_bad_query_shape() -> None:
    store = ColdStorage(collection_name="recon_bad_shape")
    with pytest.raises(ValueError, match="1-D"):
        reconstruct_strengths(
            store, torch.zeros(2, 3), k=4, n_neurons=3
        )
    with pytest.raises(ValueError, match="match"):
        reconstruct_strengths(
            store, torch.zeros(5), k=4, n_neurons=3
        )
    with pytest.raises(ValueError, match="weighting"):
        reconstruct_strengths(
            store, torch.zeros(3), k=4, n_neurons=3, weighting="bogus"
        )


def test_fetch_entries_for_query_is_a_passthrough() -> None:
    store = ColdStorage(collection_name="recon_fetch")
    _store_strengths(
        store, embedding=[1.0, 0.0], strengths=torch.ones(2, 2), entry_id="e1"
    )
    entries = fetch_entries_for_query(
        store, torch.tensor([1.0, 0.0]), k=1
    )
    assert len(entries) == 1
    assert entries[0].id == "e1"
    # Diagnostic helper does not bump access_count.
    assert store.get_by_id("e1").metadata["access_count"] == 0


# ---- Amplification variant: change 2 (confidence-weighted retrieval) ----


def _store_with_access_count(
    store: ColdStorage,
    embedding: list[float],
    strengths: torch.Tensor,
    access_count: int,
    entry_id: str,
) -> None:
    blob = quantize(strengths, precision=32)
    doc = base64.b64encode(blob).decode("ascii")
    store.store_cluster(
        embedding=embedding,
        metadata={
            "precision": 32, "n_neurons": int(strengths.shape[0]),
            "age": 0, "access_count": int(access_count),
            "created_at_step": 0,
        },
        document=doc,
        entry_id=entry_id,
    )


def test_confidence_exponent_zero_preserves_existing_weights() -> None:
    """k=0 ⇒ weight factor is (1+access)**0 = 1 ⇒ bit-exact with the
    pre-amplification path."""
    store = ColdStorage(collection_name="recon_conf_default")
    s_a = torch.full((2, 2), 1.0)
    s_b = torch.full((2, 2), -1.0)
    _store_with_access_count(store, [1.0, 0.0], s_a, access_count=0, entry_id="a")
    _store_with_access_count(store, [0.0, 1.0], s_b, access_count=999, entry_id="b")

    out_default = reconstruct_strengths(
        store, torch.tensor([0.5, 0.5]), k=2, n_neurons=2,
        bump_access_count=False,
    )
    out_explicit_zero = reconstruct_strengths(
        store, torch.tensor([0.5, 0.5]), k=2, n_neurons=2,
        bump_access_count=False, confidence_exponent=0.0,
    )
    torch.testing.assert_close(out_default, out_explicit_zero)


def test_confidence_exponent_upweights_high_access_entries() -> None:
    """Two entries equidistant from the query: with k>0, the entry with
    higher access_count dominates."""
    store = ColdStorage(collection_name="recon_conf_upweight")
    s_low = torch.full((2, 2), 1.0)
    s_high = torch.full((2, 2), -1.0)
    # Symmetric embeddings around the query so distances are equal.
    _store_with_access_count(store, [1.0, 0.0], s_low, access_count=0, entry_id="lo")
    _store_with_access_count(store, [0.0, 1.0], s_high, access_count=99, entry_id="hi")

    out_neutral = reconstruct_strengths(
        store, torch.tensor([0.5, 0.5]), k=2, n_neurons=2,
        bump_access_count=False, confidence_exponent=0.0,
    )
    out_confident = reconstruct_strengths(
        store, torch.tensor([0.5, 0.5]), k=2, n_neurons=2,
        bump_access_count=False, confidence_exponent=1.0,
    )
    # With k=0 the two entries average to zero.
    assert abs(out_neutral.mean().item()) < 1e-6
    # With k=1 the high-access entry (-1.0) outweighs the low-access (+1.0).
    assert out_confident.mean().item() < -0.5


def test_confidence_exponent_uniform_weighting_still_uses_access() -> None:
    """Even under uniform similarity weighting, the access-count factor
    applies — uniform refers only to the distance-derived component."""
    store = ColdStorage(collection_name="recon_conf_uniform")
    s_low = torch.full((2, 2), 2.0)
    s_high = torch.full((2, 2), 4.0)
    _store_with_access_count(store, [1.0, 0.0], s_low, access_count=0, entry_id="lo")
    _store_with_access_count(store, [10.0, 10.0], s_high, access_count=3, entry_id="hi")
    out = reconstruct_strengths(
        store, torch.tensor([5.0, 5.0]), k=2, n_neurons=2,
        weighting="uniform", bump_access_count=False, confidence_exponent=1.0,
    )
    # Weights: low = 1 * (1+0) = 1; high = 1 * (1+3) = 4.
    # total = 5; out = (1*s_low + 4*s_high) / 5 = (2 + 16) / 5 = 3.6
    torch.testing.assert_close(
        out, torch.full((2, 2), 3.6), atol=1e-5, rtol=0
    )


def test_out_retrieved_meta_captures_ids_and_pre_bump_access() -> None:
    """The out-parameter records every entry used in the reconstruction
    with its access_count *before* the bump fires."""
    store = ColdStorage(collection_name="recon_meta_capture")
    _store_with_access_count(store, [1.0, 0.0], torch.zeros(2, 2),
                              access_count=7, entry_id="seven")
    captured: list[tuple[str, int]] = []
    reconstruct_strengths(
        store, torch.tensor([1.0, 0.0]), k=1, n_neurons=2,
        bump_access_count=True, out_retrieved_meta=captured,
    )
    assert captured == [("seven", 7)]
    # The bump did fire — post-call the value is 8.
    assert store.get_by_id("seven").metadata["access_count"] == 8
