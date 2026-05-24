"""Tests for the cold storage Chroma wrapper."""

from __future__ import annotations

import pytest

from continual_synapse.cold_storage.store import ColdStorage, StoredEntry


def test_store_starts_empty() -> None:
    store = ColdStorage(collection_name="empty_test")
    assert store.count() == 0
    assert store.all_entries() == []


def test_store_cluster_returns_entry_id() -> None:
    store = ColdStorage(collection_name="store_test")
    eid = store.store_cluster(
        embedding=[1.0, 2.0, 3.0],
        metadata={"age": 0, "precision": 32},
        document="payload",
    )
    assert isinstance(eid, str) and len(eid) > 0
    assert store.count() == 1


def test_store_cluster_accepts_explicit_id() -> None:
    store = ColdStorage(collection_name="explicit_id")
    store.store_cluster(
        embedding=[0.0, 1.0], metadata={"k": 1}, document="d", entry_id="abc"
    )
    e = store.get_by_id("abc")
    assert e.id == "abc"
    assert e.document == "d"


def test_retrieve_similar_returns_nearest_first() -> None:
    store = ColdStorage(collection_name="retrieve_test")
    store.store_cluster(
        embedding=[1.0, 0.0],
        metadata={"task": "a"},
        document="doc_a",
        entry_id="a",
    )
    store.store_cluster(
        embedding=[0.0, 1.0],
        metadata={"task": "b"},
        document="doc_b",
        entry_id="b",
    )
    results = store.retrieve_similar([0.9, 0.1], k=2)
    assert len(results) == 2
    assert results[0].id == "a"
    assert results[1].id == "b"
    assert results[0].distance is not None
    assert results[0].distance <= results[1].distance


def test_retrieve_similar_caps_k_at_count() -> None:
    store = ColdStorage(collection_name="cap_k")
    store.store_cluster(
        embedding=[1.0], metadata={"k": 1}, document="x", entry_id="only"
    )
    results = store.retrieve_similar([0.5], k=10)
    assert len(results) == 1


def test_retrieve_similar_returns_empty_when_store_empty() -> None:
    store = ColdStorage(collection_name="empty_retrieve")
    assert store.retrieve_similar([0.0, 0.0], k=5) == []


def test_retrieve_similar_with_zero_k_is_empty() -> None:
    store = ColdStorage(collection_name="zero_k")
    store.store_cluster(
        embedding=[1.0], metadata={"k": 1}, document="x"
    )
    assert store.retrieve_similar([1.0], k=0) == []


def test_update_metadata_replaces_existing_fields() -> None:
    store = ColdStorage(collection_name="update_md")
    store.store_cluster(
        embedding=[1.0, 0.0],
        metadata={"age": 0, "access_count": 0},
        document="d",
        entry_id="m1",
    )
    store.update_metadata("m1", {"age": 5, "access_count": 2})
    e = store.get_by_id("m1")
    assert e.metadata["age"] == 5
    assert e.metadata["access_count"] == 2


def test_delete_cluster_removes_entry() -> None:
    store = ColdStorage(collection_name="delete_test")
    store.store_cluster(
        embedding=[1.0], metadata={"k": 1}, document="x", entry_id="gone"
    )
    assert store.count() == 1
    store.delete_cluster("gone")
    assert store.count() == 0
    with pytest.raises(KeyError):
        store.get_by_id("gone")


def test_clear_drops_every_entry() -> None:
    store = ColdStorage(collection_name="clear_test")
    for i in range(3):
        store.store_cluster(
            embedding=[float(i)], metadata={"i": i}, document=f"d{i}"
        )
    assert store.count() == 3
    store.clear()
    assert store.count() == 0


def test_get_by_id_raises_for_unknown_id() -> None:
    store = ColdStorage(collection_name="get_missing")
    with pytest.raises(KeyError):
        store.get_by_id("does_not_exist")


def test_stored_entry_dataclass_round_trips_fields() -> None:
    store = ColdStorage(collection_name="round_trip")
    store.store_cluster(
        embedding=[1.0, 2.0, 3.0],
        metadata={"a": 1, "b": "two"},
        document="payload",
        entry_id="rt",
    )
    e = store.get_by_id("rt")
    assert isinstance(e, StoredEntry)
    assert e.embedding == [1.0, 2.0, 3.0]
    assert e.metadata == {"a": 1, "b": "two"}
    assert e.document == "payload"


def test_collections_with_distinct_names_are_isolated() -> None:
    a = ColdStorage(collection_name="isolated_a")
    b = ColdStorage(collection_name="isolated_b")
    a.store_cluster(embedding=[1.0], metadata={"src": "a"}, document="a")
    b.store_cluster(embedding=[2.0], metadata={"src": "b"}, document="b")
    assert a.count() == 1
    assert b.count() == 1
    a.clear()
    assert a.count() == 0
    assert b.count() == 1


def test_collection_recreate_on_reuse() -> None:
    """Two ColdStorage instances with the same collection name should each
    start empty; the second instance's constructor wipes any leftover
    state from the first one."""
    a = ColdStorage(collection_name="reused")
    a.store_cluster(embedding=[1.0], metadata={"src": "a"}, document="x")
    assert a.count() == 1

    b = ColdStorage(collection_name="reused")
    assert b.count() == 0


# ---- Phase 4b follow-up: compression re-evaluation sweep ----


def _store_compressed(
    store: ColdStorage,
    strengths,
    *,
    embedding,
    precision: int = 32,
    created_at_step: int = 0,
    access_count: int = 0,
    entry_id: str | None = None,
) -> str:
    """Helper: store a synapse strengths matrix at given precision."""
    import base64

    import torch

    from continual_synapse.cold_storage.compression import quantize

    t = torch.as_tensor(strengths, dtype=torch.float32)
    blob = quantize(t, precision=precision)
    doc = base64.b64encode(blob).decode("ascii")
    meta = {
        "precision": precision,
        "n_neurons": int(t.shape[0]),
        "age": 0,
        "access_count": access_count,
        "created_at_step": created_at_step,
    }
    return store.store_cluster(
        embedding=embedding, metadata=meta, document=doc, entry_id=entry_id
    )


def test_update_entry_replaces_document_and_metadata() -> None:
    store = ColdStorage(collection_name="update_entry_test")
    store.store_cluster(
        embedding=[1.0, 0.0],
        metadata={"k": 1},
        document="old",
        entry_id="x",
    )
    store.update_entry(
        "x", metadata={"k": 2, "v": "y"}, document="new"
    )
    e = store.get_by_id("x")
    assert e.document == "new"
    assert e.metadata["k"] == 2
    assert e.metadata["v"] == "y"


def test_update_entry_metadata_only_keeps_document() -> None:
    store = ColdStorage(collection_name="update_md_only")
    store.store_cluster(
        embedding=[1.0],
        metadata={"k": 1},
        document="keep",
        entry_id="x",
    )
    store.update_entry("x", metadata={"k": 9})
    e = store.get_by_id("x")
    assert e.document == "keep"
    assert e.metadata["k"] == 9


def test_update_entry_with_neither_arg_is_noop() -> None:
    store = ColdStorage(collection_name="update_noop")
    store.store_cluster(
        embedding=[1.0],
        metadata={"k": 1},
        document="x",
        entry_id="a",
    )
    store.update_entry("a")  # neither metadata nor document
    e = store.get_by_id("a")
    assert e.document == "x"
    assert e.metadata == {"k": 1}


def test_re_evaluate_updates_age_field() -> None:
    """The first thing re_evaluate does is update each entry's `age`
    metadata to current_step - created_at_step. Verify on a single
    entry whose precision doesn't need to change."""
    import torch

    from continual_synapse.cold_storage.compression import CompressionSchedule

    store = ColdStorage(collection_name="reeval_age")
    _store_compressed(
        store,
        torch.zeros(2, 2),
        embedding=[1.0, 0.0],
        precision=32,
        created_at_step=10,
        entry_id="a",
    )
    sched = CompressionSchedule()
    store.re_evaluate_all_entries(current_step=50, schedule=sched)
    e = store.get_by_id("a")
    # current_age = 50 - 10 = 40, which is < 100 so still 32-bit.
    assert e.metadata["age"] == 40
    assert e.metadata["precision"] == 32


def test_re_evaluate_re_quantises_when_threshold_crossed() -> None:
    """When age crosses the threshold, the entry's precision shifts
    and the document is re-encoded."""
    import torch

    from continual_synapse.cold_storage.compression import CompressionSchedule

    store = ColdStorage(collection_name="reeval_quantise")
    s = torch.tensor([[1.0, -0.5], [0.25, 0.75]])
    _store_compressed(
        store,
        s,
        embedding=[1.0, 0.0],
        precision=32,
        created_at_step=0,
        entry_id="b",
    )
    before = store.get_by_id("b")
    sched = CompressionSchedule()  # age >= 500 -> 8-bit
    counts = store.re_evaluate_all_entries(current_step=600, schedule=sched)

    e = store.get_by_id("b")
    # age 600 is in [500, 2000) -> 8-bit tier.
    assert e.metadata["age"] == 600
    assert e.metadata["precision"] == 8
    # Document was re-quantised, so it differs from the original.
    assert e.document != before.document
    # The counts dict reports one entry at the new tier.
    assert counts == {8: 1}


def test_re_evaluate_round_trips_strengths_within_tolerance() -> None:
    """After re-quantisation, dequantizing returns a tensor close to the
    original within the new tier's precision error."""
    import base64
    import torch

    from continual_synapse.cold_storage.compression import (
        CompressionSchedule,
        dequantize,
    )

    store = ColdStorage(collection_name="reeval_roundtrip")
    g = torch.Generator().manual_seed(0)
    s = torch.randn(3, 3, generator=g)
    _store_compressed(
        store, s, embedding=[1.0, 0.0, 0.0], precision=32, entry_id="r"
    )
    sched = CompressionSchedule()
    store.re_evaluate_all_entries(current_step=2500, schedule=sched)
    e = store.get_by_id("r")
    assert e.metadata["precision"] == 4
    recovered = dequantize(
        base64.b64decode(e.document), precision=4, shape=(3, 3)
    )
    # 4-bit max-abs quantisation: step ~ max(|s|)/7, half-step error ~ that/2.
    err = (recovered - s).abs().max().item()
    assert err < (s.abs().max().item() / 7.0) + 1e-3


def test_re_evaluate_counts_entries_per_tier() -> None:
    import torch

    from continual_synapse.cold_storage.compression import CompressionSchedule

    store = ColdStorage(collection_name="reeval_counts")
    # Three entries, three different ages spanning the schedule tiers.
    _store_compressed(
        store, torch.zeros(2, 2), embedding=[1.0, 0.0],
        created_at_step=2000, entry_id="recent",  # age 0 at step 2000 -> 32
    )
    _store_compressed(
        store, torch.zeros(2, 2), embedding=[0.5, 0.5],
        created_at_step=1500, entry_id="medium",  # age 500 -> 8-bit
    )
    _store_compressed(
        store, torch.zeros(2, 2), embedding=[0.0, 1.0],
        created_at_step=0, entry_id="old",  # age 2000 -> 4-bit
    )
    sched = CompressionSchedule()
    counts = store.re_evaluate_all_entries(current_step=2000, schedule=sched)
    assert counts == {32: 1, 8: 1, 4: 1}


def test_re_evaluate_respects_access_count_bump() -> None:
    """access_count >= 5 bumps the tier up by one. A medium-age
    entry with high access_count should land in the next-higher tier."""
    import torch

    from continual_synapse.cold_storage.compression import CompressionSchedule

    store = ColdStorage(collection_name="reeval_access")
    _store_compressed(
        store, torch.zeros(2, 2), embedding=[1.0, 0.0],
        created_at_step=0, access_count=10, entry_id="popular",
    )
    sched = CompressionSchedule()
    store.re_evaluate_all_entries(current_step=600, schedule=sched)
    e = store.get_by_id("popular")
    # age 600 normally -> 8-bit (tier index 2). access_count 10 >= 5
    # bumps it up to tier index 1 -> 16-bit.
    assert e.metadata["precision"] == 16


def test_re_evaluate_empty_store_returns_empty_counts() -> None:
    from continual_synapse.cold_storage.compression import CompressionSchedule

    store = ColdStorage(collection_name="reeval_empty")
    counts = store.re_evaluate_all_entries(
        current_step=1000, schedule=CompressionSchedule()
    )
    assert counts == {}


def test_re_evaluate_reduces_total_byte_size() -> None:
    """Memory footprint should actually shrink after a sweep that
    moves entries to lower precision."""
    import base64
    import torch

    from continual_synapse.cold_storage.compression import CompressionSchedule

    store = ColdStorage(collection_name="reeval_memory")
    for i in range(5):
        _store_compressed(
            store,
            torch.randn(4, 4, generator=torch.Generator().manual_seed(i)),
            embedding=[float(i), 0.0],
            created_at_step=0,
            entry_id=f"e{i}",
        )
    before_bytes = sum(
        len(base64.b64decode(e.document)) for e in store.all_entries()
    )
    store.re_evaluate_all_entries(
        current_step=3000, schedule=CompressionSchedule()
    )
    after_bytes = sum(
        len(base64.b64decode(e.document)) for e in store.all_entries()
    )
    # 4-bit entries are roughly 1/8 the size of 32-bit.
    assert after_bytes < before_bytes / 4


# ---- compute_similarities (experiment 16: cosine-familiarity helper) ----


def test_compute_similarities_empty_store_returns_empty_list() -> None:
    store = ColdStorage(collection_name="sims_empty")
    assert store.compute_similarities([1.0, 0.0, 0.0]) == []


def test_compute_similarities_identical_pattern_returns_one() -> None:
    store = ColdStorage(collection_name="sims_identical")
    store.store_cluster(
        embedding=[1.0, 0.0, 0.0],
        metadata={"precision": 32, "n_neurons": 3},
        document="x",
        entry_id="same",
    )
    sims = store.compute_similarities([1.0, 0.0, 0.0])
    assert len(sims) == 1
    assert sims[0] == pytest.approx(1.0, abs=1e-6)


def test_compute_similarities_orthogonal_pattern_returns_zero() -> None:
    store = ColdStorage(collection_name="sims_orthogonal")
    store.store_cluster(
        embedding=[1.0, 0.0, 0.0],
        metadata={"precision": 32, "n_neurons": 3},
        document="x",
        entry_id="x_axis",
    )
    sims = store.compute_similarities([0.0, 1.0, 0.0])
    assert len(sims) == 1
    assert sims[0] == pytest.approx(0.0, abs=1e-6)


def test_compute_similarities_antiparallel_pattern_returns_minus_one() -> None:
    store = ColdStorage(collection_name="sims_antiparallel")
    store.store_cluster(
        embedding=[1.0, 0.0, 0.0],
        metadata={"precision": 32, "n_neurons": 3},
        document="x",
        entry_id="positive_x",
    )
    sims = store.compute_similarities([-1.0, 0.0, 0.0])
    assert sims[0] == pytest.approx(-1.0, abs=1e-6)


def test_compute_similarities_multiple_entries_in_insertion_order() -> None:
    """Returned list matches all_entries() order — caller can map back
    to entry ids by zipping."""
    store = ColdStorage(collection_name="sims_multi")
    for i, (emb, eid) in enumerate([
        ([1.0, 0.0], "a"),
        ([0.0, 1.0], "b"),
        ([1.0, 1.0], "c"),
    ]):
        store.store_cluster(
            embedding=emb, metadata={"precision": 32, "n_neurons": 2},
            document="x", entry_id=eid,
        )
    sims = store.compute_similarities([1.0, 0.0])
    assert len(sims) == 3
    # In insertion order: ~1, ~0, ~0.707
    assert sims[0] == pytest.approx(1.0, abs=1e-6)
    assert sims[1] == pytest.approx(0.0, abs=1e-6)
    assert sims[2] == pytest.approx(0.7071, abs=1e-3)
    # Ordering matches all_entries.
    by_id = {e.id: sim for e, sim in zip(store.all_entries(), sims)}
    assert by_id["a"] == pytest.approx(1.0, abs=1e-6)


def test_compute_similarities_handles_zero_vectors_without_nan() -> None:
    """Zero-norm vectors should not produce NaN (clamp_min protects)."""
    store = ColdStorage(collection_name="sims_zero")
    store.store_cluster(
        embedding=[0.0, 0.0, 0.0],
        metadata={"precision": 32, "n_neurons": 3},
        document="x",
    )
    sims = store.compute_similarities([1.0, 0.0, 0.0])
    # Numerator is zero, denominator is clamped → result is 0, not NaN.
    assert sims[0] == 0.0
    # Zero query against non-zero entry also OK.
    store2 = ColdStorage(collection_name="sims_zero_q")
    store2.store_cluster(
        embedding=[1.0, 1.0, 1.0],
        metadata={"precision": 32, "n_neurons": 3},
        document="x",
    )
    assert store2.compute_similarities([0.0, 0.0, 0.0]) == [0.0]
