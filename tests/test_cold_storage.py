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
