"""Cold storage layer — Chroma-backed vector store of consolidated synapse patterns.

Wraps the subset of the Chroma client API we actually use into a
small, typed interface. The default backend is Chroma's in-memory
client, which has no external infrastructure dependency and is
fast enough for the research-scale entry counts we expect in
Phase 4 (a few hundred to a few thousand entries per run).

Each stored entry represents one consolidation event:

- ``embedding`` is a 1-D float vector — the activation pattern
  observed at the moment of consolidation. This is what we query
  against at retrieval time.
- ``document`` is a base-64 encoded byte string holding the
  compressed strengths matrix for the consolidated synapses. We
  use Chroma's ``document`` field rather than ``metadata`` because
  metadata cannot hold raw bytes.
- ``metadata`` carries plain JSON-serialisable fields: precision,
  ``n_neurons``, age, access_count, creation step.

Backend choice: Chroma's ``Client()`` (in-memory). The interface
is narrow enough that swapping for a PersistentClient or a
different vector DB is a one-line constructor change. See the
companion design note in DESIGN.md §3.3.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

try:
    import chromadb  # type: ignore[import-untyped]
    from chromadb.api.models.Collection import Collection  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover — chromadb is in requirements.txt
    chromadb = None  # type: ignore[assignment]
    Collection = Any  # type: ignore[misc, assignment]


_COLLECTION_NAME = "synapse_archive"


@dataclass
class StoredEntry:
    """One retrieval result, materialised as a Python dataclass.

    Attributes:
        id: Chroma row id.
        embedding: The stored activation pattern (1-D).
        document: Base-64 string of compressed strengths bytes.
        metadata: Per-entry metadata dict.
        distance: Squared L2 distance from the query embedding;
            smaller is more similar. ``None`` when retrieved
            without a query (e.g. by id).
    """

    id: str
    embedding: list[float]
    document: str
    metadata: dict[str, Any]
    distance: float | None = None


class ColdStorage:
    """Thin typed wrapper over a Chroma collection.

    The class is deliberately small: every operation maps 1:1 onto
    a Chroma call so the cost surface is easy to reason about.

    Args:
        collection_name: Name of the Chroma collection to use.
            Defaults to ``"synapse_archive"``. Renaming on the same
            client gives an isolated, freshly-empty store, which is
            how experiment scripts get a clean slate per seed.
        client: Optional pre-built Chroma client (handy for tests
            that share one client across stores). If ``None`` a
            fresh in-memory client is constructed.
    """

    def __init__(
        self,
        collection_name: str = _COLLECTION_NAME,
        client: Any = None,
    ) -> None:
        if chromadb is None:
            raise RuntimeError(
                "chromadb is not installed. `pip install chromadb==1.2.1`."
            )
        self._client = client if client is not None else chromadb.Client()
        self.collection_name = collection_name
        # Recreate the collection so each ColdStorage instance starts empty
        # within the lifetime of its client. Tests that share a client across
        # stores should pick unique collection_names.
        try:
            self._client.delete_collection(name=collection_name)
        except Exception:
            pass
        self._collection: Collection = self._client.create_collection(
            name=collection_name
        )

    # ---- basic lifecycle ----

    def count(self) -> int:
        return int(self._collection.count())

    def clear(self) -> None:
        """Drop every entry. Constant-time via collection rebuild."""
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name
        )

    # ---- writes ----

    def store_cluster(
        self,
        embedding: Sequence[float],
        metadata: dict[str, Any],
        document: str,
        entry_id: str | None = None,
    ) -> str:
        """Insert one consolidation entry and return its id.

        Args:
            embedding: Float sequence used as the retrieval key.
            metadata: JSON-serialisable per-entry data. Booleans
                must be Python ``bool``; numpy/torch scalars must
                be cast by the caller.
            document: Base-64 string holding the compressed
                strengths bytes for this entry.
            entry_id: Optional explicit id. If ``None`` a UUID4 is
                generated.
        """
        if entry_id is None:
            entry_id = uuid.uuid4().hex
        self._collection.add(
            ids=[entry_id],
            embeddings=[list(embedding)],
            metadatas=[metadata],
            documents=[document],
        )
        return entry_id

    def update_metadata(self, entry_id: str, metadata: dict[str, Any]) -> None:
        """Replace ``metadata`` for ``entry_id`` in place."""
        self._collection.update(ids=[entry_id], metadatas=[metadata])

    def delete_cluster(self, entry_id: str) -> None:
        """Remove ``entry_id`` from the store."""
        self._collection.delete(ids=[entry_id])

    # ---- reads ----

    def retrieve_similar(
        self,
        query_embedding: Sequence[float],
        k: int,
    ) -> list[StoredEntry]:
        """Return up to ``k`` most-similar entries, closest first.

        If the store is empty, returns ``[]``. ``k`` is capped at
        the current entry count, so callers don't need to special-
        case small stores.
        """
        n = self.count()
        if n == 0 or k <= 0:
            return []
        k_eff = min(int(k), n)
        result = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=k_eff,
            include=["embeddings", "metadatas", "documents", "distances"],
        )
        return list(_unpack_query(result))

    def get_by_id(self, entry_id: str) -> StoredEntry:
        """Fetch one entry by id. Raises ``KeyError`` if absent."""
        result = self._collection.get(
            ids=[entry_id],
            include=["embeddings", "metadatas", "documents"],
        )
        ids = list(result.get("ids", []))
        if not ids:
            raise KeyError(entry_id)
        return StoredEntry(
            id=ids[0],
            embedding=list(_safe_index(result.get("embeddings"), 0)),
            document=_safe_index(result.get("documents"), 0),
            metadata=dict(_safe_index(result.get("metadatas"), 0)),
            distance=None,
        )

    def all_entries(self) -> list[StoredEntry]:
        """Return every entry. Mostly useful for diagnostics and tests."""
        if self.count() == 0:
            return []
        result = self._collection.get(
            include=["embeddings", "metadatas", "documents"],
        )
        ids = list(result.get("ids", []))
        out: list[StoredEntry] = []
        for i, eid in enumerate(ids):
            out.append(
                StoredEntry(
                    id=eid,
                    embedding=list(_safe_index(result.get("embeddings"), i)),
                    document=_safe_index(result.get("documents"), i),
                    metadata=dict(_safe_index(result.get("metadatas"), i)),
                    distance=None,
                )
            )
        return out


def _safe_index(value: Any, idx: int) -> Any:
    """Return ``value[idx]`` for sequences and dict-likes that may be None."""
    if value is None:
        raise KeyError(idx)
    return value[idx]


def _unpack_query(result: dict[str, Any]) -> Iterable[StoredEntry]:
    """Yield :class:`StoredEntry` from a Chroma query result.

    Chroma returns each field as a list-of-lists keyed by query
    embedding; we always query with one embedding at a time, so we
    take element ``[0]`` of each.
    """
    ids = result.get("ids", [[]])[0]
    embeddings = result.get("embeddings", [[]])[0] if result.get("embeddings") is not None else []
    documents = result.get("documents", [[]])[0] if result.get("documents") is not None else []
    metadatas = result.get("metadatas", [[]])[0] if result.get("metadatas") is not None else []
    distances = result.get("distances", [[]])[0] if result.get("distances") is not None else []
    for i, eid in enumerate(ids):
        yield StoredEntry(
            id=eid,
            embedding=list(embeddings[i]) if i < len(embeddings) else [],
            document=documents[i] if i < len(documents) else "",
            metadata=dict(metadatas[i]) if i < len(metadatas) else {},
            distance=float(distances[i]) if i < len(distances) else None,
        )
