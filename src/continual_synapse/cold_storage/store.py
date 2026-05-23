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

import base64
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Sequence

try:
    import chromadb  # type: ignore[import-untyped]
    from chromadb.api.models.Collection import Collection  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover — chromadb is in requirements.txt
    chromadb = None  # type: ignore[assignment]
    Collection = Any  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from continual_synapse.cold_storage.compression import CompressionSchedule


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

    def update_entry(
        self,
        entry_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        document: str | None = None,
    ) -> None:
        """Replace ``metadata`` and/or ``document`` for ``entry_id``.

        Embedding is not updatable through this method — by design,
        an entry's embedding key is fixed after insertion. To replace
        the embedding, delete and re-insert. When updating the
        ``document``, the existing embedding is re-supplied to
        Chroma's ``update`` to prevent it from auto-computing a
        replacement embedding from the document text (which would
        change the dimension and fail validation against the
        collection's fixed embedding dim).
        """
        if metadata is None and document is None:
            return  # nothing to do
        kwargs: dict[str, Any] = {"ids": [entry_id]}
        if metadata is not None:
            kwargs["metadatas"] = [metadata]
        if document is not None:
            # Fetch + re-supply the embedding so Chroma's auto-
            # embedding doesn't override our fixed key.
            existing = self.get_by_id(entry_id)
            kwargs["documents"] = [document]
            kwargs["embeddings"] = [existing.embedding]
        self._collection.update(**kwargs)

    def delete_cluster(self, entry_id: str) -> None:
        """Remove ``entry_id`` from the store."""
        self._collection.delete(ids=[entry_id])

    # ---- compression re-evaluation (Phase 4b follow-up) ----

    def re_evaluate_all_entries(
        self,
        current_step: int,
        schedule: "CompressionSchedule",
    ) -> dict[int, int]:
        """Walk every entry, age it, and re-quantise if the schedule shifts it.

        The compression schedule was previously consulted only at
        insertion time (with ``age=0, access_count=0``) — so every
        entry stayed at the schedule's "fresh" precision (32-bit by
        default) for the lifetime of the experiment, regardless of
        what the design intended. This method closes that gap by
        applying the schedule to existing entries.

        For each entry:
          1. Compute ``current_age = max(current_step - created_at_step, 0)``.
          2. Compute ``new_precision = schedule.precision_for(current_age,
             access_count)``.
          3. If ``new_precision != current_precision``: dequantize at
             the current precision, re-quantize at the new precision,
             update the entry's document.
          4. Always update the entry's ``metadata["age"]`` to the
             freshly-computed value and ``metadata["precision"]`` to
             the chosen tier.

        Args:
            current_step: The trainer's global step (typically
                ``synapse.global_step``). Subtracted from each entry's
                ``created_at_step`` to derive its current age.
            schedule: The :class:`CompressionSchedule` to apply.

        Returns:
            Dict mapping precision tier -> count of entries now at
            that tier. Useful for diagnostics ("how many entries are
            at 16-bit?").
        """
        from continual_synapse.cold_storage.compression import (
            dequantize,
            quantize,
        )

        counts: dict[int, int] = {}
        for entry in self.all_entries():
            old_precision = int(entry.metadata.get("precision", 32))
            created_at = int(entry.metadata.get("created_at_step", 0))
            access_count = int(entry.metadata.get("access_count", 0))
            n_neurons = int(entry.metadata.get("n_neurons", 0))
            current_age = max(int(current_step) - created_at, 0)
            new_precision = schedule.precision_for(
                age=current_age, access_count=access_count
            )

            new_metadata = dict(entry.metadata)
            new_metadata["age"] = current_age
            new_metadata["precision"] = int(new_precision)

            if new_precision != old_precision and n_neurons > 0:
                old_blob = base64.b64decode(entry.document)
                tensor = dequantize(
                    old_blob,
                    precision=old_precision,
                    shape=(n_neurons, n_neurons),
                )
                new_blob = quantize(tensor, precision=new_precision)
                new_document = base64.b64encode(new_blob).decode("ascii")
                self.update_entry(
                    entry.id,
                    metadata=new_metadata,
                    document=new_document,
                )
            else:
                # Even when precision is unchanged, update metadata so
                # the `age` field is no longer stale at 0.
                self.update_metadata(entry.id, new_metadata)

            counts[int(new_precision)] = counts.get(int(new_precision), 0) + 1
        return counts

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
