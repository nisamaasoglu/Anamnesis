"""Query side of the knowledge base: load the index once, search it many times."""

from __future__ import annotations

from pathlib import Path

from . import config
from .embeddings import embed_query
from .ingest import build_index
from .lexical import BM25, Document, expand_query
from .vectorstore import Hit, VectorStore

_STORE: VectorStore | None = None
_BM25: BM25 | None = None
_BM25_KEY: int | None = None


def get_store(index_path: Path | None = None, rebuild: bool = False) -> VectorStore:
    """Return the process-wide store, building the index on first use."""
    global _STORE
    path = Path(index_path or config.INDEX_PATH)
    if _STORE is not None and not rebuild:
        return _STORE
    if rebuild or not path.exists():
        _STORE = build_index(index_path=path)
    else:
        _STORE = VectorStore.load(path)
    return _STORE


def reset_store() -> None:
    """Drop the cached store (used by tests and by the /reindex endpoint)."""
    global _STORE, _BM25, _BM25_KEY
    _STORE = None
    _BM25 = None
    _BM25_KEY = None


def _bm25_for(store: VectorStore) -> BM25:
    global _BM25, _BM25_KEY
    if _BM25 is None or _BM25_KEY != id(store):
        _BM25 = BM25(
            [Document(c.section, c.document_title, c.text) for c in store.chunks]
        )
        _BM25_KEY = id(store)
    return _BM25


def _hybrid_search(question: str, store: VectorStore, top_k: int) -> list[Hit]:
    """Offline ranking: field-weighted BM25 first, hashed cosine as a tie-breaker.

    A passage is only eligible if BM25 finds real (stemmed or expanded) term
    overlap (see :meth:`BM25.is_evidence`) and a score above ``BM25_MIN_SCORE``.
    That keeps the guarantee the old lexical gate gave: an off-topic question
    returns nothing rather than a collision.
    """
    index = _bm25_for(store)
    query_terms = expand_query(question)
    bm25 = [
        score if index.is_evidence(i, query_terms) else 0.0
        for i, score in enumerate(index.scores(query_terms))
    ]
    best = max(bm25, default=0.0)
    if best < config.BM25_MIN_SCORE:
        return []
    cosine = store.vectors @ embed_query(question).astype(store.vectors.dtype)
    ranked: list[Hit] = []
    for i, lexical in enumerate(bm25):
        if lexical < config.BM25_MIN_SCORE:
            continue
        combined = 0.85 * (lexical / best) + 0.15 * max(float(cosine[i]), 0.0)
        ranked.append(Hit(store.chunks[i], round(combined, 6)))
    ranked.sort(key=lambda hit: -hit.score)
    return ranked[:top_k]


def search(question: str, k: int | None = None, min_score: float | None = None) -> list[Hit]:
    """Retrieve the passages most likely to answer ``question``.

    Offline mode uses :func:`_hybrid_search`, because the hashing vectoriser
    alone matches exact surface words only and misranks headings. With real
    embeddings the cosine ranking is used directly, because there a match on
    meaning without a shared word is the point.
    """
    store = get_store()
    top_k = config.TOP_K if k is None else k
    if config.EMBEDDING_PROVIDER.lower() == "offline":
        return _hybrid_search(question, store, top_k)
    floor = config.MIN_SCORE if min_score is None else min_score
    return store.search(embed_query(question), k=top_k, min_score=floor)


def format_context(hits: list[Hit]) -> str:
    """Render hits as numbered, citable context for an LLM prompt."""
    return "\n\n".join(
        f"[{i}] {hit.chunk.document_title} - {hit.chunk.section}\n{hit.chunk.text}"
        for i, hit in enumerate(hits, start=1)
    )
