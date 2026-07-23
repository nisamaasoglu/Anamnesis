"""Query side of the knowledge base: load the index once, search it many times."""

from __future__ import annotations

from pathlib import Path

from . import config
from .embeddings import content_terms, embed_query
from .ingest import build_index
from .vectorstore import Hit, VectorStore

_STORE: VectorStore | None = None


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
    global _STORE
    _STORE = None


def _shares_a_real_term(question: str, hit: Hit) -> bool:
    query_terms = content_terms(question)
    if not query_terms:
        return False
    passage = f"{hit.chunk.document_title} {hit.chunk.section} {hit.chunk.text}"
    return bool(query_terms & content_terms(passage))


def search(question: str, k: int | None = None, min_score: float | None = None) -> list[Hit]:
    """Retrieve the passages most likely to answer ``question``.

    In offline mode a lexical gate is applied on top of the cosine ranking. The
    hashing vectoriser is compact but collides, and a collision can float an
    unrelated passage above the score floor; requiring one genuinely shared term
    removes that class of false positive. With real embeddings the gate is not
    applied, because there a match on meaning without a shared word is the point.
    """
    store = get_store()
    top_k = config.TOP_K if k is None else k
    floor = config.MIN_SCORE if min_score is None else min_score
    hits = store.search(embed_query(question), k=top_k * 5, min_score=floor)
    if config.EMBEDDING_PROVIDER.lower() == "offline":
        hits = [hit for hit in hits if _shares_a_real_term(question, hit)]
    return hits[:top_k]


def format_context(hits: list[Hit]) -> str:
    """Render hits as numbered, citable context for an LLM prompt."""
    return "\n\n".join(
        f"[{i}] {hit.chunk.document_title} - {hit.chunk.section}\n{hit.chunk.text}"
        for i, hit in enumerate(hits, start=1)
    )
