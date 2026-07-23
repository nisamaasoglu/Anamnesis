"""Text -> vector, with an honest offline fallback.

Two providers are supported:

``openai``   real semantic embeddings (requires OPENAI_API_KEY and network).
``offline``  a signed hashing vectoriser over unigrams and bigrams.

The offline provider is **lexical, not semantic**: it matches on shared words,
not on shared meaning. It exists so the whole system is runnable, testable and
demonstrable with no API key and no network, and so CI can assert retrieval
behaviour deterministically. Retrieval quality in offline mode is measurably
lower than with real embeddings; the mode is reported in every API response so
a caller can never mistake one for the other.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence

import numpy as np

from . import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Very common English words carry no retrieval signal and would otherwise
# dominate the hashed vector, so they are dropped before hashing.
_STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i if in into is it its
    may must not of on or our shall should that the their there they this to was we what
    when where which who will with would you your""".split()
)


def _tokens(text: str) -> list[str]:
    """Lower-cased unigrams (stopwords removed) plus adjacent bigrams."""
    words = [w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS]
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
    return words + bigrams


def content_terms(text: str) -> set[str]:
    """Meaningful single words in a text, stopwords removed.

    Used by the offline retriever as a collision guard: the hashing trick can put
    two unrelated terms in the same bucket, so a candidate passage must share at
    least one real word with the query before it is accepted as a match.
    """
    return {w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


def _hashed_vector(text: str, dim: int) -> np.ndarray:
    """Signed hashing trick with sublinear term scaling and L2 normalisation.

    The sign bit of the same hash decides the direction of each increment, which
    keeps unrelated collisions from systematically inflating similarity.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % dim
        sign = 1.0 if (value >> 63) & 1 else -1.0
        vec[index] += sign
    # Sublinear scaling: a term repeated ten times is not ten times as important.
    vec = np.sign(vec) * np.log1p(np.abs(vec))
    return _l2_normalise(vec)


def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return (vec / norm).astype(np.float32)


def _embed_offline(texts: Sequence[str], dim: int) -> np.ndarray:
    return np.vstack([_hashed_vector(t, dim) for t in texts]).astype(np.float32)


def _embed_openai(texts: Sequence[str]) -> np.ndarray:
    from openai import OpenAI  # imported lazily: not required in offline mode

    client = OpenAI(api_key=config.OPENAI_API_KEY or None)
    response = client.embeddings.create(model=config.OPENAI_EMBEDDING_MODEL, input=list(texts))
    ordered = sorted(response.data, key=lambda item: item.index)
    return np.vstack([_l2_normalise(np.asarray(d.embedding, dtype=np.float32)) for d in ordered])


def embed_texts(texts: Iterable[str], provider: str | None = None) -> np.ndarray:
    """Embed a batch of texts and return an (n, dim) L2-normalised matrix.

    Because every row is unit length, a dot product is exactly cosine similarity.
    """
    items = list(texts)
    if not items:
        return np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)

    provider = (provider or config.EMBEDDING_PROVIDER).lower()
    if provider == "openai":
        return _embed_openai(items)
    if provider == "offline":
        return _embed_offline(items, config.EMBEDDING_DIM)
    raise ValueError(f"Unknown embedding provider: {provider!r}")


def embed_query(text: str, provider: str | None = None) -> np.ndarray:
    """Embed a single query and return a 1-D vector."""
    return embed_texts([text], provider=provider)[0]


def describe_provider(provider: str | None = None) -> dict[str, object]:
    """Machine-readable statement of what the current provider actually is."""
    provider = (provider or config.EMBEDDING_PROVIDER).lower()
    if provider == "openai":
        return {
            "provider": "openai",
            "model": config.OPENAI_EMBEDDING_MODEL,
            "semantic": True,
            "note": "Real embedding model; retrieval matches on meaning.",
        }
    return {
        "provider": "offline",
        "model": f"signed-hashing-{config.EMBEDDING_DIM}d",
        "semantic": False,
        "note": (
            "Lexical hashing vectoriser. Matches shared words, not shared meaning. "
            "Deterministic and dependency-free; retrieval quality is lower than a "
            "real embedding model."
        ),
    }
