"""A small persistent vector store.

The corpus here is a hospital's policy library — a few hundred chunks, not a
few million — so an in-memory NumPy matrix with an exact cosine scan is both
faster and easier to defend than a database. Vectors are stored L2-normalised,
which turns the whole search into a single matrix-vector product.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage plus everything needed to cite it."""

    chunk_id: str
    document: str
    document_title: str
    section: str
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float

    def citation(self) -> str:
        return f"{self.chunk.document_title} - {self.chunk.section}"


class VectorStore:
    def __init__(
        self, vectors: np.ndarray | None = None, chunks: list[Chunk] | None = None
    ) -> None:
        self.chunks: list[Chunk] = list(chunks or [])
        if vectors is None or len(self.chunks) == 0:
            self.vectors = np.zeros((0, 0), dtype=np.float32)
        else:
            self.vectors = vectors.astype(np.float32)
        if len(self.chunks) != self.vectors.shape[0]:
            raise ValueError("vector count does not match chunk count")

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    def search(self, query_vector: np.ndarray, k: int = 4, min_score: float = 0.0) -> list[Hit]:
        """Return the k highest-scoring chunks above ``min_score``."""
        if len(self) == 0:
            return []
        if query_vector.shape[0] != self.dim:
            raise ValueError(f"query dim {query_vector.shape[0]} != index dim {self.dim}")
        scores = self.vectors @ query_vector.astype(np.float32)
        order = np.argsort(-scores)[: max(k, 0)]
        return [
            Hit(self.chunks[i], float(scores[i]))
            for i in order
            if float(scores[i]) >= min_score
        ]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vectors=self.vectors)
        path.with_suffix(".meta.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> VectorStore:
        path = Path(path)
        meta_path = path.with_suffix(".meta.json")
        if not path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"No index at {path}")
        vectors = np.load(path)["vectors"]
        chunks = [Chunk(**item) for item in json.loads(meta_path.read_text(encoding="utf-8"))]
        return cls(vectors=vectors, chunks=chunks)
