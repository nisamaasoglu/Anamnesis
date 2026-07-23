"""Turn the Markdown policy library into a searchable index.

Chunking follows the document's own structure first (one chunk never spans two
``##`` sections) and only then falls back to a fixed word window with overlap.
Keeping section boundaries intact is what makes a citation meaningful: every
answer can name the document *and* the clause it came from.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import config
from .embeddings import embed_texts
from .vectorstore import Chunk, VectorStore

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")
# The fictional-hospital disclaimer is repeated verbatim in every document. It is
# essential on screen and useless in an index: identical boilerplate in every file
# adds no retrieval signal and dilutes the score of the text around it.
_BOILERPLATE_RE = re.compile(r"^\*.*fictional institution.*\*$", re.IGNORECASE)


def _read_sections(path: Path) -> tuple[str, list[tuple[str, str]]]:
    """Split one Markdown file into (title, [(section_heading, body), ...])."""
    title = path.stem.replace("-", " ").replace("_", " ").title()
    sections: list[tuple[str, list[str]]] = []
    current = "Overview"
    buffer: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        match = _HEADING_RE.match(line.strip())
        if match:
            level, heading = len(match.group(1)), match.group(2).strip()
            if level == 1:
                title = heading
                continue
            if buffer:
                sections.append((current, buffer))
            current, buffer = heading, []
            continue
        if _BOILERPLATE_RE.match(line.strip()):
            continue
        buffer.append(line)

    if buffer:
        sections.append((current, buffer))
    return title, [
        (name, "\n".join(body).strip()) for name, body in sections if "".join(body).strip()
    ]


def _window(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text] if words else []
    step = max(size - overlap, 1)
    windows = (words[i : i + size] for i in range(0, len(words), step))
    return [" ".join(w) for w in windows if w]


def build_chunks(knowledge_dir: Path | None = None) -> list[Chunk]:
    directory = Path(knowledge_dir or config.KNOWLEDGE_DIR)
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        title, sections = _read_sections(path)
        for section_name, body in sections:
            for position, piece in enumerate(
                _window(body, config.CHUNK_WORDS, config.CHUNK_OVERLAP_WORDS)
            ):
                chunks.append(
                    Chunk(
                        chunk_id=f"{path.stem}:{section_name}:{position}",
                        document=path.name,
                        document_title=title,
                        section=section_name,
                        text=piece,
                    )
                )
    return chunks


def build_index(knowledge_dir: Path | None = None, index_path: Path | None = None) -> VectorStore:
    """Chunk, embed and persist the knowledge base. Returns the store."""
    chunks = build_chunks(knowledge_dir)
    if not chunks:
        raise ValueError(f"No Markdown documents found in {knowledge_dir or config.KNOWLEDGE_DIR}")
    # The section heading is prepended so a heading-only match still scores.
    vectors = embed_texts([f"{c.document_title}. {c.section}. {c.text}" for c in chunks])
    store = VectorStore(vectors=vectors, chunks=chunks)
    store.save(Path(index_path or config.INDEX_PATH))
    return store
