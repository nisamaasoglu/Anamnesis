"""FastAPI application: a small JSON API plus the single-page console.

The response envelope is deliberately verbose. Every answer carries the tools
that were called, the passages it was grounded in, the safety verdict, and which
providers produced it — so a reviewer can always see *why* an answer says what it
says, instead of taking the text on trust.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import __version__, agent, config, db, embeddings, llm, retrieval, seed

WEB_DIR = config.BASE_DIR / "web"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


def ensure_ready(force: bool = False) -> dict[str, int]:
    """Create the database and the vector index if they are not there yet."""
    if force or not Path(config.DB_PATH).exists():
        seed.seed_database()
    if force or not Path(config.INDEX_PATH).exists():
        retrieval.reset_store()
        retrieval.get_store(rebuild=True)
    return db.counts()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_ready()
    yield


app = FastAPI(
    title="Anamnesis",
    version=__version__,
    description=(
        f"Retrieval-grounded information assistant for {config.HOSPITAL_NAME} "
        "(fictional). All records are synthetic."
    ),
    lifespan=lifespan,
)


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "version": __version__,
        "hospital": config.HOSPITAL_NAME,
        "synthetic_data": True,
        "notice": config.SYNTHETIC_DATA_NOTICE,
    }


@app.get("/api/status")
def status() -> dict[str, object]:
    store = retrieval.get_store()
    return {
        "hospital": config.HOSPITAL_NAME,
        "records": db.counts(),
        "knowledge_chunks": len(store),
        "documents": sorted({chunk.document_title for chunk in store.chunks}),
        "mode": {
            "llm": llm.describe_provider(),
            "embeddings": embeddings.describe_provider(),
        },
        "notice": config.SYNTHETIC_DATA_NOTICE,
    }


@app.post("/api/ask")
def ask(request: AskRequest) -> dict[str, object]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")
    return agent.answer(question).to_dict()


@app.post("/api/reindex")
def reindex() -> dict[str, object]:
    counts = ensure_ready(force=True)
    return {"reindexed": True, "records": counts, "chunks": len(retrieval.get_store())}


@app.get("/")
def console() -> FileResponse:
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Console not found.")
    return FileResponse(index)
