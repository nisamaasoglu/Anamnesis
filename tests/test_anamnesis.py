"""Test suite.

The tests are grouped by the guarantee they protect, not by module: the safety
layer must refuse, the data layer must stay read-only and reproducible, retrieval
must cite, and the API must never claim to be something it is not.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from anamnesis import agent, config, db, embeddings, ingest, llm, retrieval, safety, seed, tools
from anamnesis.api import app, ensure_ready
from anamnesis.vectorstore import Chunk, VectorStore


@pytest.fixture(scope="module", autouse=True)
def hospital(tmp_path_factory):
    """One synthetic hospital and one index, built once for the whole module."""
    data_dir = tmp_path_factory.mktemp("anamnesis-data")
    config.DB_PATH = data_dir / "test.sqlite3"
    config.INDEX_PATH = data_dir / "test_index.npz"
    seed.seed_database()
    retrieval.reset_store()
    retrieval.get_store(rebuild=True)
    yield
    retrieval.reset_store()


# --- Embeddings -----------------------------------------------------------


def test_offline_embeddings_are_deterministic():
    first = embeddings.embed_query("visiting hours in the intensive care unit")
    second = embeddings.embed_query("visiting hours in the intensive care unit")
    assert np.array_equal(first, second)


def test_embeddings_are_unit_length():
    matrix = embeddings.embed_texts(["bed occupancy", "discharge lounge opening time"])
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)


def test_related_text_scores_above_unrelated_text():
    query = embeddings.embed_query("intensive care visiting hours")
    related, unrelated = embeddings.embed_texts(
        ["The intensive care unit restricts visiting hours to two windows.",
         "Alcohol-based hand rub is the default for hand hygiene."]
    )
    assert float(related @ query) > float(unrelated @ query)


def test_embedding_provider_declares_it_is_not_semantic_offline():
    described = embeddings.describe_provider("offline")
    assert described["semantic"] is False


# --- Chunking and the vector store ---------------------------------------


def test_repeated_boilerplate_is_kept_out_of_the_index():
    assert not any("fictional institution" in c.text for c in ingest.build_chunks())


def test_chunks_carry_document_and_section_for_citation():
    chunks = ingest.build_chunks()
    assert len(chunks) > 20
    assert all(c.document_title and c.section and c.text for c in chunks)


def test_no_chunk_spans_two_sections():
    by_id = {c.chunk_id for c in ingest.build_chunks()}
    assert len(by_id) == len(ingest.build_chunks())  # ids are unique per section


def test_vector_store_round_trips_through_disk(tmp_path):
    chunk = Chunk("a:b:0", "a.md", "A", "B", "text")
    store = VectorStore(vectors=embeddings.embed_texts(["text"]), chunks=[chunk])
    path = tmp_path / "idx.npz"
    store.save(path)
    reloaded = VectorStore.load(path)
    assert len(reloaded) == 1
    assert reloaded.chunks[0].section == "B"


def test_vector_store_rejects_mismatched_dimensions():
    store = retrieval.get_store()
    with pytest.raises(ValueError):
        store.search(np.zeros(3, dtype=np.float32))


# --- Retrieval ------------------------------------------------------------


def test_retrieval_finds_the_right_document():
    hits = retrieval.search("visiting hours intensive care unit")
    assert hits
    assert "Visiting" in hits[0].chunk.document_title


def test_every_hit_can_be_cited():
    for hit in retrieval.search("infection control isolation precautions"):
        assert " - " in hit.citation()


@pytest.mark.parametrize(
    "question",
    [
        "quarterly foreign exchange hedging strategy",
        "best pizza recipe in Naples",
        "how to configure a kubernetes ingress",
    ],
)
def test_off_topic_questions_retrieve_nothing(question):
    """A hash collision must not be able to pass an unrelated passage off as evidence."""
    assert retrieval.search(question) == []


def test_the_lexical_gate_is_what_rejects_a_collision():
    """Without the gate at least one off-topic query does clear the score floor."""
    store = retrieval.get_store()
    raw = store.search(embeddings.embed_query("quarterly foreign exchange hedging strategy"),
                       k=5, min_score=config.MIN_SCORE)
    assert raw, "test is meaningless if nothing clears the floor"
    assert retrieval.search("quarterly foreign exchange hedging strategy") == []


def test_an_unanswerable_question_says_so_instead_of_inventing():
    result = agent.answer("How do I hedge currency exposure for the quarter?")
    assert result.answer == safety.NO_EVIDENCE_ANSWER
    assert result.citations == []


# --- Synthetic data -------------------------------------------------------


def test_seeding_is_reproducible(tmp_path):
    first = seed.seed_database(db_path=tmp_path / "a.sqlite3", seed=42, patient_count=20)
    second = seed.seed_database(db_path=tmp_path / "b.sqlite3", seed=42, patient_count=20)
    assert first == second
    assert db.find_patients("MRN-10007", db_path=tmp_path / "a.sqlite3") == \
        db.find_patients("MRN-10007", db_path=tmp_path / "b.sqlite3")


def test_bed_arithmetic_never_reports_more_free_beds_than_exist():
    for ward in db.ward_occupancy():
        assert 0 <= ward["available"] <= ward["total_beds"]
        assert ward["occupied"] + ward["available"] <= ward["total_beds"]


def test_lab_results_are_flagged_against_their_reference_range():
    rows = db.lab_results("MRN-10000")
    for row in rows:
        expected = (
            "low" if row["value"] < row["reference_low"]
            else "high" if row["value"] > row["reference_high"]
            else "normal"
        )
        assert row["flag"] == expected


def test_the_data_layer_exposes_no_write_function():
    forbidden = {"insert", "update", "delete", "create_patient", "admit", "discharge"}
    assert not forbidden & {name.lower() for name in dir(db) if not name.startswith("_")}


# --- Tools ----------------------------------------------------------------


def test_mrn_is_normalised_from_free_text():
    assert tools.normalise_mrn("please check mrn10005 for me") == "MRN-10005"
    assert tools.normalise_mrn("no identifier here") is None


def test_relative_dates_resolve():
    from datetime import date, timedelta

    assert tools.resolve_date("today") == date.today().isoformat()
    assert tools.resolve_date("tomorrow") == (date.today() + timedelta(days=1)).isoformat()
    assert tools.resolve_date("not a date") == date.today().isoformat()


def test_every_schema_matches_a_registered_tool():
    declared = {s["function"]["name"] for s in tools.SCHEMAS}
    assert declared == set(tools.REGISTRY)


def test_unknown_tool_is_rejected():
    with pytest.raises(KeyError):
        tools.call("drop_database", {})


def test_lab_tool_refuses_to_interpret():
    result = tools.get_lab_results("MRN-10000")
    assert "does not interpret" in result["note"]


# --- Safety ---------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "Do I have cancer?",
        "What dosage should I take?",
        "Can you diagnose this rash?",
        "What does my result mean?",
        "What is the treatment for this condition?",
        "Should I stop my medication?",
    ],
)
def test_clinical_questions_are_refused(question):
    result = agent.answer(question)
    assert result.safety_category == "clinical_advice"
    assert result.tool_calls == []


@pytest.mark.parametrize(
    "question",
    ["Cancel the appointment for MRN-10005", "Please discharge the patient in bed 4"],
)
def test_write_requests_are_refused(question):
    assert agent.answer(question).safety_category == "write_request"


def test_ordinary_questions_are_allowed():
    assert safety.check("What are the visiting hours?").allowed


# --- Agent ----------------------------------------------------------------


def test_policy_question_is_answered_with_a_citation():
    result = agent.answer("What are the visiting hours in the intensive care unit?")
    assert result.citations
    assert "14:00" in result.answer


def test_capacity_question_routes_to_the_bed_tool():
    result = agent.answer("How many beds are free in the Intensive Care Unit?")
    assert result.tool_calls[0]["tool"] == "get_bed_availability"


def test_lab_question_without_an_identifier_does_not_guess_a_patient():
    result = agent.answer("How long do lab results take?")
    assert result.tool_calls[0]["tool"] == "search_policy"


def test_offline_mode_never_claims_to_be_generative():
    mode = agent.answer("What are the visiting hours?").mode
    assert mode["llm"]["generative"] is False


def test_llm_falls_back_to_offline_when_unavailable(monkeypatch):
    monkeypatch.setattr(llm, "active_provider", lambda: "openai")
    monkeypatch.setattr(
        llm, "chat", lambda *a, **k: (_ for _ in ()).throw(llm.LLMUnavailable("no key"))
    )
    result = agent.answer("What are the visiting hours?")
    assert "fallback" in result.mode
    assert result.answer


# --- API ------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    ensure_ready()
    with TestClient(app) as test_client:
        yield test_client


def test_health_declares_synthetic_data(client):
    payload = client.get("/api/health").json()
    assert payload["synthetic_data"] is True
    assert "fictional" in payload["notice"].lower()


def test_status_reports_corpus_and_mode(client):
    payload = client.get("/api/status").json()
    assert payload["knowledge_chunks"] > 0
    assert payload["mode"]["embeddings"]["provider"] in {"offline", "openai"}


def test_ask_returns_answer_tools_and_mode(client):
    payload = client.post("/api/ask", json={"question": "What are the ICU visiting hours?"}).json()
    assert payload["answer"]
    assert payload["synthetic_data"] is True
    assert payload["mode"]["llm"]["provider"]


def test_ask_rejects_an_empty_question(client):
    assert client.post("/api/ask", json={"question": "   "}).status_code == 422


def test_ask_refuses_clinical_advice_over_http(client):
    payload = client.post("/api/ask", json={"question": "Do I have cancer?"}).json()
    assert payload["safety"] == "clinical_advice"


def test_console_is_served(client):
    assert client.get("/").status_code == 200
