"""The tool surface the agent is allowed to call.

Each tool is an ordinary Python function with a JSON schema attached, so the same
definitions drive three things: the OpenAI tool-calling loop, the offline
deterministic router, and the tests. Adding a tool in one place adds it
everywhere.

All five tools are read-only queries.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from . import db, retrieval

_MRN_RE = re.compile(r"\bMRN-?\d{4,6}\b", re.IGNORECASE)


def normalise_mrn(text: str) -> str | None:
    match = _MRN_RE.search(text or "")
    if not match:
        return None
    digits = re.sub(r"\D", "", match.group(0))
    return f"MRN-{digits}"


def resolve_date(expression: str | None) -> str:
    """Accept 'today', 'tomorrow', 'yesterday' or an ISO date; return ISO."""
    today = date.today()
    value = (expression or "today").strip().lower()
    if value in ("", "today"):
        return today.isoformat()
    if value == "tomorrow":
        return (today + timedelta(days=1)).isoformat()
    if value == "yesterday":
        return (today - timedelta(days=1)).isoformat()
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return today.isoformat()


# --- Tool implementations -------------------------------------------------


def search_policy(query: str, k: int = 4) -> dict[str, Any]:
    """Search the hospital's policy and procedure library."""
    hits = retrieval.search(query, k=k)
    return {
        "matches": [
            {
                "document": hit.chunk.document_title,
                "section": hit.chunk.section,
                "text": hit.chunk.text,
                "score": round(hit.score, 4),
            }
            for hit in hits
        ]
    }


def find_patient(query: str) -> dict[str, Any]:
    """Look up a patient by medical record number or name (synthetic records)."""
    return {"patients": db.find_patients(query)}


def get_appointments(
    date_expression: str = "today", department: str | None = None
) -> dict[str, Any]:
    """List appointments for a date, optionally filtered by department."""
    iso = resolve_date(date_expression)
    rows = db.appointments_on(iso, department)
    return {"date": iso, "department": department, "count": len(rows), "appointments": rows}


def get_bed_availability(ward: str | None = None) -> dict[str, Any]:
    """Report bed occupancy and free beds, per ward or hospital-wide."""
    rows = db.ward_occupancy(ward)
    total = sum(r["total_beds"] for r in rows)
    occupied = sum(r["occupied"] for r in rows)
    available = sum(r["available"] for r in rows)
    return {"wards": rows, "total_beds": total, "occupied": occupied, "available": available}


def get_lab_results(mrn: str, panel: str | None = None) -> dict[str, Any]:
    """Return released laboratory values for a patient, with reference ranges.

    Values are returned with their reference range and a low/normal/high flag.
    No interpretation is produced; that is a clinician's job and the safety layer
    refuses requests for it.
    """
    normalised = normalise_mrn(mrn) or mrn.strip().upper()
    rows = db.lab_results(normalised, panel)
    return {
        "mrn": normalised,
        "panel": panel,
        "count": len(rows),
        "results": rows,
        "note": "Values only. This system does not interpret results.",
    }


REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "search_policy": search_policy,
    "find_patient": find_patient,
    "get_appointments": get_appointments,
    "get_bed_availability": get_bed_availability,
    "get_lab_results": get_lab_results,
}

SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": (
                "Search the hospital policy and procedure library: visiting hours, "
                "admission and discharge, infection control, triage, patient rights, "
                "pharmacy and laboratory service standards."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for."},
                    "k": {"type": "integer", "description": "How many passages to return."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_patient",
            "description": "Find a synthetic patient record by medical record number or name.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_appointments",
            "description": "List appointments for a given day, optionally by department.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_expression": {
                        "type": "string",
                        "description": "'today', 'tomorrow', 'yesterday' or an ISO date.",
                    },
                    "department": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_bed_availability",
            "description": "Bed occupancy and free beds for one ward or for the whole hospital.",
            "parameters": {
                "type": "object",
                "properties": {"ward": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lab_results",
            "description": (
                "Released laboratory values for a patient with reference ranges. "
                "Returns values only; it does not interpret them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mrn": {"type": "string"},
                    "panel": {"type": "string"},
                },
                "required": ["mrn"],
            },
        },
    },
]


def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call by name, raising on anything not in the registry."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown tool: {name}")
    return REGISTRY[name](**arguments)
