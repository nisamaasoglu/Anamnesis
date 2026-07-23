"""The agent loop: safety -> plan -> tools -> grounded answer.

Two execution paths share one tool registry and one safety layer:

* **LLM path** — a standard tool-calling loop, capped at ``MAX_TOOL_STEPS``. The
  model is given only the tool results and the retrieved passages, and the system
  prompt forbids answering from anything else.
* **Offline path** — a deterministic keyword router picks exactly one tool and the
  answer is assembled from templates. This is not a language model and is never
  presented as one; ``mode.llm.generative`` is ``false`` in every response.

Having both means the repository is runnable and testable with no API key, while
the production path stays honest about what produced the text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from . import config, llm, safety, tools
from .embeddings import describe_provider as describe_embeddings

# --- Offline router -------------------------------------------------------
# Each intent is a (tool, keyword weights) pair. The highest-scoring intent wins;
# a tie or a zero score falls through to policy search.
_INTENTS: list[tuple[str, dict[str, int]]] = [
    ("get_bed_availability", {"bed": 3, "beds": 3, "occupancy": 3, "capacity": 3,
                              "free": 1, "available": 2, "full": 1, "ward": 1}),
    ("get_appointments", {"appointment": 3, "appointments": 3, "clinic": 2,
                          "scheduled": 2, "booked": 1, "today": 1, "tomorrow": 2}),
    ("get_lab_results", {"lab": 3, "labs": 3, "result": 2, "results": 2,
                         "bloods": 3, "haemoglobin": 3, "potassium": 3, "panel": 2}),
    ("find_patient", {"patient": 2, "mrn": 4, "record": 1, "admitted": 1, "who": 1}),
]


def _route(question: str) -> tuple[str, dict[str, Any]]:
    text = question.lower()
    words = set(text.replace("?", " ").replace(",", " ").split())
    best_tool, best_score = "search_policy", 0
    for tool_name, weights in _INTENTS:
        score = sum(weight for word, weight in weights.items() if word in words)
        if score > best_score:
            best_tool, best_score = tool_name, score

    mrn = tools.normalise_mrn(question)
    if best_tool == "get_lab_results" and not mrn:
        # A lab question with no patient identifier cannot be answered from data.
        return "search_policy", {"query": question}
    if best_tool == "get_lab_results":
        return best_tool, {"mrn": mrn}
    if best_tool == "find_patient":
        return best_tool, {"query": mrn or question}
    if best_tool == "get_appointments":
        for expression in ("tomorrow", "yesterday", "today"):
            if expression in text:
                return best_tool, {"date_expression": expression}
        return best_tool, {"date_expression": "today"}
    if best_tool == "get_bed_availability":
        for ward in ("intensive care", "maternity", "aster", "birch", "cedar",
                     "dogwood", "elm"):
            if ward in text:
                full = "Intensive Care Unit" if ward == "intensive care" else (
                    "Maternity Unit" if ward == "maternity" else f"{ward.title()} Ward"
                )
                return best_tool, {"ward": full}
        return best_tool, {}
    return "search_policy", {"query": question}


# --- Offline answer templates --------------------------------------------


def _render_offline(tool_name: str, arguments: dict[str, Any], result: dict[str, Any]) -> str:
    if tool_name == "search_policy":
        matches = result.get("matches", [])
        if not matches:
            return safety.NO_EVIDENCE_ANSWER
        top = matches[0]
        body = " ".join(top["text"].split())
        if len(body) > 600:
            body = body[:600].rsplit(" ", 1)[0] + " ..."
        return f"From {top['document']}, section \"{top['section']}\":\n\n{body}"

    if tool_name == "get_bed_availability":
        wards = result.get("wards", [])
        if not wards:
            return "No ward matched that name."
        lines = [
            f"- {w['name']} ({w['speciality']}): {w['occupied']} of "
            f"{w['total_beds'] - w['closed_beds']} usable beds occupied "
            f"({w['occupancy_pct']}%), {w['available']} free"
            for w in wards
        ]
        header = (
            f"Bed position across {len(wards)} wards - {result['occupied']} occupied, "
            f"{result['available']} free of {result['total_beds']} beds:"
            if len(wards) > 1
            else "Bed position:"
        )
        return header + "\n" + "\n".join(lines)

    if tool_name == "get_appointments":
        rows = result.get("appointments", [])
        if not rows:
            return f"No appointments are recorded for {result['date']}."
        preview = "\n".join(
            f"- {r['starts_at'][11:]} {r['department']} with {r['clinician']} "
            f"({r['full_name']}, {r['status']})"
            for r in rows[:8]
        )
        more = f"\n... and {len(rows) - 8} more." if len(rows) > 8 else ""
        return f"{result['count']} appointments on {result['date']}:\n{preview}{more}"

    if tool_name == "get_lab_results":
        rows = result.get("results", [])
        if not rows:
            return f"No released laboratory results are recorded for {result['mrn']}."
        lines = [
            f"- {r['analyte']}: {r['value']} {r['unit']} "
            f"(reference {r['reference_low']}-{r['reference_high']}, {r['flag']})"
            for r in rows[:10]
        ]
        return (
            f"Released results for {result['mrn']} - values only, no interpretation:\n"
            + "\n".join(lines)
            + "\n\nAny interpretation of these values is a matter for the responsible "
            "clinician."
        )

    if tool_name == "find_patient":
        rows = result.get("patients", [])
        if not rows:
            return "No patient matched that identifier or name."
        lines = [
            f"- {p['full_name']} ({p['mrn']}), born {p['date_of_birth']}, {p['status']}"
            + (f", {p['ward']} bed {p['bed']}" if p["ward"] else "")
            for p in rows
        ]
        return "Matching synthetic records:\n" + "\n".join(lines)

    return json.dumps(result)[:800]


# --- Public result type ---------------------------------------------------


@dataclass
class AgentResult:
    question: str
    answer: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    safety_category: str = "ok"
    mode: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "tool_calls": self.tool_calls,
            "citations": self.citations,
            "safety": self.safety_category,
            "mode": self.mode,
            "synthetic_data": True,
        }


def _current_mode() -> dict[str, Any]:
    return {"llm": llm.describe_provider(), "embeddings": describe_embeddings()}


def _citations_from(tool_name: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    if tool_name != "search_policy":
        return []
    return [
        {"document": m["document"], "section": m["section"], "score": m["score"]}
        for m in result.get("matches", [])
    ]


def answer(question: str) -> AgentResult:
    """Answer one question end to end. This is the only entry point callers need."""
    verdict = safety.check(question)
    if not verdict.allowed:
        return AgentResult(
            question=question,
            answer=verdict.message,
            safety_category=verdict.category,
            mode=_current_mode(),
        )

    if llm.active_provider() == "offline":
        return _answer_offline(question)
    try:
        return _answer_with_llm(question)
    except llm.LLMUnavailable as exc:
        result = _answer_offline(question)
        result.mode["fallback"] = f"LLM unavailable, answered offline ({exc})"
        return result


def _answer_offline(question: str) -> AgentResult:
    tool_name, arguments = _route(question)
    result = tools.call(tool_name, arguments)
    return AgentResult(
        question=question,
        answer=_render_offline(tool_name, arguments, result),
        tool_calls=[{"tool": tool_name, "arguments": arguments}],
        citations=_citations_from(tool_name, result),
        mode=_current_mode(),
    )


def _answer_with_llm(question: str) -> AgentResult:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": safety.SYSTEM_PROMPT.format(hospital=config.HOSPITAL_NAME)},
        {"role": "user", "content": question},
    ]
    executed: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []

    for _ in range(config.MAX_TOOL_STEPS):
        response = llm.chat(messages, tools=tools.SCHEMAS)
        if not response.tool_calls:
            text = response.content.strip() or safety.NO_EVIDENCE_ANSWER
            return AgentResult(
                question=question,
                answer=text,
                tool_calls=executed,
                citations=citations,
                mode=_current_mode(),
            )

        messages.append(
            {
                "role": "assistant",
                "content": response.content or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in response.tool_calls
                ],
            }
        )
        for call in response.tool_calls:
            try:
                result = tools.call(call.name, call.arguments)
            except (KeyError, TypeError) as exc:
                result = {"error": str(exc)}
            executed.append({"tool": call.name, "arguments": call.arguments})
            citations.extend(_citations_from(call.name, result))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str)[:6000],
                }
            )

    # Step budget exhausted: answer from what was gathered rather than looping.
    messages.append(
        {
            "role": "user",
            "content": "Answer now from the tool results already gathered, with citations.",
        }
    )
    final = llm.chat(messages)
    return AgentResult(
        question=question,
        answer=final.content.strip() or safety.NO_EVIDENCE_ANSWER,
        tool_calls=executed,
        citations=citations,
        mode=_current_mode(),
    )
