"""The guardrail layer that every question passes through first.

A hospital information assistant is dangerous in a specific way: it is fluent
enough to sound like a clinician. The whole design choice here is that it is
scoped to *information about the hospital* — policies, schedules, capacity — and
never to clinical judgement about a person.

Three rules are enforced, in order:

1. **No clinical advice.** Diagnosis, treatment choice, dosing and prognosis are
   refused outright and escalated to a human clinician, however the question is
   phrased.
2. **No writes.** The assistant has no tool that can alter a record. A request to
   book, cancel or amend is answered with the correct human route instead.
3. **Grounded or silent.** If retrieval returns nothing above the score floor,
   the answer says so. The system does not fall back to general knowledge, because
   an invented hospital policy is worse than no answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CLINICAL_ADVICE_PATTERNS = [
    r"\bdiagnos(e|is|ing)\b",
    r"\bwhat('?s| is) wrong with\b",
    r"\bdo i have\b",
    r"\bis (it|this) (cancer|a tumou?r|serious|fatal)\b",
    r"\bshould (i|we|he|she|they) (take|stop|start|switch)\b",
    r"\b(dose|dosage|dosing|how (much|many) (mg|ml|tablets?|pills?))\b",
    r"\bprescrib(e|ing)\b",
    r"\btreatment (for|of|plan)\b",
    r"\bhow (do|should) (i|we) treat\b",
    r"\blife expectancy|prognosis|survival (rate|odds|chance)\b",
    r"\binterpret (my|the|these) (results?|labs?|scan)\b",
    r"\bwhat does (my|this) result mean\b",
]

WRITE_PATTERNS = [
    r"\b(book|schedule|rebook|reschedule|cancel|delete|discharge|admit|update|amend|change)\b"
    r"[^?.!]{0,40}\b(appointment|patient|record|bed|admission)\b",
    r"\b(add|remove|edit)\b[^?.!]{0,30}\b(record|note|allergy|medication)\b",
]

_CLINICAL_RE = [re.compile(p, re.IGNORECASE) for p in CLINICAL_ADVICE_PATTERNS]
_WRITE_RE = [re.compile(p, re.IGNORECASE) for p in WRITE_PATTERNS]

CLINICAL_REFUSAL = (
    "I can't help with that one. This assistant answers questions about how the "
    "hospital works — policies, schedules, capacity and where to find things — and "
    "it is deliberately not able to give clinical advice, interpret results for a "
    "person, or comment on diagnosis, treatment or prognosis. Please speak to the "
    "responsible clinician, or contact the ward directly. In an emergency, use the "
    "emergency department or the hospital's internal emergency number."
)

WRITE_REFUSAL = (
    "This assistant is read-only by design, so it can't create or change any record. "
    "Appointment changes go through the patient portal or the central appointment "
    "line; admissions, transfers and discharges are handled by the bed management "
    "desk and the responsible clinical team."
)

NO_EVIDENCE_ANSWER = (
    "I couldn't find anything in the hospital's document library that answers that. "
    "Rather than guess, I'd rather tell you plainly: this question needs a human. "
    "The ward manager or the relevant department office is the right next step."
)


@dataclass(frozen=True)
class SafetyVerdict:
    allowed: bool
    category: str  # "ok" | "clinical_advice" | "write_request"
    message: str = ""


def check(question: str) -> SafetyVerdict:
    """Classify a question before any retrieval or tool call happens."""
    text = (question or "").strip()
    if not text:
        return SafetyVerdict(False, "empty", "Please ask a question.")
    for pattern in _CLINICAL_RE:
        if pattern.search(text):
            return SafetyVerdict(False, "clinical_advice", CLINICAL_REFUSAL)
    for pattern in _WRITE_RE:
        if pattern.search(text):
            return SafetyVerdict(False, "write_request", WRITE_REFUSAL)
    return SafetyVerdict(True, "ok")


SYSTEM_PROMPT = """You are Anamnesis, the information assistant for {hospital}.

Rules you must follow without exception:
- Answer only from the CONTEXT passages and TOOL RESULTS you are given. If they do
  not contain the answer, say so and point the user to a human. Never rely on
  general knowledge about other hospitals.
- Cite the document and section for every factual claim taken from CONTEXT, using
  the bracketed numbers provided.
- Never give clinical advice: no diagnosis, no treatment or medication guidance, no
  interpretation of a person's results, no prognosis. Refer to the responsible
  clinician instead.
- You are read-only. You cannot book, cancel or change anything.
- All records in this system are synthetic. Do not claim otherwise.
- Be brief and concrete. Prefer exact times, numbers and named routes.
"""
