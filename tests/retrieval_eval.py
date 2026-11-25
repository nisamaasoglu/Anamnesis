"""Retrieval benchmark: one realistic question per knowledge-base section.

Run directly to print top-1 and top-3 accuracy:

    python -m tests.retrieval_eval

The questions are phrased the way a patient or visitor would ask them, not by
copying section headings, so they exercise synonyms, abbreviations and
inflected word forms.
"""

from __future__ import annotations

CASES: list[tuple[str, str]] = [
    ("When can I come in for a planned operation?", "Planned Admission"),
    (
        "What happens if I am admitted through the emergency department?",
        "Emergency Admission",
    ),
    ("When will I be allowed to go home?", "Discharge Criteria"),
    ("What time of day are patients discharged?", "Discharge Timing"),
    ("Can I be moved to another ward?", "Transfers Between Wards"),
    ("How does the ER decide who is seen first?", "Triage Categories"),
    ("How often are waiting patients checked again in A&E?", "Reassessment"),
    ("Who is called if a patient gets worse in the emergency room?", "Escalation"),
    ("How long will I wait in the emergency department?", "Information for Waiting Patients"),
    (
        "What happens if a child comes to the ER alone?",
        "Unaccompanied Minors and Vulnerable Adults",
    ),
    ("Do I have the right to be told about my treatment?", "Right to Information"),
    ("Do I need to sign a consent form before surgery?", "Consent"),
    ("How can I get a copy of my medical records?", "Access to Records"),
    ("Who can see my health information?", "Confidentiality"),
    ("How do I make a complaint?", "Complaints and Feedback"),
    ("What are the visiting hours?", "General Visiting Hours"),
    ("Can I visit the ICU?", "Intensive Care Unit"),
    ("Can a partner stay on the maternity ward?", "Maternity and Neonatal Units"),
    ("Can children visit the paediatric ward?", "Paediatric Ward"),
    (
        "Can I visit outside normal hours if a relative is dying?",
        "Exceptions and Compassionate Access",
    ),
    ("How do I book an outpatient appointment?", "Booking Channels"),
    ("How early should I arrive for my appointment?", "Arrival and Registration"),
    ("How do I cancel or move my appointment?", "Cancellation and Rescheduling"),
    ("Can I get an interpreter for my appointment?", "Interpreters and Accessibility"),
    ("Will I get a text reminder before my appointment?", "Appointment Reminders"),
    ("When do staff have to wash their hands?", "Hand Hygiene"),
    ("When do nurses wear gloves, gowns and masks?", "Personal Protective Equipment"),
    ("Why is my relative in an isolation room?", "Isolation Precautions"),
    ("Can I visit during a norovirus outbreak?", "Visitor Restrictions During an Outbreak"),
    ("How are rooms cleaned after an infected patient leaves?", "Cleaning and Decontamination"),
    ("What time does the lab open for blood tests?", "Laboratory Opening Hours"),
    ("How long do blood test results take?", "Turnaround Targets"),
    ("What happens if my lab result is dangerously abnormal?", "Critical Result Notification"),
    ("Where do I collect my prescription medicines?", "Outpatient Pharmacy"),
    ("How can I see my test results?", "Result Access for Patients"),
]


Miss = tuple[str, str, list[str]]


def evaluate(
    k: int = 3, cases: list[tuple[str, str]] | None = None
) -> tuple[float, float, list[Miss]]:
    from anamnesis import retrieval

    cases = CASES if cases is None else cases
    top1 = topk = 0
    misses: list[Miss] = []
    for question, expected in cases:
        sections = [hit.chunk.section for hit in retrieval.search(question, k=k)]
        if sections[:1] == [expected]:
            top1 += 1
        if expected in sections:
            topk += 1
        else:
            misses.append((question, expected, sections))
    n = len(cases)
    return top1 / n, topk / n, misses




HELD_OUT: list[tuple[str, str]] = [
    ("How many people can visit a patient at once?", "General Visiting Hours"),
    ("Can my 10 year old come into intensive care?", "Intensive Care Unit"),
    ("Can the father visit the newborn unit?", "Maternity and Neonatal Units"),
    ("Can a parent sleep next to their sick child overnight?", "Paediatric Ward"),
    (
        "What do I need to bring when I am admitted for a scheduled procedure?",
        "Planned Admission",
    ),
    ("Who decides if a patient is well enough to leave?", "Discharge Criteria"),
    ("What does level two urgency mean in A&E?", "Triage Categories"),
    ("Can I phone to book a clinic visit?", "Booking Channels"),
    ("What if I miss my appointment?", "Cancellation and Rescheduling"),
    ("Is there wheelchair access at the clinic?", "Interpreters and Accessibility"),
    ("Do doctors need my permission before treating me?", "Consent"),
    ("Can my employer find out my diagnosis?", "Confidentiality"),
    ("Is the pharmacy open at weekends?", "Outpatient Pharmacy"),
    ("Who tells the doctor about a critical potassium level?", "Critical Result Notification"),
    (
        "Are visitors allowed when there is an infection outbreak on the ward?",
        "Visitor Restrictions During an Outbreak",
    ),
]



if __name__ == "__main__":
    for name, cases in (("tuning set", CASES), ("held-out set", HELD_OUT)):
        at1, at3, missed = evaluate(cases=cases)
        print(f"{name}: n={len(cases)}  top-1={at1:.0%}  top-3={at3:.0%}")
        for question, expected, got in missed:
            print(f"  MISS  {question!r}\n        expected {expected!r}, got {got}")
