"""Deterministic synthetic record generator.

Everything is drawn from a seeded PRNG, so the same SEED always produces the
same hospital. That matters twice over: the demo is reproducible for anyone who
clones the repository, and the tests can assert on concrete values instead of
on "something that looks like a patient".

The names are drawn from a fixed invented list. No real person, and no real
patient record, is represented anywhere in this project.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from pathlib import Path

from . import config, db

FIRST_NAMES = [
    "Amara", "Bruno", "Celine", "Dario", "Elin", "Farid", "Greta", "Hugo",
    "Ilse", "Jonas", "Katria", "Lorcan", "Mira", "Nadim", "Orla", "Petros",
    "Quilla", "Rasmus", "Sana", "Tomas", "Ursula", "Vasco", "Wren", "Yara",
]
LAST_NAMES = [
    "Adeyemi", "Bellucci", "Castellan", "Duarte", "Eriksen", "Fontaine",
    "Gallagher", "Halvorsen", "Ibsen", "Janssen", "Kowalczyk", "Lindqvist",
    "Moreau", "Novak", "Okonkwo", "Pereira", "Rahman", "Svensson", "Tanaka",
    "Vasquez",
]
WARDS = [
    ("Aster Ward", "General Medicine", 28, 2),
    ("Birch Ward", "General Surgery", 24, 0),
    ("Cedar Ward", "Cardiology", 20, 1),
    ("Dogwood Ward", "Paediatrics", 18, 0),
    ("Elm Ward", "Orthopaedics", 22, 2),
    ("Intensive Care Unit", "Critical Care", 12, 1),
    ("Maternity Unit", "Obstetrics", 16, 0),
]
DEPARTMENTS = [
    "Cardiology", "Dermatology", "Endocrinology", "General Medicine",
    "Neurology", "Obstetrics", "Ophthalmology", "Orthopaedics", "Paediatrics",
]
CLINICIANS = [
    "Dr. A. Lindqvist", "Dr. B. Okonkwo", "Dr. C. Moreau", "Dr. D. Tanaka",
    "Dr. E. Halvorsen", "Dr. F. Pereira", "Dr. G. Novak", "Dr. H. Rahman",
]
# (panel, analyte, unit, reference low, reference high, plausible spread)
PANELS = [
    ("Full Blood Count", "Haemoglobin", "g/dL", 12.0, 16.0, (9.5, 17.5)),
    ("Full Blood Count", "White Cell Count", "10^9/L", 4.0, 11.0, (2.5, 16.0)),
    ("Full Blood Count", "Platelets", "10^9/L", 150.0, 400.0, (90.0, 520.0)),
    ("Basic Metabolic Panel", "Sodium", "mmol/L", 135.0, 145.0, (128.0, 150.0)),
    ("Basic Metabolic Panel", "Potassium", "mmol/L", 3.5, 5.1, (2.9, 6.0)),
    ("Basic Metabolic Panel", "Creatinine", "mg/dL", 0.6, 1.3, (0.4, 3.2)),
    ("Liver Function", "ALT", "U/L", 7.0, 56.0, (5.0, 190.0)),
    ("Liver Function", "Bilirubin", "mg/dL", 0.1, 1.2, (0.1, 4.5)),
]


def _mrn(index: int) -> str:
    return f"MRN-{10000 + index}"


def seed_database(
    db_path: Path | None = None,
    seed: int | None = None,
    patient_count: int | None = None,
    reference_date: date | None = None,
) -> dict[str, int]:
    """Create the schema and fill it with a reproducible synthetic hospital."""
    path = Path(db_path or config.DB_PATH)
    rng = random.Random(seed if seed is not None else config.SEED)
    total_patients = patient_count or config.SYNTHETIC_PATIENTS
    today = reference_date or date.today()

    db.initialise(path)
    with db.connect(path) as conn:
        conn.executescript(
            "DELETE FROM lab_results; DELETE FROM appointments; "
            "DELETE FROM patients; DELETE FROM wards;"
        )
        conn.executemany("INSERT INTO wards VALUES (?, ?, ?, ?)", WARDS)

        patients = []
        for i in range(total_patients):
            name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
            born = date(rng.randint(1940, 2020), rng.randint(1, 12), rng.randint(1, 28))
            admitted = rng.random() < 0.45
            ward = rng.choice(WARDS)[0] if admitted else None
            patients.append(
                (
                    _mrn(i),
                    name,
                    born.isoformat(),
                    rng.choice(["F", "M"]),
                    ward,
                    f"{rng.randint(1, 30):02d}" if admitted else None,
                    "admitted" if admitted else "outpatient",
                    (today - timedelta(days=rng.randint(0, 14))).isoformat() if admitted else None,
                )
            )
        conn.executemany("INSERT INTO patients VALUES (?, ?, ?, ?, ?, ?, ?, ?)", patients)

        appointments = []
        for mrn, *_ in patients:
            for _ in range(rng.randint(0, 2)):
                start = datetime.combine(
                    today + timedelta(days=rng.randint(0, 6)),
                    datetime.min.time(),
                ) + timedelta(hours=rng.randint(9, 16), minutes=rng.choice([0, 15, 30, 45]))
                appointments.append(
                    (
                        mrn,
                        rng.choice(DEPARTMENTS),
                        rng.choice(CLINICIANS),
                        start.isoformat(timespec="minutes"),
                        rng.choices(
                            ["booked", "checked-in", "cancelled"], weights=[70, 20, 10]
                        )[0],
                    )
                )
        conn.executemany(
            "INSERT INTO appointments (mrn, department, clinician, starts_at, status) "
            "VALUES (?, ?, ?, ?, ?)",
            appointments,
        )

        results = []
        for mrn, *rest in patients:
            if rest[5] != "admitted" and rng.random() < 0.6:
                continue
            collected = datetime.combine(today, datetime.min.time()) - timedelta(
                hours=rng.randint(1, 72)
            )
            for panel, analyte, unit, low, high, (floor, ceiling) in PANELS:
                # Most results sit inside the reference range; a minority do not.
                if rng.random() < 0.78:
                    value = rng.uniform(low, high)
                else:
                    value = rng.choice([rng.uniform(floor, low), rng.uniform(high, ceiling)])
                results.append(
                    (
                        mrn,
                        panel,
                        analyte,
                        round(value, 2),
                        unit,
                        low,
                        high,
                        collected.isoformat(timespec="minutes"),
                        "final",
                    )
                )
        conn.executemany(
            "INSERT INTO lab_results (mrn, panel, analyte, value, unit, reference_low, "
            "reference_high, collected_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            results,
        )

    return db.counts(path)
