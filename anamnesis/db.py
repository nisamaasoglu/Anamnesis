"""SQLite layer over the synthetic hospital record set.

Every function here is read-only by design. The assistant is an information
system, not a record system: there is deliberately no code path that can create,
amend or delete a patient record, and that constraint is asserted in the tests.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    mrn           TEXT PRIMARY KEY,
    full_name     TEXT NOT NULL,
    date_of_birth TEXT NOT NULL,
    sex           TEXT NOT NULL,
    ward          TEXT,
    bed           TEXT,
    status        TEXT NOT NULL,
    admitted_on   TEXT
);
CREATE TABLE IF NOT EXISTS appointments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    mrn         TEXT NOT NULL,
    department  TEXT NOT NULL,
    clinician   TEXT NOT NULL,
    starts_at   TEXT NOT NULL,
    status      TEXT NOT NULL,
    FOREIGN KEY (mrn) REFERENCES patients (mrn)
);
CREATE TABLE IF NOT EXISTS wards (
    name        TEXT PRIMARY KEY,
    speciality  TEXT NOT NULL,
    total_beds  INTEGER NOT NULL,
    closed_beds INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS lab_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    mrn          TEXT NOT NULL,
    panel        TEXT NOT NULL,
    analyte      TEXT NOT NULL,
    value        REAL NOT NULL,
    unit         TEXT NOT NULL,
    reference_low  REAL NOT NULL,
    reference_high REAL NOT NULL,
    collected_at TEXT NOT NULL,
    status       TEXT NOT NULL,
    FOREIGN KEY (mrn) REFERENCES patients (mrn)
);
CREATE INDEX IF NOT EXISTS idx_appointments_start ON appointments (starts_at);
CREATE INDEX IF NOT EXISTS idx_labs_mrn ON lab_results (mrn);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialise(db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def find_patients(query: str, db_path: Path | None = None, limit: int = 5) -> list[dict]:
    """Look a patient up by medical record number or by part of their name."""
    with connect(db_path) as conn:
        exact = _rows(conn, "SELECT * FROM patients WHERE mrn = ?", (query.strip().upper(),))
        if exact:
            return exact
        return _rows(
            conn,
            "SELECT * FROM patients WHERE full_name LIKE ? ORDER BY full_name LIMIT ?",
            (f"%{query.strip()}%", limit),
        )


def appointments_on(
    date: str, department: str | None = None, db_path: Path | None = None
) -> list[dict]:
    """All appointments on an ISO date, optionally filtered by department."""
    sql = (
        "SELECT a.*, p.full_name FROM appointments a JOIN patients p ON p.mrn = a.mrn "
        "WHERE date(a.starts_at) = date(?)"
    )
    params: tuple = (date,)
    if department:
        sql += " AND lower(a.department) = lower(?)"
        params += (department,)
    sql += " ORDER BY a.starts_at"
    with connect(db_path) as conn:
        return _rows(conn, sql, params)


def ward_occupancy(ward: str | None = None, db_path: Path | None = None) -> list[dict]:
    """Occupied and free bed counts per ward, computed from live admissions."""
    sql = """
        SELECT w.name, w.speciality, w.total_beds, w.closed_beds,
               COUNT(p.mrn) AS occupied
        FROM wards w
        LEFT JOIN patients p ON p.ward = w.name AND p.status = 'admitted'
        {filter}
        GROUP BY w.name, w.speciality, w.total_beds, w.closed_beds
        ORDER BY w.name
    """.format(filter="WHERE lower(w.name) = lower(?)" if ward else "")
    params: tuple = (ward,) if ward else ()
    with connect(db_path) as conn:
        rows = _rows(conn, sql, params)
    for row in rows:
        row["available"] = max(row["total_beds"] - row["closed_beds"] - row["occupied"], 0)
        usable = max(row["total_beds"] - row["closed_beds"], 1)
        row["occupancy_pct"] = round(100.0 * row["occupied"] / usable, 1)
    return rows


def lab_results(
    mrn: str, panel: str | None = None, db_path: Path | None = None, limit: int = 20
) -> list[dict]:
    sql = "SELECT * FROM lab_results WHERE mrn = ?"
    params: tuple = (mrn.strip().upper(),)
    if panel:
        sql += " AND lower(panel) = lower(?)"
        params += (panel,)
    sql += " ORDER BY collected_at DESC, analyte LIMIT ?"
    params += (limit,)
    with connect(db_path) as conn:
        rows = _rows(conn, sql, params)
    for row in rows:
        row["flag"] = (
            "low"
            if row["value"] < row["reference_low"]
            else "high"
            if row["value"] > row["reference_high"]
            else "normal"
        )
    return rows


def counts(db_path: Path | None = None) -> dict[str, int]:
    with connect(db_path) as conn:
        return {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("patients", "appointments", "wards", "lab_results")
        }
