"""Local SQLite queue for unverified user-submitted service cases.

Data in this database NEVER enters the trusted diagnostic knowledge base automatically.
No emails are sent from this module; outbox is an explicit future integration point.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "review_queue.sqlite3"
STATUSES = ("pending", "in_review", "needs_info", "approved_for_kb", "closed")
NOTIFY_ON = {"needs_info", "approved_for_kb", "closed"}
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean(value: str | None, limit: int, *, required: bool = False) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ValueError("All submitted text fields must be strings")
    value = " ".join(value.split()).strip()
    if required and not value:
        raise ValueError("Required field is empty")
    if len(value) > limit:
        raise ValueError(f"Field exceeds {limit} characters")
    return value


@contextmanager
def connect(db_path: Path = DEFAULT_DB):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript("""
    CREATE TABLE IF NOT EXISTS submissions (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        brand TEXT NOT NULL,
        model TEXT NOT NULL,
        symptoms TEXT NOT NULL,
        reported_cause TEXT NOT NULL,
        cause_certainty TEXT NOT NULL CHECK (cause_certainty IN ('unknown','suspected','confirmed_by_reporter')),
        email TEXT,
        email_consent INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL CHECK (status IN ('pending','in_review','needs_info','approved_for_kb','closed')),
        reviewer_note TEXT NOT NULL DEFAULT '',
        dedup_hash TEXT UNIQUE NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_submissions_status ON submissions(status, created_at);
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        happened_at TEXT NOT NULL,
        kind TEXT NOT NULL,
        submission_id TEXT,
        metadata_json TEXT NOT NULL,
        FOREIGN KEY (submission_id) REFERENCES submissions(id)
    );
    CREATE TABLE IF NOT EXISTS outbox (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        submission_id TEXT NOT NULL,
        recipient TEXT NOT NULL,
        new_status TEXT NOT NULL,
        subject TEXT NOT NULL,
        body TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT 'queued',
        FOREIGN KEY (submission_id) REFERENCES submissions(id)
    );
    """)
    try:
        with con:
            yield con
    finally:
        con.close()


def _event(con: sqlite3.Connection, kind: str, *, case_id: str | None = None,
           metadata: dict | None = None) -> None:
    con.execute("INSERT INTO events(happened_at,kind,submission_id,metadata_json) VALUES(?,?,?,?)",
                (utc_now(), kind, case_id, json.dumps(metadata or {}, ensure_ascii=False)))


def log_search(*, found_count: int, brand: str = "", model: str = "",
               db_path: Path = DEFAULT_DB) -> None:
    """Analytics event: intentionally do not save the raw free-text diagnosis query."""
    if found_count < 0:
        raise ValueError("found_count cannot be negative")
    with connect(db_path) as con:
        _event(con, "search", metadata={"found_count": found_count,
                                       "brand_provided": bool(brand.strip()),
                                       "model_provided": bool(model.strip())})


def submit_case(*, brand: str, model: str, symptoms: str,
                reported_cause: str = "", cause_certainty: str = "unknown",
                email: str = "", consent_to_email: bool = False,
                db_path: Path = DEFAULT_DB) -> tuple[str, bool]:
    """Returns (case_id, newly_created). Only an explicit submission calls this."""
    brand = _clean(brand, 80, required=True)
    model = _clean(model, 100, required=True)
    symptoms = _clean(symptoms, 1200, required=True)
    reported_cause = _clean(reported_cause, 700)
    email = _clean(email, 254)
    if cause_certainty not in ("unknown", "suspected", "confirmed_by_reporter"):
        raise ValueError("Invalid cause_certainty")
    if email and (not consent_to_email or not EMAIL_RE.fullmatch(email)):
        raise ValueError("A valid email and explicit email consent are required")
    # Deterministic fingerprint prevents identical resubmissions.
    fingerprint = "\x1f".join(v.casefold() for v in
                             (brand, model, symptoms, reported_cause, cause_certainty))
    dedup_hash = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    with connect(db_path) as con:
        existing = con.execute("SELECT id FROM submissions WHERE dedup_hash=?", (dedup_hash,)).fetchone()
        if existing:
            _event(con, "duplicate_submission", case_id=existing["id"])
            return existing["id"], False
        case_id = uuid4().hex[:12]
        now = utc_now()
        con.execute("""
            INSERT INTO submissions(id,created_at,updated_at,brand,model,symptoms,
                reported_cause,cause_certainty,email,email_consent,status,dedup_hash)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        """, (case_id, now, now, brand, model, symptoms, reported_cause,
              cause_certainty, email or None, int(bool(email and consent_to_email)), "pending", dedup_hash))
        _event(con, "case_submitted", case_id=case_id)
    return case_id, True


def get_public_status(case_id: str, db_path: Path = DEFAULT_DB) -> dict | None:
    """Only fields safe for an applicant with the ticket ID. Does not expose email."""
    with connect(db_path) as con:
        row = con.execute("SELECT id,created_at,updated_at,status,reviewer_note FROM submissions WHERE id=?",
                          (case_id,)).fetchone()
        return dict(row) if row else None


def list_cases(status: str | None = None, db_path: Path = DEFAULT_DB) -> list[dict]:
    if status is not None and status not in STATUSES:
        raise ValueError("Invalid status")
    with connect(db_path) as con:
        sql = "SELECT id,created_at,brand,model,status,cause_certainty FROM submissions"
        args = ()
        if status:
            sql += " WHERE status=?"
            args = (status,)
        sql += " ORDER BY created_at DESC, id DESC"
        return [dict(row) for row in con.execute(sql, args)]


def get_admin_case(case_id: str, db_path: Path = DEFAULT_DB) -> dict | None:
    """Call only from local admin tooling, never an unauthenticated web endpoint."""
    with connect(db_path) as con:
        row = con.execute("SELECT * FROM submissions WHERE id=?", (case_id,)).fetchone()
        return dict(row) if row else None


def change_status(*, case_id: str, new_status: str, reviewer_note: str = "",
                  db_path: Path = DEFAULT_DB) -> None:
    if new_status not in STATUSES:
        raise ValueError(f"Unsupported status {new_status}")
    reviewer_note = _clean(reviewer_note, 800)
    with connect(db_path) as con:
        case = con.execute("SELECT * FROM submissions WHERE id=?", (case_id,)).fetchone()
        if not case:
            raise KeyError(f"Unknown ticket: {case_id}")
        if case["status"] == new_status and case["reviewer_note"] == reviewer_note:
            return
        now = utc_now()
        con.execute("UPDATE submissions SET status=?,reviewer_note=?,updated_at=? WHERE id=?",
                    (new_status, reviewer_note, now, case_id))
        _event(con, "status_updated", case_id=case_id,
               metadata={"old": case["status"], "new": new_status})
        if case["email"] and case["email_consent"] and new_status in NOTIFY_ON and case["status"] != new_status:
            subject = f"Coffee Diagnostic Agent: aktualizacja zgłoszenia {case_id}"
            body = (f"Status Twojego zgłoszenia {case_id}: {new_status}.\n\n"
                    f"Informacja od osoby weryfikującej: {reviewer_note or 'Brak dodatkowych uwag.'}\n\n"
                    "Ta wiadomość nie jest potwierdzeniem diagnozy ani instrukcją naprawy.")
            con.execute("INSERT INTO outbox(created_at,submission_id,recipient,new_status,subject,body)"
                        " VALUES(?,?,?,?,?,?)", (now, case_id, case["email"], new_status, subject, body))
            _event(con, "email_queued_not_sent", case_id=case_id,
                   metadata={"new_status": new_status})


def list_outbox(db_path: Path = DEFAULT_DB) -> list[dict]:
    """Local admin-only preview; queued messages are NOT sent automatically."""
    with connect(db_path) as con:
        return [dict(r) for r in con.execute(
            "SELECT id,created_at,submission_id,new_status,state,subject FROM outbox ORDER BY id DESC")]


def list_events(limit: int = 30, db_path: Path = DEFAULT_DB) -> list[dict]:
    if not 1 <= limit <= 200:
        raise ValueError("limit must be 1..200")
    with connect(db_path) as con:
        return [dict(r) for r in con.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))]
