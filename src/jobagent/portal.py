"""Mock application portal: a stand-in for an employer's careers site API.

It fails on purpose so the submission agent has to cope:
  ok                    accepts the application
  timeout_after_commit  saves the application, then times out before replying (the classic
                        duplicate trap: a naive retry would apply twice)
  unavailable_once      returns 503 on the first attempt, then works
  reject                refuses the application (unsupported requirement)
  manual                no portal at all; the employer wants email

Submissions are idempotent: the same idempotency key never creates a second application.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Engine, Integer, MetaData, String, Table, Text, insert, select, update

metadata = MetaData()
submissions = Table(
    "portal_submissions", metadata,
    Column("confirmation_id", String, primary_key=True),
    Column("idempotency_key", String, unique=True, nullable=False),
    Column("job_id", String, nullable=False),
    Column("payload", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
attempts = Table(
    "portal_attempts", metadata,
    Column("idempotency_key", String, primary_key=True),
    Column("count", Integer, nullable=False),
)

REQUIRED_FIELDS = ("full_name", "email", "location", "resume", "cover_letter")


class PortalError(Exception):
    pass


class PortalTimeout(PortalError):
    """No reply. The application may or may not have been saved."""


class PortalUnavailable(PortalError):
    """Temporary outage. Nothing was saved."""


class PortalRejected(PortalError):
    """The portal refused the application. Retrying will not help."""


class PortalUnsupported(PortalError):
    """This employer has no portal; apply another way."""


class Confirmation(BaseModel):
    confirmation_id: str
    job_id: str
    idempotency_key: str
    created_at: datetime


class MockPortal:
    def __init__(self, engine: Engine, behaviors: dict[str, str]):
        self.engine = engine
        self.behaviors = behaviors  # job id -> behaviour name
        metadata.create_all(engine)

    def submit(self, job_id: str, fields: dict[str, str], idempotency_key: str) -> Confirmation:
        behavior = self.behaviors.get(job_id, "ok")
        if behavior == "manual":
            raise PortalUnsupported("This employer does not accept applications through a portal.")
        existing = self.lookup(idempotency_key)
        if existing:
            return existing
        attempt = self._count_attempt(idempotency_key)

        if behavior == "reject":
            raise PortalRejected("The employer requires a PDF résumé upload, which this integration does not support.")
        missing = [f for f in REQUIRED_FIELDS if not fields.get(f, "").strip()]
        missing += [k for k, v in fields.items() if k.startswith("answer_") and not v.strip()]
        if missing:
            raise PortalRejected(f"Missing required fields: {', '.join(missing)}")
        if behavior == "unavailable_once" and attempt == 1:
            raise PortalUnavailable("503 Service Unavailable")

        confirmation = Confirmation(
            confirmation_id=f"APP-{uuid.uuid4().hex[:8].upper()}",
            job_id=job_id,
            idempotency_key=idempotency_key,
            created_at=datetime.now(timezone.utc),
        )
        with self.engine.begin() as conn:
            conn.execute(insert(submissions).values(**confirmation.model_dump(), payload=json.dumps(fields)))
        if behavior == "timeout_after_commit" and attempt == 1:
            raise PortalTimeout("Read timed out")
        return confirmation

    def lookup(self, idempotency_key: str) -> Confirmation | None:
        """Check whether an application with this key exists (the portal's status endpoint)."""
        with self.engine.connect() as conn:
            row = conn.execute(select(submissions).where(submissions.c.idempotency_key == idempotency_key)).first()
        if row is None:
            return None
        return Confirmation(confirmation_id=row.confirmation_id, job_id=row.job_id,
                            idempotency_key=row.idempotency_key, created_at=row.created_at)

    def list_submissions(self) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(submissions).order_by(submissions.c.created_at)).all()
        return [dict(r._mapping) for r in rows]

    def _count_attempt(self, key: str) -> int:
        with self.engine.begin() as conn:
            row = conn.execute(select(attempts.c.count).where(attempts.c.idempotency_key == key)).first()
            if row is None:
                conn.execute(insert(attempts).values(idempotency_key=key, count=1))
                return 1
            conn.execute(update(attempts).where(attempts.c.idempotency_key == key).values(count=row.count + 1))
            return row.count + 1
