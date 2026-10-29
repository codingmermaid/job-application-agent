"""Application tracker: the coordinator's record of every application and its history."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Engine, Integer, MetaData, String, Table, Text, insert, select, update

metadata = MetaData()
applications = Table(
    "applications", metadata,
    Column("id", String, primary_key=True),  # "<run_id>:<job_id>"
    Column("run_id", String, nullable=False),
    Column("job_id", String, nullable=False),
    Column("title", String, nullable=False),
    Column("company", String, nullable=False),
    Column("status", String, nullable=False),
    Column("detail", Text, nullable=False, default=""),
    Column("confirmation_id", String),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
runs = Table(
    "runs", metadata,
    Column("run_id", String, primary_key=True),
    Column("request", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
events = Table(
    "application_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("application_id", String, nullable=False),
    Column("status", String, nullable=False),
    Column("detail", Text, nullable=False),
    Column("at", DateTime(timezone=True), nullable=False),
)

# shortlisted -> selected -> drafted -> reviewed -> approved -> submitted | handoff | failed
# Anything not chosen or not approved ends as "skipped".


class Tracker:
    def __init__(self, engine: Engine):
        self.engine = engine
        metadata.create_all(engine)

    def add_run(self, run_id: str, request: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(runs).values(run_id=run_id, request=request, created_at=datetime.now(timezone.utc)))

    def list_runs(self, limit: int = 20) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(runs).order_by(runs.c.created_at.desc()).limit(limit))
            return [dict(r._mapping) for r in rows]

    def record(self, run_id: str, job_id: str, *, title: str, company: str, status: str,
               detail: str = "", confirmation_id: str | None = None) -> None:
        app_id = f"{run_id}:{job_id}"
        now = datetime.now(timezone.utc)
        values = {"status": status, "detail": detail, "updated_at": now}
        if confirmation_id:
            values["confirmation_id"] = confirmation_id
        with self.engine.begin() as conn:
            exists = conn.execute(select(applications.c.id).where(applications.c.id == app_id)).first()
            if exists:
                conn.execute(update(applications).where(applications.c.id == app_id).values(**values))
            else:
                conn.execute(insert(applications).values(id=app_id, run_id=run_id, job_id=job_id,
                                                         title=title, company=company, **values))
            conn.execute(insert(events).values(application_id=app_id, status=status, detail=detail, at=now))

    def list(self, run_id: str | None = None) -> list[dict]:
        query = select(applications).order_by(applications.c.updated_at.desc())
        if run_id:
            query = query.where(applications.c.run_id == run_id)
        with self.engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(query)]

    def history(self, application_id: str) -> list[dict]:
        query = select(events).where(events.c.application_id == application_id).order_by(events.c.id)
        with self.engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(query)]
