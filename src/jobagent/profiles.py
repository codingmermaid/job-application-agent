"""The active candidate profile: your own (once saved) or the fictional sample."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Engine, MetaData, String, Table, Text, delete, insert, select

from .models import Candidate

metadata = MetaData()
profiles = Table(
    "profile", metadata,
    Column("key", String, primary_key=True),
    Column("data", Text, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


coaching = Table(
    "job_coaching", metadata,
    Column("profile_id", String, primary_key=True),
    Column("job_id", String, primary_key=True),
    Column("data", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)


class CoachingCache:
    """Fit coach results per (profile, job). A changed profile has a new id, so advice is never stale."""

    def __init__(self, engine: Engine):
        self.engine = engine
        metadata.create_all(engine)

    def get(self, profile_id: str, job_id: str) -> dict | None:
        query = select(coaching.c.data).where(coaching.c.profile_id == profile_id, coaching.c.job_id == job_id)
        with self.engine.connect() as conn:
            row = conn.execute(query).first()
        return json.loads(row.data) if row else None

    def put(self, profile_id: str, job_id: str, fit: dict) -> None:
        with self.engine.begin() as conn:
            conn.execute(delete(coaching).where(coaching.c.profile_id == profile_id, coaching.c.job_id == job_id))
            conn.execute(insert(coaching).values(profile_id=profile_id, job_id=job_id, data=json.dumps(fit),
                                                 created_at=datetime.now(timezone.utc)))


class ProfileStore:
    def __init__(self, engine: Engine, sample: Candidate):
        self.engine = engine
        self.sample = sample
        metadata.create_all(engine)

    def get(self) -> Candidate:
        with self.engine.connect() as conn:
            row = conn.execute(select(profiles.c.data).where(profiles.c.key == "active")).first()
        return Candidate.model_validate_json(row.data) if row else self.sample

    def is_sample(self) -> bool:
        return self.get().id == self.sample.id

    def save(self, candidate: Candidate) -> None:
        with self.engine.begin() as conn:
            conn.execute(delete(profiles).where(profiles.c.key == "active"))
            conn.execute(insert(profiles).values(key="active", data=candidate.model_dump_json(),
                                                 updated_at=datetime.now(timezone.utc)))

    def reset(self) -> None:
        """Go back to the sample candidate and forget the saved profile."""
        with self.engine.begin() as conn:
            conn.execute(delete(profiles).where(profiles.c.key == "active"))
