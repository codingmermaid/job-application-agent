"""The active candidate profile: your own (once saved) or the fictional sample."""
from __future__ import annotations

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
