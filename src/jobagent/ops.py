"""Operations: what a deployed copy needs that a laptop copy gets for free.

On a laptop there is one process, so an in-memory lock, an in-memory error list and llm.calls are
enough. On serverless hosting every request may land on a different, short-lived instance, so all
of that has to live in the database:

  RecordingLLM   writes every LLM call (agent, model, latency, tokens) to llm_calls and to the log
  OpsStore       run locks that work across instances, the last error per run, and alerts
  stats()        the numbers the /ops page shows
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import statistics
import threading
from datetime import datetime, timedelta, timezone
from typing import Iterator, TypeVar

from pydantic import BaseModel
from sqlalchemy import (Boolean, Column, DateTime, Engine, Float, Integer, MetaData, String, Table, Text, delete,
                        func, insert, select, text)

from .guardrails import redact
from .llm import CallRecord, LLMClient

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger("jobagent.ops")

metadata = MetaData()
llm_calls = Table(
    "llm_calls", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("at", DateTime(timezone=True), nullable=False),
    Column("agent", String, nullable=False),
    Column("model", String, nullable=False),
    Column("latency_s", Float, nullable=False),
    Column("input_tokens", Integer),
    Column("output_tokens", Integer),
    Column("cached", Boolean, nullable=False),
    Column("ok", Boolean, nullable=False),
    Column("error", Text),
)
run_errors = Table(
    "run_errors", metadata,
    Column("run_id", String, primary_key=True),
    Column("error", Text, nullable=False),
    Column("at", DateTime(timezone=True), nullable=False),
)
alerts = Table(
    "alerts", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("at", DateTime(timezone=True), nullable=False),
    Column("kind", String, nullable=False),  # guardrail | error | budget
    Column("run_id", String),
    Column("detail", Text, nullable=False),
)


def now() -> datetime:
    return datetime.now(timezone.utc)


class OpsStore:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.postgres = engine.dialect.name == "postgresql"
        self._local_locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        metadata.create_all(engine)

    # locks

    @contextlib.contextmanager
    def try_lock(self, run_id: str) -> Iterator[bool]:
        """Yield True if this caller may advance the run. On Postgres the lock holds across instances."""
        if self.postgres:
            key = int.from_bytes(hashlib.sha256(run_id.encode()).digest()[:8], "big", signed=True)
            with self.engine.connect() as conn:
                got = conn.execute(text("select pg_try_advisory_lock(:k)"), {"k": key}).scalar()
                try:
                    yield bool(got)
                finally:
                    if got:
                        conn.execute(text("select pg_advisory_unlock(:k)"), {"k": key})
                    conn.commit()
            return
        with self._guard:
            lock = self._local_locks.setdefault(run_id, threading.Lock())
        got = lock.acquire(blocking=False)
        try:
            yield got
        finally:
            if got:
                lock.release()

    # errors

    def set_error(self, run_id: str, error: str) -> None:
        error = redact(error)[:2000]
        with self.engine.begin() as conn:
            conn.execute(delete(run_errors).where(run_errors.c.run_id == run_id))
            conn.execute(insert(run_errors).values(run_id=run_id, error=error, at=now()))
        self.alert("error", error, run_id)

    def clear_error(self, run_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(delete(run_errors).where(run_errors.c.run_id == run_id))

    def get_error(self, run_id: str) -> str | None:
        with self.engine.connect() as conn:
            return conn.execute(select(run_errors.c.error).where(run_errors.c.run_id == run_id)).scalar()

    # alerts and calls

    def alert(self, kind: str, detail: str, run_id: str | None = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(alerts).values(at=now(), kind=kind, run_id=run_id, detail=redact(detail)[:2000]))
        log.warning(json.dumps({"event": "alert", "kind": kind, "run_id": run_id, "detail": redact(detail)[:300]}))

    def record_call(self, record: CallRecord, ok: bool = True, error: str | None = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(llm_calls).values(
                at=now(), agent=record.agent, model=record.model, latency_s=round(record.latency_s, 3),
                input_tokens=record.input_tokens, output_tokens=record.output_tokens, cached=record.cached,
                ok=ok, error=redact(error)[:500] if error else None))
        log.info(json.dumps({"event": "llm_call", "agent": record.agent, "model": record.model, "ok": ok,
                             "latency_s": round(record.latency_s, 2), "input_tokens": record.input_tokens,
                             "output_tokens": record.output_tokens, "cached": record.cached}))

    def stats(self, hours: int = 24) -> dict:
        since = now() - timedelta(hours=hours)
        with self.engine.connect() as conn:
            calls = [dict(r._mapping) for r in conn.execute(select(llm_calls).where(llm_calls.c.at >= since))]
            recent = [dict(r._mapping) for r in conn.execute(
                select(alerts).order_by(alerts.c.id.desc()).limit(10))]
            alert_counts = dict(conn.execute(select(alerts.c.kind, func.count()).where(alerts.c.at >= since)
                                             .group_by(alerts.c.kind)).all())
        by_agent: dict[str, dict] = {}
        for c in calls:
            a = by_agent.setdefault(c["agent"], {"calls": 0, "failed": 0, "cached": 0, "tokens": 0, "latencies": []})
            a["calls"] += 1
            a["failed"] += not c["ok"]
            a["cached"] += c["cached"]
            a["tokens"] += (c["input_tokens"] or 0) + (c["output_tokens"] or 0)
            if c["ok"] and not c["cached"]:
                a["latencies"].append(c["latency_s"])
        for a in by_agent.values():
            lat = sorted(a.pop("latencies"))
            a["p50_s"] = round(statistics.median(lat), 1) if lat else None
            a["p95_s"] = round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1) if lat else None
        live = [c for c in calls if not c["cached"]]
        return {
            "hours": hours,
            "calls": len(calls),
            "failed": sum(not c["ok"] for c in calls),
            "cached": sum(c["cached"] for c in calls),
            "tokens": sum((c["input_tokens"] or 0) + (c["output_tokens"] or 0) for c in calls),
            "error_rate": round(sum(not c["ok"] for c in live) / len(live), 3) if live else 0.0,
            "by_agent": dict(sorted(by_agent.items())),
            "alerts": alert_counts,
            "recent_alerts": recent,
        }


class RecordingLLM:
    """Wraps any LLM client and stores each call, including failures, in the ops tables."""

    def __init__(self, inner: LLMClient, ops: OpsStore):
        self.inner, self.ops = inner, ops
        self.model = getattr(inner, "model", "unknown")

    @property
    def calls(self) -> list[CallRecord]:
        return self.inner.calls

    def generate(self, *, agent: str, system: str, prompt: str, schema: type[T]) -> T:
        before = len(self.inner.calls)
        try:
            result = self.inner.generate(agent=agent, system=system, prompt=prompt, schema=schema)
        except Exception as exc:
            self.ops.record_call(CallRecord(agent, self.model, 0.0, None, None), ok=False,
                                 error=f"{type(exc).__name__}: {exc}")
            raise
        for record in self.inner.calls[before:]:
            self.ops.record_call(record)
        return result
