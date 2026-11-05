"""Database setup. SQLite locally; set DATABASE_URL (e.g. Postgres) to use a hosted database."""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import NullPool

ROOT = Path(__file__).resolve().parents[2]


def on_serverless() -> bool:
    """True on Vercel (and anything that sets VERCEL=1): read-only disk, short-lived instances."""
    return os.getenv("VERCEL") == "1"


def data_dir() -> Path:
    default = Path("/tmp/jobagent") if on_serverless() else ROOT / "var"  # only /tmp is writable on Vercel
    path = Path(os.getenv("JOBAGENT_DATA_DIR") or default)
    path.mkdir(parents=True, exist_ok=True)
    return path


def database_url() -> str:
    return os.getenv("DATABASE_URL") or f"sqlite:///{data_dir() / 'app.db'}"


def is_postgres(url: str) -> bool:
    return url.startswith(("postgres://", "postgresql://", "postgresql+psycopg://"))


def sqlalchemy_url(url: str) -> str:
    """Hosted Postgres hands out postgres://... URLs; SQLAlchemy needs to be told to use psycopg 3."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def libpq_url(url: str) -> str:
    """The same URL in the form psycopg itself (and the LangGraph checkpointer) accepts."""
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


def make_engine(url: str | None = None) -> Engine:
    url = url or database_url()
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    # A serverless instance may be frozen between requests; holding pooled connections then only
    # uses up the database's connection limit. Open one per request instead.
    kwargs = {"poolclass": NullPool} if on_serverless() else {"pool_pre_ping": True, "pool_size": 5}
    return create_engine(sqlalchemy_url(url), **kwargs)
