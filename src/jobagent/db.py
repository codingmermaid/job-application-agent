"""Database setup. SQLite locally; set DATABASE_URL (e.g. Postgres) to use a hosted database."""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine

ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    path = Path(os.getenv("JOBAGENT_DATA_DIR") or ROOT / "var")
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_engine(url: str | None = None) -> Engine:
    url = url or os.getenv("DATABASE_URL") or f"sqlite:///{data_dir() / 'app.db'}"
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
    return create_engine(url, **kwargs)
