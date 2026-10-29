"""The mock job board and company pages the research agents use as their tools."""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from .models import Candidate, Company, Job

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class World:
    def __init__(self, data_dir: Path = DATA_DIR):
        read = lambda name: json.loads((data_dir / name).read_text(encoding="utf-8"))
        self.candidate = Candidate.model_validate(read("candidate.json"))
        self.jobs = {j["id"]: Job.model_validate(j) for j in read("jobs.json")}
        self.companies = {c["id"]: Company.model_validate(c) for c in read("companies.json")}

    def search_jobs(self, keywords: list[str]) -> list[Job]:
        """Job board search: postings whose title contains any keyword (case-insensitive)."""
        words = [k.lower().strip() for k in keywords if k.strip()]
        return [j for j in self.jobs.values() if any(w in j.title.lower() for w in words)]

    def job(self, job_id: str) -> Job:
        return self.jobs[job_id]

    def company(self, company_id: str) -> Company:
        return self.companies[company_id]


@cache
def default_world() -> World:
    return World()
