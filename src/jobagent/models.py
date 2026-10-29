"""Domain models for the mock world: candidate, companies and job postings."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Fact(BaseModel):
    """One verified piece of the candidate's experience. Every claim in an application cites these."""
    id: str
    text: str


class Candidate(BaseModel):
    id: str
    name: str
    email: str
    location: str
    country: str
    work_authorization: str
    portfolio_url: str
    headline: str
    facts: list[Fact]
    no_experience_with: list[str] = Field(default_factory=list)

    def fact_map(self) -> dict[str, str]:
        return {f.id: f.text for f in self.facts}

    def facts_text(self) -> str:
        return "\n".join(f"[{f.id}] {f.text}" for f in self.facts)


class Company(BaseModel):
    id: str
    name: str
    website: str
    pages: dict[str, str]

    def sources(self) -> dict[str, str]:
        """Source id ('nimbus-ledger/about') to page text: what company evidence must quote."""
        return {f"{self.id}/{page}": text for page, text in self.pages.items()}


class ApplyInfo(BaseModel):
    method: Literal["portal", "email"]
    email: str | None = None
    questions: list[str] = Field(default_factory=list)


class Job(BaseModel):
    id: str
    title: str
    company_id: str
    location: str
    salary: str
    posted: str
    description: str
    apply: ApplyInfo
    portal_behavior: str = "ok"  # how the mock portal reacts; see portal.py

    @property
    def url(self) -> str:
        return f"/board/jobs/{self.id}"
