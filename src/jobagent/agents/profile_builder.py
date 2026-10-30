"""Profile builder agent: turns an uploaded CV into the verified facts every application cites.

The facts it extracts are a proposal. The user reviews and edits them before they're saved,
because from then on they are the only experience the writer is allowed to use.
"""
from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field

from ..llm import LLMClient
from ..models import Candidate, Fact


class ProfileDraft(BaseModel):
    name: str
    email: str = Field(description="Empty string if not in the CV")
    location: str = Field(description="City and country as stated; empty if not stated")
    country: str = Field(description="Country of residence as stated; empty if not stated")
    work_authorization: str = Field(description="Only if the CV states it; otherwise empty")
    portfolio_url: str = Field(description="Portfolio, GitHub or LinkedIn URL if present; otherwise empty")
    headline: str = Field(description="One-sentence professional summary based only on the CV")
    facts: list[str] = Field(
        description="8-20 separate facts. Each is one specific statement the CV supports: a degree with "
                    "dates, a role with employer and dates, a responsibility, a project, a tool list, or a "
                    "result with the exact numbers stated."
    )


SYSTEM = """You turn a CV into a list of verifiable facts for a job application assistant.

Rules:
- Use only what the CV says. Never infer skills, tools, seniority or results it doesn't state.
- Keep numbers, dates, employer names and job titles exactly as written.
- One idea per fact, phrased as a complete sentence in the third person without the person's name.
- Leave a field empty rather than guessing."""


def extract_profile(llm: LLMClient, cv_text: str) -> Candidate:
    draft = llm.generate(agent="profile_builder", system=SYSTEM, prompt=f"CV:\n\n{cv_text}", schema=ProfileDraft)
    return build_candidate(
        name=draft.name, email=draft.email, location=draft.location, country=draft.country,
        work_authorization=draft.work_authorization, portfolio_url=draft.portfolio_url,
        headline=draft.headline, facts=draft.facts, no_experience_with=[],
    )


def build_candidate(*, name: str, email: str, location: str, country: str, work_authorization: str,
                    portfolio_url: str, headline: str, facts: list[str], no_experience_with: list[str]) -> Candidate:
    """A candidate with fact ids F1, F2, ... and an id that changes when the profile changes."""
    clean = [f.strip() for f in facts if f.strip()]
    digest = hashlib.sha256("|".join([name, email, *clean]).encode()).hexdigest()[:10]
    return Candidate(
        id=f"profile-{digest}", name=name.strip() or "Unnamed candidate", email=email.strip(),
        location=location.strip(), country=country.strip(),
        work_authorization=work_authorization.strip() or "Not stated",
        portfolio_url=portfolio_url.strip(), headline=headline.strip(),
        facts=[Fact(id=f"F{i}", text=text) for i, text in enumerate(clean, start=1)],
        no_experience_with=[s.strip() for s in no_experience_with if s.strip()],
    )
