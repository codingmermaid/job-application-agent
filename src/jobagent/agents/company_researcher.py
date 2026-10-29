"""Company researcher agent: writes an evidence-backed brief on the employer and role."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..grounding import verify_brief
from ..llm import LLMClient
from ..models import Company, Job


class Evidence(BaseModel):
    id: str = Field(description="E1, E2, ...")
    source: str = Field(description="The source id the quote comes from, e.g. 'nimbus-ledger/news'")
    quote: str = Field(description="An exact, word-for-word quote from that source")


class TalkingPoint(BaseModel):
    point: str = Field(description="Something the candidate could mention, in one sentence")
    evidence_ids: list[str]


class CompanyBrief(BaseModel):
    overview: str
    role_context: str = Field(description="What this role likely contributes to the company")
    talking_points: list[TalkingPoint]
    evidence: list[Evidence]


SYSTEM = """You are a company researcher preparing a candidate for an application. Write a short brief
using only the sources provided.

Rules:
- Every talking point must cite evidence ids.
- Each evidence quote must be copied word for word from its source. Do not paraphrase quotes.
- If the sources don't say something, leave it out. Never add outside knowledge."""


def research_company(llm: LLMClient, company: Company, job: Job) -> tuple[dict, list[str]]:
    """Returns (brief as a dict, issues found by the evidence check)."""
    sources = "\n\n".join(f"[{sid}]\n{text}" for sid, text in company.sources().items())
    prompt = (
        f"Company: {company.name} ({company.website})\nRole: {job.title}\n\n"
        f"Job posting:\n{job.description}\n\nSources:\n\n{sources}"
    )
    brief = llm.generate(agent="company_researcher", system=SYSTEM, prompt=prompt, schema=CompanyBrief)
    return verify_brief(brief.model_dump(), company)
