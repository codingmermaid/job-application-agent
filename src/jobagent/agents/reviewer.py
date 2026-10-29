"""Reviewer agent: checks accuracy, relevance, missing requirements and unsupported claims.

Two layers: deterministic grounding checks in code, then an LLM review of whether each claim
actually matches the facts it cites. The draft is ready only if both pass.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..grounding import check_draft
from ..llm import LLMClient
from ..models import Candidate, Job


class ClaimIssue(BaseModel):
    claim: str
    problem: str


class ReviewVerdict(BaseModel):
    unsupported_claims: list[ClaimIssue] = Field(
        description="Claims that overstate or go beyond the facts they cite"
    )
    missing_requirements: list[str] = Field(
        description="Posting requirements the candidate's facts cover but the draft fails to mention"
    )
    corrections: list[str] = Field(description="Specific, actionable fixes for the writer")
    ready: bool = Field(description="true only if there are no unsupported claims and the draft is relevant")
    summary: str


SYSTEM = """You are a strict application reviewer. Compare every claim in the draft with the candidate
facts and company evidence it cites.

Flag a claim as unsupported if it adds anything the cited facts don't state: tools, numbers, scope,
seniority, results or employers. Flag requirements the candidate's facts cover but the draft ignores.
Do not flag honest statements about wanting to learn something. Be specific in corrections."""


def review_application(
    llm: LLMClient, candidate: Candidate, job: Job, company_name: str, brief: dict, draft: dict
) -> dict:
    code_issues = check_draft(draft, candidate, brief, job, company_name)
    evidence = "\n".join(f"[{e['id']}] {e['quote']}" for e in brief["evidence"])
    prompt = (
        f"Candidate facts:\n{candidate.facts_text()}\n\nCompany evidence:\n{evidence}\n\n"
        f"Job: {job.title} at {company_name}\n{job.description}\n\nDraft (JSON):\n{draft}"
    )
    verdict = llm.generate(agent="reviewer", system=SYSTEM, prompt=prompt, schema=ReviewVerdict)
    corrections = list(verdict.corrections) + code_issues
    return {
        **verdict.model_dump(),
        "code_issues": code_issues,
        "corrections": corrections,
        "ready": verdict.ready and not code_issues,
    }
