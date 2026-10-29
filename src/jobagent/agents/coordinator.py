"""Coordinator agent: turns the candidate's goal into a search plan.

The rest of the coordinator's job (delegating, tracking state, next steps) is the graph itself.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import LLMClient
from ..models import Candidate


class SearchPlan(BaseModel):
    keywords: list[str] = Field(
        description="Job title keywords for the job board, broad enough to catch variants, "
                    "e.g. ['data analyst', 'bi analyst', 'insights analyst']"
    )
    seniority: str = Field(description="junior, mid or senior")
    remote_only: bool
    applicant_country: str = Field(description="Country the candidate will work from")
    max_applications: int = Field(description="How many applications to prepare. Default 3 if not stated.")
    other_constraints: list[str] = Field(description="Any other stated constraints, e.g. salary or industry")


SYSTEM = """You are the coordinator of a job application team. Turn the candidate's request into a
search plan for a job board that matches on job titles. Use the candidate profile for anything the
request leaves out (for example, the country they live in). Do not invent constraints."""


def plan_search(llm: LLMClient, request: str, candidate: Candidate) -> SearchPlan:
    prompt = (
        f"Candidate: {candidate.name}, lives in {candidate.location}. {candidate.headline}\n"
        f"Work authorization: {candidate.work_authorization}\n\nRequest:\n{request}"
    )
    plan = llm.generate(agent="coordinator", system=SYSTEM, prompt=prompt, schema=SearchPlan)
    plan.max_applications = min(max(plan.max_applications, 1), 5)
    return plan
