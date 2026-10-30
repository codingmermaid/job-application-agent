"""Job researcher agent: searches the job board, extracts requirements and builds a shortlist."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..llm import LLMClient
from ..models import Candidate
from ..world import World
from .coordinator import SearchPlan

SHORTLIST_SIZE = 6


class JobAssessment(BaseModel):
    job_id: str
    requirements: list[str] = Field(description="Requirements stated in the posting, one per item")
    location_constraint: str = Field(description="Where the employer allows the person to work from, as stated")
    eligible: Literal["yes", "no", "unclear"] = Field(
        description="Can the candidate work this job from their country under the posting's own terms?"
    )
    eligibility_reason: str
    seniority_fit: bool = Field(description="Does the role's level match what the candidate asked for?")
    application_instructions: str = Field(description="How to apply, as stated in the posting")
    match_score: int = Field(description="0-100: how well the candidate's facts cover the requirements")
    matched_fact_ids: list[str] = Field(description="Candidate fact ids that support the match")
    gaps: list[str] = Field(description="Requirements the candidate's facts do not cover")
    summary: str = Field(description="One sentence on why this is or isn't a good match")


class JobResearch(BaseModel):
    assessments: list[JobAssessment]


SYSTEM = """You are a job researcher. For each posting, extract the requirements, the location rule and
the application instructions exactly as the posting states them. Then judge eligibility and fit for
this candidate.

Rules:
- Eligibility comes only from the posting's text. "EU residents only" excludes a non-EU country.
  Hybrid or on-site roles in another country are not eligible for a remote candidate.
- Score the match only from the candidate's listed facts. Cite the fact ids you rely on.
- List real gaps honestly. Never assume skills the facts don't show."""


def research_jobs(
    llm: LLMClient, world: World, plan: SearchPlan, candidate: Candidate | None = None
) -> tuple[list[dict], list[dict]]:
    """Returns (shortlist, excluded). Each item is a JSON-friendly dict for the graph state."""
    found = world.search_jobs(plan.keywords)
    if not found:
        return [], []
    candidate = candidate or world.candidate
    postings = "\n\n".join(
        f"### Job {j.id}: {j.title} at {world.company(j.company_id).name}\nListed location: {j.location}\n"
        f"Salary: {j.salary}\n{j.description}"
        for j in found
    )
    prompt = (
        f"Candidate wants: {plan.seniority} roles, remote only: {plan.remote_only}, "
        f"working from {plan.applicant_country}. Other constraints: {plan.other_constraints or 'none'}\n"
        f"Work authorization: {candidate.work_authorization}\n\n"
        f"Candidate facts:\n{candidate.facts_text()}\n\nPostings:\n\n{postings}"
    )
    research = llm.generate(agent="job_researcher", system=SYSTEM, prompt=prompt, schema=JobResearch)

    known_facts = candidate.fact_map()
    by_id = {j.id: j for j in found}
    shortlist, excluded = [], []
    for a in research.assessments:
        job = by_id.get(a.job_id)
        if job is None:
            continue  # the model referred to a job that wasn't in the results
        a.matched_fact_ids = [f for f in a.matched_fact_ids if f in known_facts]
        a.match_score = min(max(a.match_score, 0), 100)
        item = {
            **a.model_dump(),
            "title": job.title,
            "company": world.company(job.company_id).name,
            "company_id": job.company_id,
            "location": job.location,
            "salary": job.salary,
            "url": job.url,
        }
        if a.eligible == "no" or not a.seniority_fit:
            excluded.append(item)
        else:
            shortlist.append(item)
    shortlist.sort(key=lambda i: i["match_score"], reverse=True)
    return shortlist[:SHORTLIST_SIZE], excluded
