"""Application writer agent: tailors résumé wording and writes the cover letter from verified facts."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import LLMClient
from ..models import Candidate, Job


class Claim(BaseModel):
    text: str
    fact_ids: list[str] = Field(description="Candidate fact ids this sentence relies on")
    evidence_ids: list[str] = Field(description="Company evidence ids this sentence relies on")


class ScreeningAnswer(BaseModel):
    question: str
    answer: Claim


class ApplicationDraft(BaseModel):
    resume_summary: Claim = Field(description="Two-sentence profile summary tailored to the role")
    resume_bullets: list[Claim] = Field(description="4-6 tailored experience bullets")
    cover_letter: list[Claim] = Field(description="3-4 short paragraphs")
    screening_answers: list[ScreeningAnswer] = Field(description="One answer per application question, in order")


SYSTEM = """You are an application writer. Tailor the candidate's résumé wording and write a cover letter
for one job.

Hard rules:
- Use only the candidate's facts. You may reword and emphasise, but never add experience, tools,
  numbers, employers or results that the facts do not state.
- Every sentence about the candidate cites the fact ids it uses. Sentences about the company cite
  company evidence ids. A sentence that cites nothing is not allowed.
- Never claim skills from the "no experience with" list. If the job wants one, either leave it out or
  say honestly that the candidate is keen to learn it, citing no fact for that part.
- Plain, confident English. No clichés like "I am writing to express my interest"."""


def write_application(
    llm: LLMClient,
    candidate: Candidate,
    job: Job,
    company_name: str,
    assessment: dict,
    brief: dict,
    corrections: list[str] | None = None,
    previous: dict | None = None,
) -> ApplicationDraft:
    evidence = "\n".join(f"[{e['id']}] {e['quote']}" for e in brief["evidence"])
    points = "\n".join(f"- {p['point']} ({', '.join(p['evidence_ids'])})" for p in brief["talking_points"])
    questions = "\n".join(f"{i}. {q}" for i, q in enumerate(job.apply.questions, start=1)) or "(none)"
    prompt = (
        f"Candidate: {candidate.name}, {candidate.location}\nFacts:\n{candidate.facts_text()}\n"
        f"No experience with: {', '.join(candidate.no_experience_with)}\n\n"
        f"Job: {job.title} at {company_name}\n{job.description}\n\n"
        f"Requirements found by the researcher: {'; '.join(assessment['requirements'])}\n"
        f"Gaps found by the researcher: {'; '.join(assessment['gaps']) or 'none'}\n\n"
        f"Company brief: {brief['overview']} {brief['role_context']}\nTalking points:\n{points}\n"
        f"Company evidence:\n{evidence}\n\nApplication questions:\n{questions}"
    )
    if corrections and previous:
        prompt += (
            "\n\nA reviewer found problems with your previous draft. Fix every one of them.\n"
            f"Previous draft (JSON): {previous}\nCorrections:\n" + "\n".join(f"- {c}" for c in corrections)
        )
    return llm.generate(agent="writer", system=SYSTEM, prompt=prompt, schema=ApplicationDraft)
