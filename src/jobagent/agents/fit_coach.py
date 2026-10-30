"""Fit coach agent: for one job, how the candidate stands out, CV adjustments and a sample cover letter.

Same honesty rules as the application writer: everything about the candidate cites their facts,
and deterministic checks remove anything that doesn't hold up before it's shown.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..llm import LLMClient
from ..models import Candidate, Company, Job


class Strength(BaseModel):
    point: str = Field(description="Why this makes the candidate stand out for this job, in one sentence")
    fact_ids: list[str]


class CvEdit(BaseModel):
    suggestion: str = Field(description="What to change in the CV and why, e.g. 'Lead with the Power BI dashboard'")
    rewritten_bullet: str = Field(description="A CV bullet rewritten for this job, using only the cited facts")
    fact_ids: list[str]


class LetterParagraph(BaseModel):
    text: str
    fact_ids: list[str] = Field(description="Candidate fact ids this paragraph relies on")
    source_ids: list[str] = Field(description="Company page ids this paragraph relies on, e.g. 'nimbus-ledger/news'")


class JobFit(BaseModel):
    fit_summary: str = Field(description="2-3 sentences on how the candidate stands out for this job")
    match_score: int = Field(description="0-100: how well the candidate's facts cover the requirements")
    strengths: list[Strength] = Field(description="3-5 strongest points")
    gaps: list[str] = Field(description="Requirements the facts don't cover, each with a short honest tip")
    cv_edits: list[CvEdit] = Field(description="3-5 concrete CV adjustments for this job")
    keywords: list[str] = Field(description="Terms from the posting the candidate genuinely has, to use in the CV")
    cover_letter: list[LetterParagraph] = Field(description="3-4 short paragraphs")


SYSTEM = """You are a career coach helping a candidate tailor their application to one job.

Hard rules:
- Use only the candidate's facts for anything about them. Reword and emphasise, but never add tools,
  numbers, employers, scope or results the facts don't state. Cite fact ids on every point.
- Company statements may only use the company pages provided; cite their source ids.
- Never claim skills from the "no experience with" list. Put real gaps under gaps, with an honest tip.
- Keywords must be things the facts show the candidate has.
- Plain, confident English. No clichés like "I am writing to express my interest"."""


def coach(llm: LLMClient, candidate: Candidate, job: Job, company: Company) -> dict:
    sources = "\n".join(f"[{sid}] {text}" for sid, text in company.sources().items())
    prompt = (
        f"Candidate: {candidate.name}, {candidate.location}. {candidate.work_authorization}\n"
        f"Facts:\n{candidate.facts_text()}\n"
        f"No experience with: {', '.join(candidate.no_experience_with) or 'not specified'}\n\n"
        f"Job: {job.title} at {company.name} ({job.location})\n{job.description}\nTeam: {job.team}\n\n"
        f"Company pages:\n{sources}"
    )
    fit = llm.generate(agent="fit_coach", system=SYSTEM, prompt=prompt, schema=JobFit)
    return check_fit(fit.model_dump(), candidate, company)


def check_fit(fit: dict, candidate: Candidate, company: Company) -> dict:
    """Remove anything not backed by real facts or company pages, and say what was removed."""
    facts = candidate.fact_map()
    sources = set(company.sources())
    facts_text = " ".join(facts.values()).lower()
    removed: list[str] = []

    def claims_missing_skill(text: str) -> str | None:
        for skill in candidate.no_experience_with:
            if re.search(rf"\b{re.escape(skill.lower())}\b", text.lower()) and not re.search(
                r"\b(learn|learning|keen to|eager to)\b", text.lower()
            ):
                return skill
        return None

    def keep(items: list[dict], label: str, text_key: str) -> list[dict]:
        kept = []
        for item in items:
            item["fact_ids"] = [f for f in item["fact_ids"] if f in facts]
            skill = claims_missing_skill(item[text_key])
            if not item["fact_ids"]:
                removed.append(f"A {label} that wasn't backed by any of your facts.")
            elif skill:
                removed.append(f"A {label} that claimed {skill}, which you don't have.")
            else:
                kept.append(item)
        return kept

    fit["match_score"] = min(max(fit["match_score"], 0), 100)
    fit["strengths"] = keep(fit["strengths"], "strength", "point")
    fit["cv_edits"] = keep(fit["cv_edits"], "CV suggestion", "rewritten_bullet")

    keywords = [k for k in fit["keywords"] if k.lower() in facts_text]
    if len(keywords) < len(fit["keywords"]):
        dropped = [k for k in fit["keywords"] if k not in keywords]
        removed.append(f"Keywords not found in your facts: {', '.join(dropped)}.")
    fit["keywords"] = keywords

    letter = []
    for p in fit["cover_letter"]:
        p["fact_ids"] = [f for f in p["fact_ids"] if f in facts]
        p["source_ids"] = [s for s in p["source_ids"] if s in sources]
        skill = claims_missing_skill(p["text"])
        if skill:
            removed.append(f"A cover letter paragraph that claimed {skill}.")
            continue
        p["uncited"] = not p["fact_ids"] and not p["source_ids"]
        letter.append(p)
    fit["cover_letter"] = letter
    fit["removed"] = removed
    return fit


def cover_letter_text(candidate: Candidate, fit: dict) -> str:
    return "\n\n".join([*(p["text"] for p in fit["cover_letter"]), f"Best regards,\n{candidate.name}"])
