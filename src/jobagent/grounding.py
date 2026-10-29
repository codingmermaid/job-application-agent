"""Deterministic grounding checks. They catch fabrication without trusting any LLM."""
from __future__ import annotations

import re

from .models import Candidate, Company, Job


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower().strip(".\"' ")


def verify_brief(brief: dict, company: Company) -> tuple[dict, list[str]]:
    """Keep only evidence quoted word for word from a real source, and only talking points backed by it."""
    sources = {sid: _normalise(text) for sid, text in company.sources().items()}
    issues, kept = [], []
    for ev in brief["evidence"]:
        source = sources.get(ev["source"])
        if source is None:
            issues.append(f"Evidence {ev['id']} cites unknown source {ev['source']!r}; removed.")
        elif _normalise(ev["quote"]) not in source:
            issues.append(f"Evidence {ev['id']} is not a word-for-word quote from {ev['source']}; removed.")
        else:
            kept.append(ev)
    valid_ids = {ev["id"] for ev in kept}
    points = []
    for p in brief["talking_points"]:
        p["evidence_ids"] = [e for e in p["evidence_ids"] if e in valid_ids]
        if p["evidence_ids"]:
            points.append(p)
        else:
            issues.append(f"Talking point dropped, no valid evidence: {p['point']}")
    return {**brief, "evidence": kept, "talking_points": points}, issues


def draft_claims(draft: dict) -> list[dict]:
    claims = [draft["resume_summary"], *draft["resume_bullets"], *draft["cover_letter"]]
    return claims + [a["answer"] for a in draft["screening_answers"]]


def check_draft(draft: dict, candidate: Candidate, brief: dict, job: Job, company_name: str) -> list[str]:
    facts = candidate.fact_map()
    evidence = {e["id"] for e in brief["evidence"]}
    issues = []
    for claim in draft_claims(draft):
        text = claim["text"]
        short = text if len(text) <= 80 else text[:77] + "..."
        bad_facts = [f for f in claim["fact_ids"] if f not in facts]
        bad_evidence = [e for e in claim["evidence_ids"] if e not in evidence]
        if bad_facts:
            issues.append(f"'{short}' cites facts that don't exist: {', '.join(bad_facts)}.")
        if bad_evidence:
            issues.append(f"'{short}' cites company evidence that doesn't exist: {', '.join(bad_evidence)}.")
        if not claim["fact_ids"] and not claim["evidence_ids"] and not _is_learning_statement(text):
            issues.append(f"'{short}' cites no facts or evidence.")
        for skill in candidate.no_experience_with:
            if re.search(rf"\b{re.escape(skill.lower())}\b", text.lower()) and not _is_learning_statement(text):
                issues.append(f"'{short}' claims {skill}, which the candidate has no experience with.")
    if len(draft["screening_answers"]) != len(job.apply.questions):
        issues.append(f"Answers {len(draft['screening_answers'])} of {len(job.apply.questions)} application questions.")
    letter = " ".join(c["text"] for c in draft["cover_letter"])
    if company_name.lower() not in letter.lower():
        issues.append(f"The cover letter never names {company_name}.")
    return issues


def _is_learning_statement(text: str) -> bool:
    return bool(re.search(r"\b(learn|learning|keen to|eager to|looking forward)\b", text.lower()))
