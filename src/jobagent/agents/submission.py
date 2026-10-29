"""Submission agent: fills in application fields, submits safely and records the outcome.

Safety rules:
- Never mark an application submitted without verifying the portal's confirmation.
- Retries reuse one idempotency key and check the portal first, so a retry after a timeout
  finds the earlier application instead of creating a duplicate.
- Where automated submission isn't possible, prepare a clear manual handoff instead.
"""
from __future__ import annotations

import hashlib
import time
from typing import Callable, Literal

from pydantic import BaseModel

from ..models import Candidate, Job
from ..portal import MockPortal, PortalRejected, PortalTimeout, PortalUnavailable, PortalUnsupported


class SubmissionResult(BaseModel):
    job_id: str
    status: Literal["submitted", "handoff", "failed"]
    confirmation_id: str | None = None
    attempts: int = 0
    detail: str
    handoff: dict | None = None


def idempotency_key(candidate_id: str, job_id: str) -> str:
    return hashlib.sha256(f"{candidate_id}:{job_id}".encode()).hexdigest()[:24]


def render_resume(candidate: Candidate, draft: dict) -> str:
    lines = [candidate.name, f"{candidate.location} | {candidate.email} | {candidate.portfolio_url}", "",
             draft["resume_summary"]["text"], "", "Experience and projects"]
    lines += [f"- {b['text']}" for b in draft["resume_bullets"]]
    return "\n".join(lines)


def render_cover_letter(candidate: Candidate, draft: dict) -> str:
    return "\n\n".join([*(p["text"] for p in draft["cover_letter"]), f"Best regards,\n{candidate.name}"])


def build_fields(candidate: Candidate, job: Job, draft: dict) -> dict[str, str]:
    fields = {
        "full_name": candidate.name,
        "email": candidate.email,
        "location": candidate.location,
        "portfolio_url": candidate.portfolio_url,
        "resume": render_resume(candidate, draft),
        "cover_letter": render_cover_letter(candidate, draft),
    }
    answers = draft["screening_answers"]
    for i, _question in enumerate(job.apply.questions, start=1):
        fields[f"answer_{i}"] = answers[i - 1]["answer"]["text"] if i <= len(answers) else ""
    return fields


def handoff_package(candidate: Candidate, job: Job, company_name: str, draft: dict, reason: str) -> dict:
    answers = draft["screening_answers"]
    return {
        "reason": reason,
        "how_to_apply": f"Email {job.apply.email}" if job.apply.email else f"Apply on {company_name}'s careers site",
        "subject": f"{job.title} application: {candidate.name}",
        "cover_letter": render_cover_letter(candidate, draft),
        "resume": render_resume(candidate, draft),
        "answers": {q: answers[i]["answer"]["text"] for i, q in enumerate(job.apply.questions) if i < len(answers)},
    }


def submit_application(
    portal: MockPortal,
    candidate: Candidate,
    job: Job,
    company_name: str,
    draft: dict,
    *,
    max_attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> SubmissionResult:
    key = idempotency_key(candidate.id, job.id)
    if job.apply.method != "portal":
        return SubmissionResult(
            job_id=job.id, status="handoff", detail="This employer applies by email, so it needs a manual send.",
            handoff=handoff_package(candidate, job, company_name, draft, "Apply by email"),
        )

    fields = build_fields(candidate, job, draft)
    for attempt in range(1, max_attempts + 1):
        # Check first: an earlier attempt may have gone through even if we never heard back.
        existing = portal.lookup(key)
        if existing:
            return SubmissionResult(
                job_id=job.id, status="submitted", confirmation_id=existing.confirmation_id, attempts=attempt - 1,
                detail="Found the application from an earlier attempt; no duplicate was sent.",
            )
        try:
            confirmation = portal.submit(job.id, fields, key)
        except PortalTimeout:
            continue  # unknown outcome: the next loop checks the portal before retrying
        except PortalUnavailable:
            sleep(2 ** attempt)
            continue
        except (PortalRejected, PortalUnsupported) as exc:
            return SubmissionResult(
                job_id=job.id, status="handoff", attempts=attempt,
                detail=f"The portal refused the application: {exc} Prepared for manual submission.",
                handoff=handoff_package(candidate, job, company_name, draft, str(exc)),
            )
        verified = portal.lookup(key)
        if verified and verified.confirmation_id == confirmation.confirmation_id:
            return SubmissionResult(
                job_id=job.id, status="submitted", confirmation_id=confirmation.confirmation_id,
                attempts=attempt, detail="Submitted and confirmation verified with the portal.",
            )
        return SubmissionResult(job_id=job.id, status="failed", attempts=attempt,
                                detail="The portal replied, but the confirmation could not be verified.")

    existing = portal.lookup(key)  # one last check after the final attempt
    if existing:
        return SubmissionResult(job_id=job.id, status="submitted", confirmation_id=existing.confirmation_id,
                                attempts=max_attempts, detail="Confirmed on the final check; no duplicate sent.")
    return SubmissionResult(job_id=job.id, status="failed", attempts=max_attempts,
                            detail=f"No confirmation after {max_attempts} attempts. Safe to retry later.")
