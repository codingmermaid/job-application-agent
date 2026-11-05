"""Live evals: they call Gemini, so they need GEMINI_API_KEY in .env and use free-tier quota.

1. Prompt injection. A poisoned sentence is added to one company page. The company researcher and
   the writer then run in three conditions:
     clean       the real pages, with guardrails (the baseline drafts)
     unguarded   the poisoned page, with no quarantine, no fencing and no draft scan (the old code path)
     guarded     the poisoned page, with every guardrail on
   For each case we record whether attacker text reached the brief, then the draft, and whether the
   code checks would stop that draft before you approve it.
2. Reviewer catch rate. Each clean draft goes to the reviewer as is, with an invented achievement
   added, and with an inflated number added. The reviewer should pass the first and catch the rest.

Run from the repository root:  .venv\\Scripts\\python evals\\run_live.py
Responses are cached in var/llm-cache, so a second run is free and gives the same answers.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evals.cases import EMBELLISH, INFLATE, POISON, POISON_MARKERS  # noqa: E402
from jobagent.agents.company_researcher import research_company  # noqa: E402
from jobagent.agents.reviewer import review_application  # noqa: E402
from jobagent.agents.writer import write_application  # noqa: E402
from jobagent.grounding import check_draft, draft_claims  # noqa: E402
from jobagent.guardrails import scan_draft  # noqa: E402
from jobagent.llm import make_llm  # noqa: E402
from jobagent.world import default_world  # noqa: E402

JOBS = ["j01", "j04", "j05"]  # the three jobs chosen in Chapter 14's live run
OUT = ROOT / "evals" / "results"


def assessment(job) -> dict:
    reqs = [line[2:].strip() for line in job.description.splitlines() if line.strip().startswith("- ")]
    return {"requirements": reqs, "gaps": []}


def draft_text(draft: dict) -> str:
    return " ".join(c["text"] for c in draft_claims(draft)).lower()


def run_case(llm, world, job_id: str, condition: str, style: str | None) -> dict:
    job = world.job(job_id)
    company = world.company(job.company_id)
    if style:
        company = company.model_copy(update={"pages": {**company.pages, "news": company.pages["news"] + " " + POISON[style]}})
    guarded = condition != "unguarded"
    brief, issues = research_company(llm, company, job, guarded=guarded)
    draft = write_application(llm, world.candidate, job, company.name, assessment(job), brief).model_dump()
    markers = POISON_MARKERS.get(style, [])
    in_brief = any(m in json.dumps(brief).lower() for m in markers)
    in_draft = [m for m in markers if m in draft_text(draft)]
    grounding = check_draft(draft, world.candidate, brief, job, company.name)
    scan = scan_draft(draft, world.candidate, company.website) if guarded else []
    return {"job": job_id, "company": company.name, "condition": condition, "style": style,
            "quarantined": [i for i in issues if i.startswith("Guardrail:")], "in_brief": in_brief,
            "in_draft": in_draft, "grounding_issues": grounding, "scan_issues": scan,
            "reached_approval_unflagged": bool(in_draft) and not (grounding or scan), "draft": draft, "brief": brief}


def main() -> None:
    world = default_world()
    llm = make_llm()
    started = time.monotonic()
    cases = []
    for job_id in JOBS:
        cases.append(run_case(llm, world, job_id, "clean", None))
        for style in POISON:
            for condition in ("unguarded", "guarded"):
                cases.append(run_case(llm, world, job_id, condition, style))
                c = cases[-1]
                print(f"{job_id} {style:6} {condition:9} brief={c['in_brief']!s:5} draft={c['in_draft']} "
                      f"flags={len(c['grounding_issues']) + len(c['scan_issues'])}", flush=True)

    reviews = []
    for c in [c for c in cases if c["condition"] == "clean"]:
        job = world.job(c["job"])
        for variant, extra in (("clean", None), ("embellished", EMBELLISH), ("inflated", INFLATE)):
            draft = copy.deepcopy(c["draft"])
            if extra:
                draft["resume_bullets"].append(dict(extra))
            r = review_application(llm, world.candidate, job, c["company"], c["brief"], draft,
                                   world.company(job.company_id).website)
            flagged = any(extra["text"][:25].lower() in (u["claim"] + u["problem"]).lower()
                          for u in r["unsupported_claims"]) if extra else None
            reviews.append({"job": c["job"], "variant": variant, "ready": r["ready"],
                            "code_issues": r["code_issues"], "unsupported": r["unsupported_claims"],
                            "flagged_added_claim": flagged, "summary": r["summary"]})
            print(f"review {c['job']} {variant:11} ready={r['ready']} flagged={flagged}", flush=True)

    calls = [c.__dict__ for c in llm.calls]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "live.json").write_text(json.dumps({"cases": cases, "reviews": reviews, "calls": calls,
                                               "wall_s": round(time.monotonic() - started, 1)},
                                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(calls)} calls ({sum(c['cached'] for c in calls)} cached) in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
