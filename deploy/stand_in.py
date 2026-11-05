"""The real Vercel entry point (api/index.py) with the LLM replaced by a stand-in, for rehearsals.

The rehearsal tests the platform: instances that die, concurrent polls, the time limit, checkpoints
in Postgres. None of that depends on what the model says, so when live calls are not possible (the
free Gemini quota is 20 requests per model per day) the agents get the test suite's scripted answers,
each after a delay drawn from the latencies measured in the live evals.

    python deploy/simulate.py --stand-in
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from jobagent.llm import CallRecord  # noqa: E402
from tests.conftest import BRIEFS, DRAFTS, PLAN, READY, RESEARCH  # noqa: E402
from jobagent.agents.company_researcher import CompanyBrief  # noqa: E402
from jobagent.agents.coordinator import SearchPlan  # noqa: E402
from jobagent.agents.job_researcher import JobResearch  # noqa: E402
from jobagent.agents.reviewer import ReviewVerdict  # noqa: E402
from jobagent.agents.writer import ApplicationDraft  # noqa: E402

LATENCIES_S = (1.6, 2.1, 2.4, 2.8, 2.8, 3.1, 3.6, 4.4, 6.5)  # live eval calls, rounded


class StandInLLM:
    """Scripted answers, chosen by the company named in the prompt, after a realistic delay."""
    model = "stand-in"

    def __init__(self):
        self.calls: list[CallRecord] = []
        self.random = random.Random()

    def generate(self, *, agent: str, system: str, prompt: str, schema):
        delay = self.random.choice(LATENCIES_S)
        time.sleep(delay)
        self.calls.append(CallRecord(agent, self.model, delay, len(prompt) // 4, 300))
        if schema is SearchPlan:
            return PLAN
        if schema is JobResearch:
            return RESEARCH
        if schema is ReviewVerdict:
            return READY
        if schema is CompanyBrief:
            return next((b for b in BRIEFS if b.evidence[0].source.split("/")[0] in prompt), BRIEFS[0])
        if schema is ApplicationDraft:
            names = ("Nimbus Ledger", "Quill", "Brightloom")
            return next((d for d, n in zip(DRAFTS, names) if n in prompt), DRAFTS[0])
        raise KeyError(f"no stand-in answer for {schema.__name__}")


import jobagent.web.app as web_app  # noqa: E402

_real_create_app = web_app.create_app
web_app.create_app = lambda **kw: _real_create_app(llm=StandInLLM(), **kw)
from api.index import app  # noqa: E402,F401  the real entry point, now built with the stand-in
