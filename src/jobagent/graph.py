"""The coordinator: a LangGraph workflow that delegates to the specialist agents.

    plan -> research_jobs -> select (you choose)
         -> [company_research | write | review]* (one job per step, until every draft is reviewed)
         -> approve (you approve) -> [submit]* -> finish

Each node does one unit of work (at most one LLM call), so a run can be advanced in short,
time-boxed slices and resumed from its checkpoint. select and approve are interrupts.
"""
from __future__ import annotations

import operator
import uuid
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .agents.company_researcher import research_company
from .agents.coordinator import plan_search
from .agents.job_researcher import research_jobs
from .agents.reviewer import review_application
from .agents.submission import submit_application
from .agents.writer import write_application
from .llm import LLMClient
from .portal import MockPortal
from .tracker import Tracker
from .world import World

MAX_REVISIONS = 1


class RunState(TypedDict, total=False):
    run_id: str
    request: str
    plan: dict
    shortlist: list[dict]
    excluded: list[dict]
    selected: list[str]
    apps: dict  # job id -> {brief, brief_issues, draft, review, revisions, rewrite, approved, submission}
    status: str  # running | cancelled | done
    next_steps: list[str]
    log: Annotated[list[str], operator.add]


def new_run(request: str, run_id: str | None = None) -> RunState:
    return {"run_id": run_id or uuid.uuid4().hex[:8], "request": request, "apps": {}, "status": "running", "log": []}


def next_work(state: RunState) -> str:
    """The coordinator's routing: what needs doing next for the selected jobs."""
    apps = state.get("apps", {})
    for jid in state.get("selected", []):
        if "brief" not in apps.get(jid, {}):
            return "company_research"
    for jid in state.get("selected", []):
        if "draft" not in apps[jid] or apps[jid].get("rewrite"):
            return "write"
    for jid in state.get("selected", []):
        if "review" not in apps[jid]:
            return "review"
    return "approve"


def build_graph(world: World, llm: LLMClient, portal: MockPortal, tracker: Tracker, *, checkpointer=None):
    candidate = world.candidate

    def track(state: RunState, job_id: str, status: str, detail: str = "", confirmation_id: str | None = None):
        job = world.job(job_id)
        tracker.record(state["run_id"], job_id, title=job.title, company=world.company(job.company_id).name,
                       status=status, detail=detail, confirmation_id=confirmation_id)

    def first(state: RunState, condition) -> str:
        return next(jid for jid in state["selected"] if condition(state["apps"].get(jid, {})))

    def plan(state: RunState) -> dict:
        p = plan_search(llm, state["request"], candidate)
        return {"plan": p.model_dump(),
                "log": [f"Coordinator: searching for {', '.join(p.keywords)} ({p.seniority}, remote only: {p.remote_only})"]}

    def research(state: RunState) -> dict:
        from .agents.coordinator import SearchPlan

        shortlist, excluded = research_jobs(llm, world, SearchPlan.model_validate(state["plan"]))
        for item in shortlist:
            track(state, item["job_id"], "shortlisted", f"Match {item['match_score']}: {item['summary']}")
        return {"shortlist": shortlist, "excluded": excluded,
                "log": [f"Job researcher: shortlisted {len(shortlist)}, excluded {len(excluded)}"]}

    def select(state: RunState) -> dict:
        limit = state["plan"]["max_applications"]
        choice = interrupt({"kind": "select", "shortlist": state["shortlist"], "excluded": state["excluded"],
                            "limit": limit})
        allowed = {i["job_id"] for i in state["shortlist"]}
        chosen = [j for j in (choice or {}).get("job_ids", []) if j in allowed][:limit]
        for item in state["shortlist"]:
            track(state, item["job_id"], "selected" if item["job_id"] in chosen else "skipped")
        if not chosen:
            return {"selected": [], "status": "cancelled", "log": ["You chose no jobs; stopping."]}
        return {"selected": chosen, "apps": {j: {"revisions": 0} for j in chosen},
                "log": [f"You chose {len(chosen)} job(s): {', '.join(chosen)}"]}

    def route_after_select(state: RunState) -> str:
        return END if state.get("status") == "cancelled" else next_work(state)

    def company_research(state: RunState) -> dict:
        jid = first(state, lambda a: "brief" not in a)
        job = world.job(jid)
        brief, issues = research_company(llm, world.company(job.company_id), job)
        apps = {**state["apps"], jid: {**state["apps"][jid], "brief": brief, "brief_issues": issues}}
        note = f" ({len(issues)} unverifiable item(s) removed)" if issues else ""
        return {"apps": apps, "log": [f"Company researcher: briefed {world.company(job.company_id).name}{note}"]}

    def write(state: RunState) -> dict:
        jid = first(state, lambda a: "draft" not in a or a.get("rewrite"))
        app = state["apps"][jid]
        job = world.job(jid)
        company = world.company(job.company_id).name
        assessment = next(i for i in state["shortlist"] if i["job_id"] == jid)
        revising = bool(app.get("rewrite"))
        draft = write_application(
            llm, candidate, job, company, assessment, app["brief"],
            corrections=app["review"]["corrections"] if revising else None,
            previous=app["draft"] if revising else None,
        )
        updated = {**app, "draft": draft.model_dump(), "rewrite": False,
                   "revisions": app["revisions"] + (1 if revising else 0)}
        updated.pop("review", None)
        track(state, jid, "drafted", "Revised after review" if revising else "")
        verb = "revised" if revising else "drafted"
        return {"apps": {**state["apps"], jid: updated}, "log": [f"Application writer: {verb} {job.title} at {company}"]}

    def review(state: RunState) -> dict:
        jid = first(state, lambda a: "draft" in a and "review" not in a)
        app = state["apps"][jid]
        job = world.job(jid)
        company = world.company(job.company_id).name
        result = review_application(llm, candidate, job, company, app["brief"], app["draft"])
        rewrite = not result["ready"] and app["revisions"] < MAX_REVISIONS
        track(state, jid, "reviewed", "Ready" if result["ready"] else f"{len(result['corrections'])} correction(s)")
        verdict = "ready" if result["ready"] else ("sent back for revision" if rewrite else "not ready; you decide")
        return {"apps": {**state["apps"], jid: {**app, "review": result, "rewrite": rewrite}},
                "log": [f"Reviewer: {job.title} at {company}: {verdict}"]}

    def approve(state: RunState) -> dict:
        decision = interrupt({"kind": "approve", "selected": state["selected"], "apps": state["apps"]})
        approved = [j for j in (decision or {}).get("job_ids", []) if j in state["selected"]]
        apps = {jid: {**app, "approved": jid in approved} for jid, app in state["apps"].items()}
        for jid in state["selected"]:
            track(state, jid, "approved" if jid in approved else "skipped")
        if not approved:
            return {"apps": apps, "status": "cancelled", "log": ["You approved nothing; nothing was submitted."]}
        return {"apps": apps, "log": [f"You approved {len(approved)} application(s)"]}

    def route_after_approve(state: RunState) -> str:
        return END if state.get("status") == "cancelled" else route_submissions(state)

    def route_submissions(state: RunState) -> str:
        pending = [j for j, a in state["apps"].items() if a.get("approved") and "submission" not in a]
        return "submit" if pending else "finish"

    def submit(state: RunState) -> dict:
        jid = next(j for j in state["selected"] if state["apps"][j].get("approved") and "submission" not in state["apps"][j])
        job = world.job(jid)
        company = world.company(job.company_id).name
        result = submit_application(portal, candidate, job, company, state["apps"][jid]["draft"])
        track(state, jid, result.status, result.detail, result.confirmation_id)
        apps = {**state["apps"], jid: {**state["apps"][jid], "submission": result.model_dump()}}
        return {"apps": apps, "log": [f"Submission agent: {job.title} at {company}: {result.status}. {result.detail}"]}

    def finish(state: RunState) -> dict:
        steps = []
        for jid, app in state["apps"].items():
            sub = app.get("submission")
            if not sub:
                continue
            job = world.job(jid)
            label = f"{job.title} at {world.company(job.company_id).name}"
            if sub["status"] == "submitted":
                steps.append(f"{label}: submitted ({sub['confirmation_id']}). Wait for a reply.")
            elif sub["status"] == "handoff":
                steps.append(f"{label}: send it yourself. {sub['handoff']['how_to_apply']}.")
            else:
                steps.append(f"{label}: failed. Safe to retry; no duplicate will be created.")
        return {"status": "done", "next_steps": steps, "log": ["Coordinator: all done"]}

    g = StateGraph(RunState)
    for name, fn in [("plan", plan), ("research_jobs", research), ("select", select),
                     ("company_research", company_research), ("write", write), ("review", review),
                     ("approve", approve), ("submit", submit), ("finish", finish)]:
        g.add_node(name, fn)
    work = ["company_research", "write", "review", "approve"]
    g.add_edge(START, "plan")
    g.add_edge("plan", "research_jobs")
    g.add_edge("research_jobs", "select")
    g.add_conditional_edges("select", route_after_select, [*work, END])
    for node in ("company_research", "write", "review"):
        g.add_conditional_edges(node, next_work, work)
    g.add_conditional_edges("approve", route_after_approve, ["submit", "finish", END])
    g.add_conditional_edges("submit", route_submissions, ["submit", "finish"])
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)
