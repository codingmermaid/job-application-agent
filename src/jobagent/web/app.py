"""FastAPI web app: start runs, watch the agents work, choose jobs, approve, and track outcomes.

The app instance lives in main.py:  .venv\\Scripts\\uvicorn.exe jobagent.web.main:app
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langgraph.checkpoint.sqlite import SqliteSaver

from ..db import data_dir, make_engine
from ..graph import build_graph
from ..llm import LLMClient, make_llm
from ..portal import MockPortal
from ..runner import Runner
from ..tracker import Tracker
from ..world import World, default_world

HERE = Path(__file__).resolve().parent
EXAMPLE_REQUEST = (
    "Find remote junior data analyst roles that accept applicants from Serbia, shortlist the best "
    "matches, and prepare applications for the three I choose."
)
POLL_BUDGET_S = 8  # work done per browser poll; keeps each request short


def create_app(world: World | None = None, llm: LLMClient | None = None, engine=None, checkpointer=None) -> FastAPI:
    world = world or default_world()
    engine = engine or make_engine()
    llm = llm or make_llm()
    if checkpointer is None:
        checkpointer = SqliteSaver(sqlite3.connect(data_dir() / "checkpoints.db", check_same_thread=False))
    portal = MockPortal(engine, {j.id: j.portal_behavior for j in world.jobs.values()})
    tracker = Tracker(engine)
    runner = Runner(build_graph(world, llm, portal, tracker, checkpointer=checkpointer))

    app = FastAPI(title="Job Application Agents")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.globals.update(world=world, candidate=world.candidate, facts=world.candidate.fact_map())
    app.state.runner, app.state.tracker, app.state.portal, app.state.llm = runner, tracker, portal, llm

    def render(request: Request, name: str, **context) -> HTMLResponse:
        return templates.TemplateResponse(request, name, context)

    def usage() -> dict:
        calls = llm.calls
        return {"calls": len(calls), "cached": sum(c.cached for c in calls),
                "tokens_in": sum(c.input_tokens or 0 for c in calls),
                "tokens_out": sum(c.output_tokens or 0 for c in calls)}

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return render(request, "index.html", example=EXAMPLE_REQUEST, runs=tracker.list_runs(), usage=usage())

    @app.post("/runs")
    def start_run(request_text: str = Form(..., alias="request")):
        run_id = runner.start(request_text.strip())
        tracker.add_run(run_id, request_text.strip())
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_page(request: Request, run_id: str):
        view = runner.view(run_id)
        if not view["state"]:
            raise HTTPException(404, "No such run")
        return render(request, "run.html", view=view, usage=usage())

    @app.get("/runs/{run_id}/progress", response_class=HTMLResponse)
    def progress(request: Request, run_id: str, retry: bool = False):
        view = runner.advance(run_id, budget_s=POLL_BUDGET_S, retry=retry)
        return render(request, "_progress.html", view=view, usage=usage())

    @app.post("/runs/{run_id}/select")
    async def select_jobs(request: Request, run_id: str):
        form = await request.form()
        runner.advance(run_id, resume={"job_ids": form.getlist("job_ids")}, budget_s=0)
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @app.post("/runs/{run_id}/approve")
    async def approve(request: Request, run_id: str):
        form = await request.form()
        runner.advance(run_id, resume={"job_ids": form.getlist("job_ids")}, budget_s=0)
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @app.get("/board", response_class=HTMLResponse)
    def board(request: Request):
        return render(request, "board.html", jobs=list(world.jobs.values()))

    @app.get("/board/jobs/{job_id}", response_class=HTMLResponse)
    def job_page(request: Request, job_id: str):
        if job_id not in world.jobs:
            raise HTTPException(404, "No such job")
        job = world.job(job_id)
        return render(request, "job.html", job=job, company=world.company(job.company_id))

    @app.get("/tracker", response_class=HTMLResponse)
    def tracker_page(request: Request):
        apps = tracker.list()
        return render(request, "tracker.html", apps=apps,
                      history={a["id"]: tracker.history(a["id"]) for a in apps})

    @app.get("/portal", response_class=HTMLResponse)
    def portal_page(request: Request):
        return render(request, "portal.html", submissions=portal.list_submissions())

    return app
