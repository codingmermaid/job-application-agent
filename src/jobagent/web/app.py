"""FastAPI web app: start runs, watch the agents work, choose jobs, approve, and track outcomes.

The app instance lives in main.py:  .venv\\Scripts\\uvicorn.exe jobagent.web.main:app
"""
from __future__ import annotations

import sqlite3
import unicodedata
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langgraph.checkpoint.sqlite import SqliteSaver

from .. import documents
from ..agents.profile_builder import build_candidate, extract_profile
from ..db import data_dir, make_engine
from ..graph import build_graph
from ..llm import LLMClient, make_llm
from ..models import Candidate
from ..portal import MockPortal
from ..profiles import ProfileStore
from ..runner import Runner
from ..tracker import Tracker
from ..world import World, default_world

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def content_disposition(filename: str) -> str:
    """Attachment header that survives names like 'Jovanović': an ASCII fallback plus the UTF-8 name."""
    ascii_name = unicodedata.normalize("NFKD", filename).encode("ascii", "ignore").decode() or "download"
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"

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
    profiles = ProfileStore(engine, world.candidate)
    runner = Runner(build_graph(world, llm, portal, tracker, checkpointer=checkpointer))

    app = FastAPI(title="Job Application Agents")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.globals.update(world=world)
    app.state.runner, app.state.tracker, app.state.portal, app.state.llm = runner, tracker, portal, llm

    def render(request: Request, name: str, **context) -> HTMLResponse:
        return templates.TemplateResponse(request, name, context)

    def run_candidate(view: dict) -> Candidate:
        raw = view["state"].get("candidate")
        return Candidate.model_validate(raw) if raw else world.candidate

    def render_run(request: Request, name: str, view: dict) -> HTMLResponse:
        candidate = run_candidate(view)
        return render(request, name, view=view, usage=usage(), candidate=candidate, facts=candidate.fact_map())

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
        candidate = profiles.get()
        return render(request, "index.html", example=EXAMPLE_REQUEST, runs=tracker.list_runs(), usage=usage(),
                      candidate=candidate, is_sample=profiles.is_sample())

    @app.post("/runs")
    def start_run(request_text: str = Form(..., alias="request")):
        run_id = runner.start(request_text.strip(), profiles.get())
        tracker.add_run(run_id, request_text.strip())
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_page(request: Request, run_id: str):
        view = runner.view(run_id)
        if not view["state"]:
            raise HTTPException(404, "No such run")
        return render_run(request, "run.html", view)

    @app.get("/runs/{run_id}/progress", response_class=HTMLResponse)
    def progress(request: Request, run_id: str, retry: bool = False):
        view = runner.advance(run_id, budget_s=POLL_BUDGET_S, retry=retry)
        return render_run(request, "_progress.html", view)

    @app.get("/runs/{run_id}/apps/{job_id}/{doc}.{fmt}")
    def download(run_id: str, job_id: str, doc: str, fmt: str):
        view = runner.view(run_id)
        app_state = view["state"].get("apps", {}).get(job_id)
        if not app_state or "draft" not in app_state or doc not in ("resume", "cover-letter") or fmt not in ("docx", "txt"):
            raise HTTPException(404, "No such document")
        candidate, draft = run_candidate(view), app_state["draft"]
        company = world.company(world.job(job_id).company_id).name.replace("&", "and").replace(" ", "-")
        filename = f"{candidate.name.replace(' ', '-')}-{doc}-{company}.{fmt}"
        if fmt == "docx":
            make = documents.resume_docx if doc == "resume" else documents.cover_letter_docx
            content, media = make(candidate, draft), DOCX
        else:
            make = documents.resume_text if doc == "resume" else documents.cover_letter_text
            content, media = make(candidate, draft).encode("utf-8"), "text/plain; charset=utf-8"
        return Response(content, media_type=media, headers={"Content-Disposition": content_disposition(filename)})

    # your profile

    @app.get("/profile", response_class=HTMLResponse)
    def profile_page(request: Request, saved: bool = False):
        return render(request, "profile.html", candidate=profiles.get(), is_sample=profiles.is_sample(),
                      saved=saved, error=None)

    @app.post("/profile/extract", response_class=HTMLResponse)
    async def profile_extract(request: Request, cv_file: UploadFile | None = File(None), cv_text: str = Form("")):
        try:
            if cv_file is not None and cv_file.filename:
                text = documents.extract_text(cv_file.filename, await cv_file.read())
            elif cv_text.strip():
                text = cv_text.strip()[:documents.MAX_CV_CHARS]
            else:
                raise ValueError("Upload a CV file or paste its text.")
            draft = extract_profile(llm, text)
        except Exception as exc:  # bad file, or the LLM is unavailable
            return render(request, "profile.html", candidate=profiles.get(), is_sample=profiles.is_sample(),
                          saved=False, error=str(exc))
        return render(request, "profile_review.html", draft=draft)

    @app.post("/profile/save")
    def profile_save(name: str = Form(...), email: str = Form(""), location: str = Form(""), country: str = Form(""),
                     work_authorization: str = Form(""), portfolio_url: str = Form(""), headline: str = Form(""),
                     facts: str = Form(...), no_experience_with: str = Form("")):
        candidate = build_candidate(
            name=name, email=email, location=location, country=country, work_authorization=work_authorization,
            portfolio_url=portfolio_url, headline=headline, facts=facts.splitlines(),
            no_experience_with=no_experience_with.split(","),
        )
        if not candidate.facts:
            raise HTTPException(400, "Add at least one fact.")
        profiles.save(candidate)
        return RedirectResponse("/profile?saved=true", status_code=303)

    @app.post("/profile/edit", response_class=HTMLResponse)
    def profile_edit(request: Request):
        return render(request, "profile_review.html", draft=profiles.get())

    @app.post("/profile/reset")
    def profile_reset():
        profiles.reset()
        return RedirectResponse("/profile", status_code=303)

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
