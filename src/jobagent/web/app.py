"""FastAPI web app: start runs, watch the agents work, choose jobs, approve, and track outcomes.

The app instance lives in main.py:  .venv\\Scripts\\uvicorn.exe jobagent.web.main:app
"""
from __future__ import annotations

import base64
import hmac
import os
import sqlite3
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langgraph.checkpoint.sqlite import SqliteSaver

from .. import documents
from ..agents.fit_coach import coach, cover_letter_text
from ..agents.profile_builder import build_candidate, extract_profile
from ..db import data_dir, database_url, is_postgres, libpq_url, make_engine, on_serverless
from ..graph import build_graph
from ..guardrails import screen_request
from ..llm import LLMClient, make_llm
from ..models import Candidate
from ..ops import OpsStore, RecordingLLM
from ..portal import MockPortal
from ..profiles import CoachingCache, ProfileStore
from ..runner import Runner
from ..tracker import Tracker
from ..world import World, default_world

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def description_blocks(text: str) -> list[dict]:
    """Split a posting into headings, paragraphs and bullet lists for display."""
    blocks: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("- "):
            if not blocks or blocks[-1]["type"] != "list":
                blocks.append({"type": "list", "items": []})
            blocks[-1]["items"].append(line[2:])
        elif line.endswith(":") and len(line) < 40:
            blocks.append({"type": "heading", "text": line[:-1]})
        elif ":" in line[:20] and line.split(":", 1)[0] in ("Location", "How to apply"):
            label, rest = line.split(":", 1)
            blocks.append({"type": "heading", "text": label})
            rest = rest.strip()
            blocks.append({"type": "text", "text": rest[:1].upper() + rest[1:]})
        else:
            blocks.append({"type": "text", "text": line})
    return blocks


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
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
    "Content-Security-Policy": (
        "default-src 'self'; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "script-src 'self' 'unsafe-inline' https://unpkg.com; img-src 'self' data:; frame-ancestors 'none'"
    ),
}


def make_checkpointer(url: str | None = None):
    """Postgres when DATABASE_URL points at one (needed on serverless), else a local SQLite file."""
    url = url or database_url()
    if is_postgres(url):
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(libpq_url(url), min_size=0, max_size=4, open=True,
                              kwargs={"autocommit": True, "prepare_threshold": None, "row_factory": dict_row})
        saver = PostgresSaver(pool)
        saver.setup()  # creates the checkpoint tables once; safe to call on every cold start
        return saver
    return SqliteSaver(sqlite3.connect(data_dir() / "checkpoints.db", check_same_thread=False))


def check_password(header: str | None, password: str) -> bool:
    """HTTP Basic auth: any user name, the configured password. Compared in constant time."""
    if not header or not header.startswith("Basic "):
        return False
    try:
        _, _, given = base64.b64decode(header[6:]).decode("utf-8").partition(":")
    except Exception:
        return False
    return hmac.compare_digest(given.encode(), password.encode())


def create_app(world: World | None = None, llm: LLMClient | None = None, engine=None, checkpointer=None,
               password: str | None = None, max_runs_per_day: int | None = None) -> FastAPI:
    password = password if password is not None else os.getenv("APP_PASSWORD", "")
    if on_serverless() and not password:
        # Fail closed: a public URL without a password would let anyone spend your API quota.
        raise RuntimeError("Set APP_PASSWORD before deploying; refusing to serve publicly without it.")
    if max_runs_per_day is None:
        max_runs_per_day = int(os.getenv("MAX_RUNS_PER_DAY", "20"))
    world = world or default_world()
    engine = engine or make_engine()
    ops = OpsStore(engine)
    llm = RecordingLLM(llm or make_llm(), ops)
    if checkpointer is None:
        checkpointer = make_checkpointer()
    portal = MockPortal(engine, {j.id: j.portal_behavior for j in world.jobs.values()})
    tracker = Tracker(engine)
    profiles = ProfileStore(engine, world.candidate)
    coaching = CoachingCache(engine)
    runner = Runner(build_graph(world, llm, portal, tracker, checkpointer=checkpointer, on_alert=ops.alert), ops)

    app = FastAPI(title="Job Application Agents")

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if password and request.url.path != "/health" and not check_password(request.headers.get("authorization"), password):
            return PlainTextResponse("Sign in required", status_code=401,
                                     headers={"WWW-Authenticate": 'Basic realm="job-agents"'})
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    # Changes whenever the stylesheet changes, so browsers never keep an old copy.
    static_version = int((HERE / "static" / "style.css").stat().st_mtime)
    templates.env.globals.update(world=world, static_version=static_version)
    app.state.runner, app.state.tracker, app.state.portal, app.state.llm = runner, tracker, portal, llm
    app.state.ops = ops

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
        tracker.list_runs(limit=1)  # fails (500) if the database is unreachable
        return {"ok": True}

    def runs_today() -> int:
        since = datetime.now(timezone.utc) - timedelta(days=1)
        return sum(1 for r in tracker.list_runs(limit=1000) if _aware(r["created_at"]) >= since)

    @app.get("/ops", response_class=HTMLResponse)
    def ops_page(request: Request, hours: int = 24):
        return render(request, "ops.html", stats=ops.stats(hours), runs_today=runs_today(),
                      max_runs=max_runs_per_day, database=engine.dialect.name, serverless=on_serverless())

    @app.get("/ops.json")
    def ops_json(hours: int = 24) -> dict:
        stats = ops.stats(hours)
        stats["recent_alerts"] = [{**a, "at": a["at"].isoformat()} for a in stats["recent_alerts"]]
        return {**stats, "runs_today": runs_today(), "max_runs_per_day": max_runs_per_day}

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        candidate = profiles.get()
        return render(request, "index.html", example=EXAMPLE_REQUEST, runs=tracker.list_runs(), usage=usage(),
                      candidate=candidate, is_sample=profiles.is_sample())

    @app.post("/runs")
    def start_run(request_text: str = Form(..., alias="request")):
        text, problems = screen_request(request_text)
        if problems:
            raise HTTPException(400, " ".join(problems))
        if runs_today() >= max_runs_per_day:
            ops.alert("budget", f"Run refused: {max_runs_per_day} runs in the last 24 hours")
            raise HTTPException(429, f"Daily limit of {max_runs_per_day} runs reached. Try again tomorrow.")
        run_id = runner.start(text, profiles.get())
        tracker.add_run(run_id, text)
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
        candidate = profiles.get()
        return render(request, "job.html", job=job, company=world.company(job.company_id),
                      blocks=description_blocks(job.description), candidate=candidate,
                      is_sample=profiles.is_sample(), has_coaching=coaching.get(candidate.id, job_id) is not None)

    @app.post("/board/jobs/{job_id}/coach", response_class=HTMLResponse)
    def job_coach(request: Request, job_id: str, regenerate: bool = False):
        if job_id not in world.jobs:
            raise HTTPException(404, "No such job")
        job, candidate = world.job(job_id), profiles.get()
        company = world.company(job.company_id)
        fit, error = (None if regenerate else coaching.get(candidate.id, job_id)), None
        if fit is None:
            try:
                fit = coach(llm, candidate, job, company)
                coaching.put(candidate.id, job_id, fit)
            except Exception as exc:  # e.g. Gemini unavailable
                error = str(exc)
        return render(request, "_coach.html", job=job, company=company, candidate=candidate, fit=fit,
                      error=error, facts=candidate.fact_map(),
                      letter=cover_letter_text(candidate, fit) if fit else "")

    @app.get("/board/jobs/{job_id}/coach/cover-letter.{fmt}")
    def job_coach_letter(job_id: str, fmt: str):
        candidate = profiles.get()
        fit = coaching.get(candidate.id, job_id) if job_id in world.jobs else None
        if fit is None or fmt not in ("docx", "txt"):
            raise HTTPException(404, "No cover letter yet. Open the agent panel on the job page first.")
        company = world.company(world.job(job_id).company_id).name.replace("&", "and").replace(" ", "-")
        filename = f"{candidate.name.replace(' ', '-')}-cover-letter-{company}.{fmt}"
        text = cover_letter_text(candidate, fit)
        if fmt == "docx":
            content, media = documents.text_docx(text), DOCX
        else:
            content, media = text.encode("utf-8"), "text/plain; charset=utf-8"
        return Response(content, media_type=media, headers={"Content-Disposition": content_disposition(filename)})

    @app.get("/tracker", response_class=HTMLResponse)
    def tracker_page(request: Request):
        apps = tracker.list()
        return render(request, "tracker.html", apps=apps,
                      history={a["id"]: tracker.history(a["id"]) for a in apps})

    @app.get("/portal", response_class=HTMLResponse)
    def portal_page(request: Request):
        return render(request, "portal.html", submissions=portal.list_submissions())

    return app


def _aware(dt: datetime) -> datetime:
    """SQLite returns naive datetimes even for timezone=True columns; they were stored as UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
