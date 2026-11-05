"""Rehearse a Vercel deployment on your own machine, with the conditions that break laptop code.

What it imitates:
  - several short-lived instances of api/index.py (the real Vercel entry point), each its own process
    with its own empty /tmp, so nothing in memory or on local disk is shared between requests
  - a router that sends each request to any instance, and recycles instances (cold starts)
  - a 30-second limit per request (vercel.json maxDuration)
  - an instance killed in the middle of a request, as happens when a function times out or crashes
  - VERCEL=1, so the app runs exactly as deployed: Postgres required, password required
What it does not imitate: Vercel's build, its network, regions or real cold-start times.

    pip install pgserver          # an embedded Postgres, only for this rehearsal
    .venv\\Scripts\\python deploy\\simulate.py --out deploy-sim.json

Needs GEMINI_API_KEY in .env: the agents make real LLM calls, as they would when deployed.
"""
from __future__ import annotations

import argparse
import base64
import itertools
import json
import os
import random
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
MAX_DURATION_S = 30
REQUEST = ("Find remote junior data analyst roles that accept applicants from Serbia, shortlist the best "
           "matches, and prepare applications for the three I choose.")
CHOOSE = ["j01", "j04", "j05"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Instance:
    """One function instance: a fresh process with an empty data directory."""
    counter = itertools.count(1)

    def __init__(self, env: dict):
        self.id = next(self.counter)
        self.port = free_port()
        self.tmp = tempfile.mkdtemp(prefix=f"fn{self.id}-")
        started = time.monotonic()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.index:app", "--port", str(self.port), "--log-level", "warning"],
            cwd=ROOT, env={**env, "JOBAGENT_DATA_DIR": self.tmp}, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        while True:  # a cold start lasts until the app answers
            try:
                if httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError(self.proc.stderr.read().decode()[-2000:])
            time.sleep(0.1)
        self.cold_start_s = time.monotonic() - started

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def kill(self) -> None:
        self.proc.kill()
        self.proc.wait()


class Platform:
    """Routes each request to some instance, recycles instances and enforces the time limit."""

    def __init__(self, env: dict, instances: int, recycle_every: int, password: str, seed: int):
        self.env, self.password = env, password
        self.recycle_every = recycle_every
        self.random = random.Random(seed)
        self.pool = [Instance(env) for _ in range(instances)]
        self.log: list[dict] = [{"event": "cold_start", "instance": i.id, "cold_start_s": round(i.cold_start_s, 2), "t": 0.0}
                                for i in self.pool]
        self.requests = 0
        self.t0 = time.monotonic()
        self.auth = {"authorization": "Basic " + base64.b64encode(f"me:{password}".encode()).decode()}

    def now(self) -> float:
        return round(time.monotonic() - self.t0, 2)

    def recycle(self, index: int, reason: str) -> Instance:
        old = self.pool[index]
        old.kill()
        new = Instance(self.env)
        self.pool[index] = new
        self.log.append({"event": "cold_start", "instance": new.id, "replaces": old.id, "reason": reason,
                         "cold_start_s": round(new.cold_start_s, 2), "t": self.now()})
        return new

    def request(self, method: str, path: str, *, kill_after_s: float | None = None, **kwargs) -> httpx.Response | None:
        self.requests += 1
        index = self.random.randrange(len(self.pool))
        cold = False
        if self.recycle_every and self.requests % self.recycle_every == 0:
            self.recycle(index, "idle instance reclaimed")
            cold = True
        inst = self.pool[index]
        start = self.now()
        killer = None
        if kill_after_s is not None:
            killer = threading.Timer(kill_after_s, inst.kill)
            killer.start()
        status, error = None, None
        try:
            r = httpx.request(method, inst.url + path, headers=self.auth, timeout=MAX_DURATION_S,
                              follow_redirects=False, **kwargs)
            status = r.status_code
        except httpx.TimeoutException:
            r, status, error = None, 504, "FUNCTION_INVOCATION_TIMEOUT"
        except httpx.HTTPError as exc:
            r, status, error = None, 502, f"instance died: {type(exc).__name__}"
        if killer is not None:
            killer.join()
            self.recycle(index, "crashed mid-request")
        self.log.append({"event": "request", "method": method, "path": re.sub(r"/runs/\w+", "/runs/{id}", path),
                         "instance": inst.id, "cold": cold, "status": status, "error": error,
                         "t": start, "duration_s": round(self.now() - start, 2)})
        return r

    def close(self) -> None:
        for inst in self.pool:
            inst.kill()


def drive_run(p: Platform, crash_on_poll: int) -> dict:
    """Behave like the browser: start a run, poll, choose jobs, approve, poll until done."""
    r = p.request("POST", "/runs", data={"request": REQUEST})
    run_id = r.headers["location"].rsplit("/", 1)[1]
    polls, retries, stage, lock_result = 0, 0, "research", None
    while True:
        polls += 1
        if stage == "selected" and lock_result is None:
            lock_result = lock_check(p, run_id)  # while the agents are busy researching companies
            continue
        kill = 3.0 if polls == crash_on_poll else None
        r = p.request("GET", f"/runs/{run_id}/progress" + ("?retry=true" if retries and stage == "error" else ""),
                      kill_after_s=kill)
        if r is None or r.status_code >= 500:
            time.sleep(1)
            continue  # the browser's next poll lands on another instance and resumes from the checkpoint
        html = r.text
        if "Something failed" in html:
            stage, retries = "error", retries + 1
            p.log.append({"event": "run_error", "t": p.now(), "detail": re.sub(r"<[^>]+>", "", html)[:200]})
            time.sleep(10)
            continue
        stage = "running" if stage != "selected" else stage
        if "Choose up to" in html:
            p.log.append({"event": "human", "action": "select", "t": p.now()})
            p.request("POST", f"/runs/{run_id}/select", data={"job_ids": CHOOSE})
            stage = "selected"
        elif "Review and approve" in html:
            ready = re.findall(r'name="job_ids" value="(\w+)" checked', html)
            p.log.append({"event": "human", "action": "approve", "t": p.now(), "ready": ready})
            p.request("POST", f"/runs/{run_id}/approve", data={"job_ids": ready})
        elif "Results" in html or "Nothing was submitted" in html:
            return {"run_id": run_id, "polls": polls, "retries": retries, "lock_check_s": lock_result}
        time.sleep(0.5)  # the page's hx-trigger delay


def lock_check(p: Platform, run_id: str) -> dict:
    """Two polls for the same run at once, on two different instances: only one may do work."""
    a, b = p.pool[0], p.pool[1]
    results = {}

    def hit(name, inst):
        start = p.now()
        httpx.get(f"{inst.url}/runs/{run_id}/progress", headers=p.auth, timeout=MAX_DURATION_S)
        results[name] = {"instance": inst.id, "t": start, "duration_s": round(p.now() - start, 2)}

    threads = [threading.Thread(target=hit, args=(n, i)) for n, i in (("a", a), ("b", b))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    p.log.append({"event": "lock_check", **{k: v for k, v in results.items()}})
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", help="Postgres URL; default: start an embedded one with pgserver")
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--recycle-every", type=int, default=7, help="reclaim an instance every N requests")
    ap.add_argument("--crash-on-poll", type=int, default=4, help="kill the instance serving this poll")
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--out", default="deploy-sim.json")
    args = ap.parse_args()

    url = args.database_url
    if not url:
        import pgserver  # noqa: PLC0415

        server = pgserver.get_server(tempfile.mkdtemp(prefix="pg-"), cleanup_mode="stop")
        url = server.get_uri()
    password = base64.b16encode(os.urandom(8)).decode()
    from dotenv import dotenv_values  # the key goes to the instances' environment, never to the log

    env = {**os.environ, **{k: v for k, v in dotenv_values(ROOT / ".env").items() if v},
           "VERCEL": "1", "DATABASE_URL": url, "APP_PASSWORD": password, "PYTHONPATH": str(ROOT / "src")}

    p = Platform(env, args.instances, args.recycle_every, password, args.seed)
    try:
        run = drive_run(p, args.crash_on_poll)
        ops = p.request("GET", "/ops.json").json()
        portal = p.request("GET", "/portal").text
        unauth = httpx.get(p.pool[0].url + "/", timeout=10).status_code
    finally:
        p.close()

    import psycopg  # noqa: PLC0415

    with psycopg.connect(url) as conn:
        checkpoints = conn.execute("select count(*) from checkpoints where thread_id = %s", (run["run_id"],)).fetchone()[0]
        submissions = conn.execute("select job_id, count(*) from portal_submissions group by job_id").fetchall()

    reqs = [e for e in p.log if e["event"] == "request"]
    summary = {
        "run": run, "instances_used": len({e["instance"] for e in reqs}),
        "cold_starts": [e for e in p.log if e["event"] == "cold_start"],
        "requests": len(reqs), "failed_requests": [e for e in reqs if e["status"] and e["status"] >= 500],
        "max_request_s": max(e["duration_s"] for e in reqs), "limit_s": MAX_DURATION_S,
        "checkpoints_in_postgres": checkpoints,
        "portal_rows_per_job": dict(submissions), "unauthenticated_status": unauth,
        "submitted": portal.count("APP-"), "ops": ops,
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "log": p.log}, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("ops", "cold_starts", "failed_requests")}, indent=2))


if __name__ == "__main__":
    main()
