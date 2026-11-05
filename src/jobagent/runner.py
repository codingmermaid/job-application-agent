"""Advances runs in short, time-boxed slices.

Each call runs graph steps until the time budget is used, the graph pauses for the user, or it
finishes. Progress is checkpointed after every step, so the next call carries on from there. This
suits serverless hosting, where one request can't run for minutes: the browser keeps polling and
each poll does a slice of work.
"""
from __future__ import annotations

import time
from typing import Any

from langgraph.types import Command
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from .graph import new_run
from .ops import OpsStore


class Runner:
    def __init__(self, graph, ops: OpsStore | None = None):
        self.graph = graph
        # Locks and the last error per run live in the database, so they hold across server instances.
        self.ops = ops or OpsStore(create_engine("sqlite://", poolclass=StaticPool,
                                                 connect_args={"check_same_thread": False}))

    def _config(self, run_id: str) -> dict:
        return {"configurable": {"thread_id": run_id}}

    def start(self, request: str, candidate=None, *, budget_s: float = 0) -> str:
        """Save the run's input without doing any work, so starting a run is always instant.

        Doing even one step here would put an LLM call (with its retries) inside the request that
        starts the run, which can outlast a serverless time limit. The first poll does the work instead.
        """
        state = new_run(request, candidate)
        self.graph.update_state(self._config(state["run_id"]), state, as_node="__start__")
        if budget_s > 0:
            self._step(state["run_id"], None, budget_s)
        return state["run_id"]

    def advance(self, run_id: str, *, budget_s: float = 10, resume: Any = None, retry: bool = False) -> dict:
        """Do up to budget_s of work (or deliver the user's answer), then report where the run stands."""
        if retry:
            self.ops.clear_error(run_id)
        if resume is None and self.view(run_id)["status"] != "running":
            return self.view(run_id)  # waiting for the user, finished, or failed: nothing to do
        self._step(run_id, Command(resume=resume) if resume is not None else None, budget_s)
        return self.view(run_id)

    def _step(self, run_id: str, graph_input: Any, budget_s: float) -> None:
        with self.ops.try_lock(run_id) as got:
            if not got:
                return  # another request (maybe on another instance) is already advancing this run
            try:
                started = time.monotonic()
                for _ in self.graph.stream(graph_input, self._config(run_id), stream_mode="updates"):
                    if time.monotonic() - started >= budget_s:
                        break  # the checkpoint holds everything done so far
            except Exception as exc:  # e.g. Gemini unavailable: the failed step can be retried from its checkpoint
                self.ops.set_error(run_id, f"{type(exc).__name__}: {exc}")

    def view(self, run_id: str) -> dict:
        snapshot = self.graph.get_state(self._config(run_id))
        values = snapshot.values or {}
        interrupts = [i.value for task in snapshot.tasks for i in task.interrupts]
        error = self.ops.get_error(run_id)
        if interrupts:
            status = "waiting"
        elif error and snapshot.next:
            status = "error"
        elif snapshot.next:
            status = "running"
        else:
            status = values.get("status", "running" if not values else "done")
        return {"run_id": run_id, "status": status, "state": values,
                "interrupt": interrupts[0] if interrupts else None, "error": error if status == "error" else None}
