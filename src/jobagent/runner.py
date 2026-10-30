"""Advances runs in short, time-boxed slices.

Each call runs graph steps until the time budget is used, the graph pauses for the user, or it
finishes. Progress is checkpointed after every step, so the next call carries on from there. This
suits serverless hosting, where one request can't run for minutes: the browser keeps polling and
each poll does a slice of work.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from langgraph.types import Command

from .graph import new_run


class Runner:
    def __init__(self, graph):
        self.graph = graph
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._errors: dict[str, str] = {}  # last failure per run; cleared on retry

    def _config(self, run_id: str) -> dict:
        return {"configurable": {"thread_id": run_id}}

    def _lock(self, run_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(run_id, threading.Lock())

    def start(self, request: str, candidate=None, *, budget_s: float = 0) -> str:
        state = new_run(request, candidate)
        self._step(state["run_id"], state, budget_s)
        return state["run_id"]

    def advance(self, run_id: str, *, budget_s: float = 10, resume: Any = None, retry: bool = False) -> dict:
        """Do up to budget_s of work (or deliver the user's answer), then report where the run stands."""
        if retry:
            self._errors.pop(run_id, None)
        if resume is None and self.view(run_id)["status"] != "running":
            return self.view(run_id)  # waiting for the user, finished, or failed: nothing to do
        self._step(run_id, Command(resume=resume) if resume is not None else None, budget_s)
        return self.view(run_id)

    def _step(self, run_id: str, graph_input: Any, budget_s: float) -> None:
        lock = self._lock(run_id)
        if not lock.acquire(blocking=False):
            return  # another request is already advancing this run
        try:
            started = time.monotonic()
            for _ in self.graph.stream(graph_input, self._config(run_id), stream_mode="updates"):
                if time.monotonic() - started >= budget_s:
                    break  # the checkpoint holds everything done so far
        except Exception as exc:  # e.g. Gemini unavailable: the failed step can be retried from its checkpoint
            self._errors[run_id] = f"{type(exc).__name__}: {exc}"
        finally:
            lock.release()

    def view(self, run_id: str) -> dict:
        snapshot = self.graph.get_state(self._config(run_id))
        values = snapshot.values or {}
        interrupts = [i.value for task in snapshot.tasks for i in task.interrupts]
        error = self._errors.get(run_id)
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
