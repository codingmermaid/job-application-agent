import re

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from jobagent.llm import ScriptedLLM
from jobagent.web.app import create_app


def poll_until_paused(client, run_id):
    for _ in range(50):
        html = client.get(f"/runs/{run_id}/progress").text
        if "Agents working" not in html:
            return html
    raise AssertionError("run never paused")


def test_web_flow_end_to_end(world, engine, scripted_responses):
    app = create_app(world=world, llm=ScriptedLLM(scripted_responses), engine=engine, checkpointer=InMemorySaver())
    client = TestClient(app)

    assert client.get("/health").json() == {"ok": True}
    assert "Mila Jovanović" in client.get("/").text

    response = client.post("/runs", data={"request": "junior data analyst, remote from Serbia"})
    run_id = re.search(r"/runs/(\w+)", str(response.url)).group(1)

    html = poll_until_paused(client, run_id)
    assert "Choose up to 3 jobs" in html
    client.post(f"/runs/{run_id}/select", data={"job_ids": ["j01", "j06"]})

    html = poll_until_paused(client, run_id)
    assert "Review and approve" in html and "F5" in html
    client.post(f"/runs/{run_id}/approve", data={"job_ids": ["j01", "j06"]})

    html = poll_until_paused(client, run_id)
    assert "Results" in html and "submitted" in html and "handoff" in html
    assert "APP-" in client.get("/portal").text
    assert "Nimbus Ledger" in client.get("/tracker").text
    assert client.get("/board/jobs/j01").status_code == 200
    assert client.get("/board/jobs/nope").status_code == 404
