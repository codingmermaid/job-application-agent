from langgraph.checkpoint.memory import InMemorySaver

from jobagent.agents.reviewer import ReviewVerdict
from jobagent.agents.writer import ApplicationDraft
from jobagent.graph import build_graph
from jobagent.llm import ScriptedLLM
from jobagent.runner import Runner

from .conftest import DRAFTS, READY, draft


def make_runner(world, portal, tracker, responses):
    llm = ScriptedLLM(responses)
    return Runner(build_graph(world, llm, portal, tracker, checkpointer=InMemorySaver())), llm


def run_until_pause(runner, run_id):
    for _ in range(50):  # one small slice at a time, like the browser polling
        view = runner.advance(run_id, budget_s=0)
        if view["status"] != "running":
            return view
    raise AssertionError("run never paused")


def test_full_flow_selects_writes_reviews_and_submits_safely(world, portal, tracker, scripted_responses):
    runner, llm = make_runner(world, portal, tracker, scripted_responses)
    run_id = runner.start("Find remote junior data analyst roles from Serbia", budget_s=0)

    view = run_until_pause(runner, run_id)
    assert view["interrupt"]["kind"] == "select"
    shortlisted = [j["job_id"] for j in view["interrupt"]["shortlist"]]
    assert shortlisted == ["j01", "j04", "j06"]  # ineligible j07 and senior j09 excluded
    assert {j["job_id"] for j in view["interrupt"]["excluded"]} == {"j07", "j09"}

    runner.advance(run_id, resume={"job_ids": ["j01", "j04", "j06"]}, budget_s=0)
    view = run_until_pause(runner, run_id)
    assert view["interrupt"]["kind"] == "approve"
    assert portal.list_submissions() == []  # nothing submitted before approval
    assert [c.agent for c in llm.calls] == [
        "coordinator", "job_researcher",
        "company_researcher", "company_researcher", "company_researcher",
        "writer", "writer", "writer", "reviewer", "reviewer", "reviewer",
    ]

    runner.advance(run_id, resume={"job_ids": ["j01", "j04", "j06"]}, budget_s=0)
    view = run_until_pause(runner, run_id)
    assert view["status"] == "done"
    subs = {jid: app["submission"] for jid, app in view["state"]["apps"].items()}
    assert subs["j01"]["status"] == "submitted"
    assert subs["j04"]["status"] == "submitted"  # survived a timeout after the portal saved it
    assert subs["j06"]["status"] == "handoff"    # email-only employer
    assert sorted(s["job_id"] for s in portal.list_submissions()) == ["j01", "j04"]  # no duplicates
    statuses = {a["job_id"]: a["status"] for a in tracker.list(run_id)}
    assert statuses == {"j01": "submitted", "j04": "submitted", "j06": "handoff"}
    assert len(view["state"]["next_steps"]) == 3


def test_reviewer_sends_a_bad_draft_back_once(world, portal, tracker, scripted_responses):
    bad = draft("Nimbus Ledger", 2, fact="F99")  # cites a fact that doesn't exist
    scripted_responses[ApplicationDraft] = [bad, DRAFTS[0]]
    runner, llm = make_runner(world, portal, tracker, scripted_responses)
    run_id = runner.start("junior data analyst", budget_s=0)
    run_until_pause(runner, run_id)
    runner.advance(run_id, resume={"job_ids": ["j01"]}, budget_s=0)

    view = run_until_pause(runner, run_id)
    app = view["state"]["apps"]["j01"]
    assert app["revisions"] == 1 and app["review"]["ready"]
    writer_prompts = [p for agent, p in llm.prompts if agent == "writer"]
    assert "F99" in writer_prompts[1] and "Corrections" in writer_prompts[1]


def test_still_not_ready_after_revision_is_left_to_the_user(world, portal, tracker, scripted_responses):
    scripted_responses[ReviewVerdict] = READY.model_copy(update={"ready": False, "summary": "Overstated."})
    runner, _ = make_runner(world, portal, tracker, scripted_responses)
    run_id = runner.start("junior data analyst", budget_s=0)
    run_until_pause(runner, run_id)
    runner.advance(run_id, resume={"job_ids": ["j01"]}, budget_s=0)

    view = run_until_pause(runner, run_id)
    assert view["interrupt"]["kind"] == "approve"
    assert view["state"]["apps"]["j01"]["revisions"] == 1
    assert not view["state"]["apps"]["j01"]["review"]["ready"]


def test_choosing_nothing_stops_without_submitting(world, portal, tracker, scripted_responses):
    runner, _ = make_runner(world, portal, tracker, scripted_responses)
    run_id = runner.start("junior data analyst", budget_s=0)
    run_until_pause(runner, run_id)
    view = runner.advance(run_id, resume={"job_ids": []}, budget_s=0)
    assert view["status"] == "cancelled"
    assert portal.list_submissions() == []


def test_a_failing_step_can_be_retried_from_its_checkpoint(world, portal, tracker, scripted_responses):
    runner, llm = make_runner(world, portal, tracker, scripted_responses)
    original = llm.generate
    failures = {"left": 1}

    def flaky(**kwargs):
        if kwargs["agent"] == "job_researcher" and failures["left"]:
            failures["left"] -= 1
            raise RuntimeError("Gemini unavailable")
        return original(**kwargs)

    llm.generate = flaky
    run_id = runner.start("junior data analyst", budget_s=0)
    view = run_until_pause(runner, run_id)
    assert view["status"] == "error" and "Gemini unavailable" in view["error"]

    runner.advance(run_id, retry=True, budget_s=0)
    view = run_until_pause(runner, run_id)
    assert view["interrupt"]["kind"] == "select"
    assert [c.agent for c in llm.calls].count("coordinator") == 1  # earlier work was not repeated
