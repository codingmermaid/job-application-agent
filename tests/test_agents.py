from jobagent.agents.job_researcher import JobResearch, research_jobs
from jobagent.agents.submission import idempotency_key, submit_application
from jobagent.grounding import check_draft, verify_brief
from jobagent.llm import ScriptedLLM

from .conftest import BRIEFS, PLAN, RESEARCH, assessment, draft


def no_sleep(_seconds):
    pass


def submit(portal, world, job_id, questions):
    job = world.job(job_id)
    name = world.company(job.company_id).name
    return submit_application(portal, world.candidate, job, name, draft(name, questions).model_dump(), sleep=no_sleep)


# submission agent

def test_timeout_after_commit_does_not_create_a_duplicate(portal, world):
    result = submit(portal, world, "j04", 1)
    assert result.status == "submitted" and result.attempts == 1
    assert "no duplicate" in result.detail
    assert len([s for s in portal.list_submissions() if s["job_id"] == "j04"]) == 1


def test_temporary_outage_is_retried(portal, world):
    result = submit(portal, world, "j05", 1)
    assert result.status == "submitted" and result.attempts == 2
    assert portal.lookup(idempotency_key(world.candidate.id, "j05")).confirmation_id == result.confirmation_id


def test_rejected_and_email_jobs_become_manual_handoffs(portal, world):
    rejected = submit(portal, world, "j11", 1)
    email = submit(portal, world, "j06", 0)
    assert rejected.status == "handoff" and "PDF" in rejected.detail
    assert email.status == "handoff" and "careers@brightloom.example" in email.handoff["how_to_apply"]
    assert portal.list_submissions() == []


def test_submitting_again_later_never_duplicates(portal, world):
    first = submit(portal, world, "j01", 2)
    again = submit(portal, world, "j01", 2)
    assert first.confirmation_id == again.confirmation_id
    assert len(portal.list_submissions()) == 1


def test_missing_answers_are_rejected_by_the_portal(portal, world):
    result = submit(portal, world, "j01", 0)  # j01 asks two questions
    assert result.status == "handoff" and "answer_1" in result.detail


# grounding checks

def test_verify_brief_drops_invented_quotes(world):
    company = world.company("nimbus-ledger")
    fake = BRIEFS[0].model_dump()
    fake["evidence"].append({"id": "E2", "source": "nimbus-ledger/news", "quote": "Nimbus raised $50M in 2026"})
    fake["talking_points"].append({"point": "Big funding round.", "evidence_ids": ["E2"]})
    cleaned, issues = verify_brief(fake, company)
    assert [e["id"] for e in cleaned["evidence"]] == ["E1"]
    assert len(cleaned["talking_points"]) == 1
    assert len(issues) == 2


def test_check_draft_catches_fabrication(world):
    d = draft("Nimbus Ledger", 2).model_dump()
    d["resume_bullets"].append({"text": "Built dbt models for finance.", "fact_ids": ["F5"], "evidence_ids": []})
    d["resume_bullets"].append({"text": "Led a team of ten.", "fact_ids": [], "evidence_ids": []})
    d["cover_letter"][0]["text"] = "I'd like to join your company."
    issues = check_draft(d, world.candidate, BRIEFS[0].model_dump(), world.job("j01"), "Nimbus Ledger")
    assert any("dbt" in i for i in issues)
    assert any("cites no facts" in i for i in issues)
    assert any("never names Nimbus Ledger" in i for i in issues)


def test_honest_learning_statements_are_allowed(world):
    d = draft("Nimbus Ledger", 2).model_dump()
    d["cover_letter"].append({"text": "I'm keen to learn Looker.", "fact_ids": [], "evidence_ids": []})
    assert check_draft(d, world.candidate, BRIEFS[0].model_dump(), world.job("j01"), "Nimbus Ledger") == []


# job researcher

def test_researcher_filters_and_cleans_model_output(world):
    research = RESEARCH.model_copy(deep=True)
    research.assessments.append(assessment("j99", 99))           # a job that doesn't exist
    research.assessments[0].matched_fact_ids.append("F77")       # a fact that doesn't exist
    llm = ScriptedLLM({JobResearch: research})
    shortlist, excluded = research_jobs(llm, world, PLAN)
    assert [i["job_id"] for i in shortlist] == ["j01", "j04", "j06"]
    assert "F77" not in shortlist[0]["matched_fact_ids"]
    assert {i["job_id"] for i in excluded} == {"j07", "j09"}
    assert shortlist[0]["url"] == "/board/jobs/j01"
