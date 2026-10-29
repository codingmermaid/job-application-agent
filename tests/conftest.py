import pytest
from sqlalchemy import create_engine

from jobagent.agents.company_researcher import CompanyBrief, Evidence, TalkingPoint
from jobagent.agents.coordinator import SearchPlan
from jobagent.agents.job_researcher import JobAssessment, JobResearch
from jobagent.agents.reviewer import ReviewVerdict
from jobagent.agents.writer import ApplicationDraft, Claim, ScreeningAnswer
from jobagent.portal import MockPortal
from jobagent.tracker import Tracker
from jobagent.world import World


@pytest.fixture
def world() -> World:
    return World()


@pytest.fixture
def engine(tmp_path):
    return create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})


@pytest.fixture
def portal(world, engine) -> MockPortal:
    return MockPortal(engine, {j.id: j.portal_behavior for j in world.jobs.values()})


@pytest.fixture
def tracker(engine) -> Tracker:
    return Tracker(engine)


PLAN = SearchPlan(keywords=["analyst", "data scientist"], seniority="junior", remote_only=True, applicant_country="Serbia",
                  max_applications=3, other_constraints=[])


def assessment(job_id: str, score: int, eligible: str = "yes", seniority_fit: bool = True) -> JobAssessment:
    return JobAssessment(
        job_id=job_id, requirements=["SQL", "BI tool"], location_constraint="Remote", eligible=eligible,
        eligibility_reason="stated in posting", seniority_fit=seniority_fit, application_instructions="Portal",
        match_score=score, matched_fact_ids=["F4", "F5"], gaps=[], summary="Good match",
    )


RESEARCH = JobResearch(assessments=[
    assessment("j01", 90), assessment("j04", 85), assessment("j06", 70),
    assessment("j07", 80, eligible="no"), assessment("j09", 40, seniority_fit=False),
])


def brief(company_source: str, quote: str) -> CompanyBrief:
    return CompanyBrief(
        overview="Overview.", role_context="Context.",
        talking_points=[TalkingPoint(point="A real point.", evidence_ids=["E1"])],
        evidence=[Evidence(id="E1", source=company_source, quote=quote)],
    )


BRIEFS = [
    brief("nimbus-ledger/news", "launched multi-currency invoicing"),
    brief("quill-analytics/about", "builds dashboards and reporting for retailers and non-profit organisations"),
    brief("brightloom/careers", "We hire remotely across Europe, the Middle East and Africa (EMEA)."),
]


def draft(company: str, questions: int, fact: str = "F4") -> ApplicationDraft:
    return ApplicationDraft(
        resume_summary=Claim(text="Junior analyst with Power BI and SQL experience.", fact_ids=["F8"], evidence_ids=[]),
        resume_bullets=[Claim(text="Built a weekly Power BI sales dashboard.", fact_ids=[fact], evidence_ids=[])],
        cover_letter=[
            Claim(text=f"I'd like to join {company} because of its recent work.", fact_ids=[], evidence_ids=["E1"]),
            Claim(text="I cut a monthly report from two days to half a day with SQL.", fact_ids=["F5"], evidence_ids=[]),
        ],
        screening_answers=[
            ScreeningAnswer(question=f"Q{i}", answer=Claim(text="I sped up a report with SQL.", fact_ids=["F5"], evidence_ids=[]))
            for i in range(questions)
        ],
    )


DRAFTS = [draft("Nimbus Ledger", 2), draft("Quill & Co Analytics", 1), draft("Brightloom Education", 0)]
READY = ReviewVerdict(unsupported_claims=[], missing_requirements=[], corrections=[], ready=True, summary="Looks good.")


@pytest.fixture
def scripted_responses() -> dict:
    return {SearchPlan: PLAN, JobResearch: RESEARCH, CompanyBrief: list(BRIEFS),
            ApplicationDraft: list(DRAFTS), ReviewVerdict: READY}
