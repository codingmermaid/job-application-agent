from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from jobagent.agents.fit_coach import CvEdit, JobFit, LetterParagraph, Strength, check_fit
from jobagent.llm import ScriptedLLM
from jobagent.web.app import create_app, description_blocks

FIT = JobFit(
    fit_summary="Your PostgreSQL reporting work and Power BI dashboard match this role closely.",
    match_score=88,
    strengths=[
        Strength(point="You built a Power BI dashboard used weekly.", fact_ids=["F4"]),
        Strength(point="You led a data team of ten.", fact_ids=["F42"]),  # invented fact id
    ],
    gaps=["Looker: mention you're keen to learn it."],
    cv_edits=[
        CvEdit(suggestion="Lead with the reporting win", rewritten_bullet="Cut a monthly report from two days to half a day with PostgreSQL.", fact_ids=["F5"]),
        CvEdit(suggestion="Add dbt", rewritten_bullet="Built dbt models for finance reporting.", fact_ids=["F5"]),
    ],
    keywords=["Power BI", "PostgreSQL", "Looker"],
    cover_letter=[
        LetterParagraph(text="Nimbus Ledger's multi-currency launch caught my eye.", fact_ids=[], source_ids=["nimbus-ledger/news"]),
        LetterParagraph(text="At PazarPlus I built a weekly Power BI sales dashboard.", fact_ids=["F4"], source_ids=["nimbus-ledger/made-up"]),
    ],
)


def test_check_fit_removes_unbacked_content(world):
    fit = check_fit(FIT.model_dump(), world.candidate, world.company("nimbus-ledger"))
    assert [s["fact_ids"] for s in fit["strengths"]] == [["F4"]]
    assert [e["suggestion"] for e in fit["cv_edits"]] == ["Lead with the reporting win"]
    assert fit["keywords"] == ["Power BI", "PostgreSQL"]
    assert fit["cover_letter"][1]["source_ids"] == []  # unknown company page dropped
    assert not fit["cover_letter"][0]["uncited"]
    assert any("dbt" in r for r in fit["removed"]) and any("Looker" in r for r in fit["removed"])


def test_description_blocks_find_headings_and_lists(world):
    blocks = description_blocks(world.job("j01").description)
    headings = [b["text"] for b in blocks if b["type"] == "heading"]
    assert {"You will", "Requirements", "Location", "How to apply"} <= set(headings)
    assert any(b["type"] == "list" and "Solid SQL" in b["items"] for b in blocks)


def test_coach_panel_caches_regenerates_and_downloads(world, engine, scripted_responses):
    scripted_responses[JobFit] = FIT
    llm = ScriptedLLM(scripted_responses)
    client = TestClient(create_app(world=world, llm=llm, engine=engine, checkpointer=InMemorySaver()))

    page = client.get("/board/jobs/j01").text
    assert "✨ Agent" in page and "Interview process" in page and "Take-home SQL exercise" in page
    assert client.get("/board/jobs/j01/coach/cover-letter.txt").status_code == 404  # nothing generated yet

    panel = client.post("/board/jobs/j01/coach").text
    assert "How you stand out" in panel and "CV adjustments" in panel and "Sample cover letter" in panel
    assert "led a data team of ten" not in panel  # removed by the fact check
    assert "Removed by the fact check" in panel

    calls = len(llm.calls)
    client.post("/board/jobs/j01/coach")
    assert len(llm.calls) == calls  # reopening uses the cache
    client.post("/board/jobs/j01/coach?regenerate=true")
    assert len(llm.calls) == calls + 1
    assert '<span class="dot"' in client.get("/board/jobs/j01").text  # button shows advice is ready

    letter = client.get("/board/jobs/j01/coach/cover-letter.txt").text
    assert "multi-currency" in letter and letter.rstrip().endswith("Mila Jovanović")
    docx = client.get("/board/jobs/j01/coach/cover-letter.docx")
    assert docx.status_code == 200 and docx.content[:2] == b"PK"  # a zip, i.e. a Word file


def test_coach_shows_a_retry_when_the_llm_fails(world, engine, scripted_responses):
    llm = ScriptedLLM(scripted_responses)  # no JobFit response scripted, so the call fails
    client = TestClient(create_app(world=world, llm=llm, engine=engine, checkpointer=InMemorySaver()))
    panel = client.post("/board/jobs/j01/coach").text
    assert "couldn't finish" in panel and "Try again" in panel
