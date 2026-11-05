"""Guardrails around the agents: untrusted text in, risky text out.

Grounding (grounding.py) stops the agents inventing facts. It does not stop a web page from giving
them orders: a quote copied word for word from a poisoned page passes verify_brief. These checks
cover that gap, with plain code that no prompt can talk its way past.

  screen_text()       flags instruction-like text in anything the agents read
  quarantine()        removes flagged sentences from untrusted pages before an LLM sees them
  fence()             wraps untrusted text in markers the system prompt tells the model to treat as data
  scan_draft()        flags links, emails and instruction echoes in a draft before you approve it
  screen_request()    limits what a user request can be (length, control characters)
  redact()            masks emails and phone numbers before anything is logged
"""
from __future__ import annotations

import re
import unicodedata

from .models import Candidate

MAX_REQUEST_CHARS = 600

# Phrases that address the model instead of a human reader. Each pattern is one family of attack.
INJECTION_PATTERNS: dict[str, re.Pattern] = {
    "override": re.compile(r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your)\b"
                           r".{0,30}\b(instructions?|rules?|prompts?|guidelines?|directions?)\b", re.I | re.S),
    "role_play": re.compile(r"\b(you are now|pretend to be|from now on,? you)\b|\bact as (the |a |an )?(recruiter|hiring "
                           r"manager|hr|assistant|admin(istrator)?|system|developer|bot)\b", re.I),
    "addressing_ai": re.compile(r"\b(note|message|instructions?) (to|for) (the )?(ai|assistant|llm|model|agent|language model|chatbot)s?\b"
                                r"|\b(ai|llm|language model) (assistants?|agents?|models?),? (must|should|reading this)\b", re.I),
    "fake_system": re.compile(r"(^|\n|\[|<)\s*(system|assistant|developer)\s*(prompt|message)?\s*[:\]>]", re.I),
    # Sending someone else's data somewhere. "Send your CV to careers@..." is normal advice to a reader.
    "exfiltration": re.compile(r"\b(send|forward|email|e-mail|upload|post|share)\b.{0,25}\b(the (candidate|applicant|user)'?s?|"
                               r"all|every|applicant|candidate)\b.{0,30}\b(cv|resume|résumé|data|details|profile|"
                               r"application|password|key)s?\b.{0,40}\b(to|at)\b", re.I | re.S),
    "hidden_claim": re.compile(r"\b(say|state|claim|mention|write|add|include)\b.{0,30}\b(that )?(the )?(candidate|applicant)\b"
                               r".{0,40}\b(has|have|is|knows?)\b", re.I | re.S),
}
URL_RE = re.compile(r"\bhttps?://[^\s<>\"')]+|\bwww\.[^\s<>\"')]+", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(\.[\w-]+)+\b")
PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def _clean(text: str) -> str:
    """Drop control and invisible formatting characters (zero-width spaces, bidi overrides)."""
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")


def screen_text(text: str) -> list[str]:
    """Names of the injection patterns found in the text; empty if it reads like normal content."""
    cleaned = _clean(text)
    hits = [name for name, pattern in INJECTION_PATTERNS.items() if pattern.search(cleaned)]
    if cleaned != text and any(unicodedata.category(ch) == "Cf" for ch in text):
        hits.append("invisible_characters")
    return hits


def quarantine(text: str) -> tuple[str, list[str]]:
    """Remove flagged sentences from an untrusted page. Returns (kept text, removed sentences)."""
    kept, removed = [], []
    for sentence in _SENTENCE.split(_clean(text)):
        if not sentence.strip():
            continue
        (removed if screen_text(sentence) else kept).append(sentence.strip())
    return " ".join(kept), removed


def fence(source_id: str, text: str) -> str:
    """Mark text as data. The system prompt says nothing inside these markers is an instruction."""
    text = text.replace("<untrusted", "&lt;untrusted").replace("</untrusted", "&lt;/untrusted")
    return f'<untrusted source="{source_id}">\n{text}\n</untrusted>'


def allowed_links(candidate: Candidate, company_website: str | None = None) -> set[str]:
    """Links and addresses a draft may contain: the candidate's own, and the employer's site."""
    allowed = {candidate.email.lower()} if candidate.email else set()
    for url in (candidate.portfolio_url, company_website):
        if url:
            allowed.add(_host(url))
    return allowed


def _host(url: str) -> str:
    return re.sub(r"^(https?://)?(www\.)?", "", url.lower()).split("/")[0]


def scan_draft(draft: dict, candidate: Candidate, company_website: str | None = None) -> list[str]:
    """Issues that must block approval: unknown links or emails, and text that reads like an injected order."""
    from .grounding import draft_claims

    allowed = allowed_links(candidate, company_website)
    issues = []
    for claim in draft_claims(draft):
        text = claim["text"]
        short = text if len(text) <= 80 else text[:77] + "..."
        for url in URL_RE.findall(text):
            if _host(url).rstrip(".,;") not in allowed:
                issues.append(f"'{short}' links to {url}, which is not the candidate's or the employer's site.")
        for email in (m.group(0) for m in EMAIL_RE.finditer(text)):
            if email.lower() not in allowed:
                issues.append(f"'{short}' contains the address {email}, which is not the candidate's.")
        hits = [h for h in screen_text(text) if h != "hidden_claim"]
        if hits:
            issues.append(f"'{short}' reads like an instruction ({', '.join(hits)}).")
    return issues


def screen_request(text: str) -> tuple[str, list[str]]:
    """Normalise a user's request. Returns (request to use, problems that should stop the run)."""
    cleaned = _clean(text).strip()
    problems = []
    if not cleaned:
        problems.append("Describe the jobs you want.")
    if len(cleaned) > MAX_REQUEST_CHARS:
        problems.append(f"Keep the request under {MAX_REQUEST_CHARS} characters.")
    return cleaned, problems


def redact(text: str) -> str:
    """Mask emails and phone numbers, for logs and monitoring."""
    return PHONE_RE.sub("[phone]", EMAIL_RE.sub("[email]", text))
