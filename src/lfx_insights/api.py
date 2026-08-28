"""FastAPI backend that serves the local LFX Insights browser Web UI.

Exposes:

- ``GET /health`` — lightweight liveness probe (Ollama reachable check).
- ``POST /api/analyze`` — accepts selected Word text + optional instruction,
  routes it through the LFX Insights LLM, and returns a structured result.
- ``POST /api/insights`` — interactive task-pane actions (ask, improve, review,
  gap, evidence, citations, verify) grounded in the knowledge base and Ollama.

Start with::

    python -m lfx_insights.api

or via the CLI::

    lfx-insights api
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from lfx_insights.office_kb import (
    evidence_payload,
    load_corpus,
    render_citations,
    retrieve,
    verify_text_citations,
)

if TYPE_CHECKING:
    from lfx_insights.models import Paper

app = FastAPI(title="lfx Insights API", version="2.0.0")

# Self-contained browser UI (HTML/CSS/JS) served by this same API server so the
# whole stack is a single local service. Kept next to the package, no build step.
_WEBUI_HTML = Path(__file__).resolve().parent / "webui" / "index.html"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://127.0.0.1:3000",
        "https://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Body for ``POST /api/analyze``."""

    text: str = Field(..., min_length=1, description="Selected Word text to analyse.")
    instruction: str = Field(
        default="Analyze this text for scientific accuracy, identify key claims, "
        "suggest improvements, and provide relevant context.",
        description="Optional instruction guiding the analysis.",
    )


class AnalyzeResponse(BaseModel):
    """Structured response from ``POST /api/analyze``."""

    result: str = Field(..., description="Generated analysis text.")
    model: str = Field(default="", description="LLM model used.")
    elapsed_seconds: float = Field(default=0.0, description="Server-side wall time.")


class HealthResponse(BaseModel):
    """``GET /health`` response."""

    status: str = "ok"
    ollama: str = "unknown"
    version: str = "2.0.0"


# Actions supported by the interactive task pane.
InsightsAction = Literal[
    "ask",      # free research question (prompt-based)
    "improve",  # improve the selected Word text
    "review",   # review the selected Word text
    "gap",      # generate a research gap from a topic / selection
    "evidence", # find supporting knowledge-base evidence + literature explanation
    "citations",  # generate cited, insertion-ready text with APA references
    "verify",   # verify the citations in a block of text against the corpus
]


class InsightsRequest(BaseModel):
    """Body for ``POST /api/insights``."""

    action: InsightsAction = Field(
        ..., description="Which task-pane action to run."
    )
    query: str = Field(
        default="",
        description="Free-text prompt (used by 'ask', 'gap', and as the topic "
        "for evidence/citations).",
    )
    text: str = Field(
        default="",
        description="Selected Word text (used by 'improve', 'review', and as "
        "the topic for evidence/citations when 'query' is empty).",
    )
    instruction: str = Field(
        default="",
        description="Optional extra guidance for the model.",
    )


class EvidenceItem(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    source: str | None = None
    relevance: float = 0.0
    snippet: str = ""


class CitationsBlock(BaseModel):
    intext: str = ""
    references: str = ""


class InsightsResponse(BaseModel):
    action: str
    result: str = ""
    model: str = ""
    elapsed_seconds: float = 0.0
    evidence: list[EvidenceItem] = Field(default_factory=list)
    citations: CitationsBlock = Field(default_factory=CitationsBlock)
    cited_ids: list[str] = Field(default_factory=list)
    verified: bool | None = None
    issues: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Lazy singletons (avoid import-time side effects)
# ---------------------------------------------------------------------------

_llm_client: Any = None
_llm_model: str = ""
_kb_corpus: Any = None


def _get_llm() -> Any:
    """Return (and cache) the LLM client built from the current settings."""
    global _llm_client, _llm_model
    if _llm_client is not None:
        return _llm_client

    from lfx_insights.config import load_settings
    from lfx_insights.llm.client import build_client

    settings = load_settings()
    _llm_client = build_client(settings)
    _llm_model = settings.llm.model
    return _llm_client


def _get_corpus() -> Any:
    """Return (and cache) the knowledge-base corpus used for evidence/citations."""
    global _kb_corpus
    if _kb_corpus is None:
        _kb_corpus = load_corpus()
    return _kb_corpus


def _check_ollama() -> str:
    """Return 'ok' if Ollama is reachable, else a short error string."""
    try:
        from lfx_insights.config import load_settings

        settings = load_settings()
        if not settings.llm.model.lower().startswith("ollama/"):
            return "ok (non-ollama model)"
        from lfx_insights.llm.client import validate_ollama

        validate_ollama(settings)
        return "ok"
    except Exception as exc:
        return str(exc)[:120]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", ollama=_check_ollama(), version="2.0.0")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the local browser UI for LFX Insights."""
    try:
        return HTMLResponse(_WEBUI_HTML.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Web UI not found.") from None


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """Send selected Word text through the LFX Insights LLM."""
    t0 = time.monotonic()
    try:
        llm = _get_llm()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"LLM client unavailable: {exc}",
        ) from exc

    prompt = (
        f"You are an expert scientific research assistant integrated into "
        f"Microsoft Word via the lfx Insights add-in.\n\n"
        f"Instruction: {req.instruction}\n\n"
        f"Selected text from the Word document:\n---\n{req.text}\n---\n\n"
        f"Provide a thorough, structured analysis. Use clear headings where "
        f"appropriate. Be concise but thorough."
    )

    try:
        result = llm.complete(prompt)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM generation failed: {exc}",
        ) from exc

    return AnalyzeResponse(
        result=result,
        model=_llm_model,
        elapsed_seconds=round(time.monotonic() - t0, 2),
    )


# ---------------------------------------------------------------------------
# Insights endpoint (interactive task pane)
# ---------------------------------------------------------------------------


def _topic(req: InsightsRequest) -> str:
    """Resolve the effective topic for an action from query/text."""
    return (req.query or "").strip() or (req.text or "").strip()


def _kb_context(
    corpus: Any, topic: str, k: int = 5
) -> tuple[list[tuple[Paper, float]], str]:
    """Retrieve supporting papers and render them as an indexed context block."""
    hits = retrieve(corpus, topic, k=k)
    if not hits:
        return [], ""
    lines = ["The following verified knowledge-base records are available to cite:"]
    for i, (paper, _score) in enumerate(hits, 1):
        authors = ", ".join(a.name for a in paper.authors) or "Unknown"
        head = f"[{i}] {paper.title}"
        if paper.year:
            head += f" ({paper.year})"
        meta = f"    Authors: {authors}"
        if paper.doi:
            meta += f"  | DOI: {paper.doi}"
        lines.append(head)
        lines.append(meta)
        if paper.abstract:
            lines.append(f"    Abstract: {paper.abstract[:400]}")
    return hits, "\n".join(lines)


def _evidence_items(hits: list[tuple[Paper, float]]) -> list[EvidenceItem]:
    """Convert retrieved knowledge-base hits into typed ``EvidenceItem`` objects."""
    return [EvidenceItem(**e) for e in evidence_payload(hits)]


def _chat(prompt: str) -> str:
    """Run a single LLM completion, reusing the cached client."""
    try:
        llm = _get_llm()
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"LLM client unavailable: {exc}"
        ) from exc
    try:
        return str(llm.complete(prompt))
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"LLM generation failed: {exc}"
        ) from exc


def _synthesize(
    topic: str, instruction: str, kind: str, llm_hint: str
) -> InsightsResponse:
    """Shared path for evidence / citations / ask-with-context actions."""
    corpus = _get_corpus()
    hits, ctx = _kb_context(corpus, topic, k=5)
    if not hits:
        raise HTTPException(
            status_code=404,
            detail=(
                "No knowledge-base evidence matches this topic. "
                "Run the lfx Insights pipeline to build the corpus, or rephrase."
            ),
        )
    prompt = (
        f"You are an expert scientific research assistant embedded in Microsoft "
        f"Word via the lfx Insights add-in.\n\n"
        f"Task: {kind}\n"
        f"Topic: {topic}\n"
    )
    if instruction:
        prompt += f"Additional instructions: {instruction}\n"
    prompt += (
        f"\n{ctx}\n\n"
        f"Write your answer as a single, well-structured paragraph suitable for "
        f"insertion into a Word document. Cite the provided records using their "
        f"bracketed numbers exactly as listed (e.g. [1], [2], [1;3]). "
        f"Only cite records that are listed; never invent references."
    )
    raw = _chat(prompt)
    intext, references, cited_ids = render_citations(
        raw, [p for p, _ in hits], corpus
    )
    return InsightsResponse(
        action="citations" if kind == "Generate insertion-ready cited text" else "evidence",
        result=intext,
        model=_llm_model,
        evidence=_evidence_items(hits),
        citations=CitationsBlock(intext=intext, references=references),
        cited_ids=cited_ids,
    )


@app.post("/api/insights", response_model=InsightsResponse)
async def insights(req: InsightsRequest) -> InsightsResponse:
    """Run one interactive lfx Insights task-pane action."""
    t0 = time.monotonic()
    action = req.action
    topic = _topic(req)

    if action in ("ask", "evidence", "citations", "gap") and not topic:
        raise HTTPException(
            status_code=400,
            detail="Provide a prompt (query) or select Word text for this action.",
        )

    if action == "ask":
        # Free research question, optionally grounded in retrieved context.
        corpus = _get_corpus()
        hits, ctx = _kb_context(corpus, topic, k=3)
        prompt = (
            f"You are an expert scientific research assistant embedded in "
            f"Microsoft Word via the lfx Insights add-in.\n\n"
            f"Answer the user's research question clearly and concisely. "
            f"Use headings/bullets where helpful.\n\n"
            f"Question: {topic}\n"
        )
        if req.instruction:
            prompt += f"Additional instructions: {req.instruction}\n"
        if ctx:
            prompt += (
                f"\nFor context, these verified records exist in the knowledge "
                f"base (you may reference them, but do not invent others):\n{ctx}\n"
            )
        result = _chat(prompt)
        return InsightsResponse(
            action=action,
            result=result,
            model=_llm_model,
            evidence=_evidence_items(hits),
            cited_ids=[p.id for p, _ in hits],
        )

    if action == "improve":
        if not req.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Select some Word text first, or provide it as 'text'.",
            )
        prompt = (
            f"You are an expert scientific editor embedded in Microsoft Word via "
            f"the lfx Insights add-in.\n\n"
            f"Improve the following selected text for clarity, precision, flow, "
            f"and scientific rigor. Return ONLY the improved text, preserving its "
            f"meaning and length approximately.\n\n"
            f"Selected text:\n---\n{req.text}\n---\n"
        )
        if req.instruction:
            prompt += f"\nFocus your improvements on: {req.instruction}\n"
        result = _chat(prompt)
        corpus = _get_corpus()
        hits, _ = _kb_context(corpus, req.text, k=3)
        return InsightsResponse(
            action=action, result=result, model=_llm_model,
            evidence=_evidence_items(hits),
        )

    if action == "review":
        if not req.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Select some Word text first, or provide it as 'text'.",
            )
        prompt = (
            f"You are a peer reviewer embedded in Microsoft Word via the lfx "
            f"Insights add-in.\n\n"
            f"Review the selected text. Identify: (1) strengths, (2) weaknesses, "
            f"(3) unsupported or speculative claims, (4) concrete suggestions. "
            f"Be specific and constructive.\n\n"
            f"Selected text:\n---\n{req.text}\n---\n"
        )
        if req.instruction:
            prompt += f"\nFocus your review on: {req.instruction}\n"
        result = _chat(prompt)
        corpus = _get_corpus()
        hits, _ = _kb_context(corpus, req.text, k=3)
        return InsightsResponse(
            action=action, result=result, model=_llm_model,
            evidence=_evidence_items(hits),
        )

    if action == "gap":
        corpus = _get_corpus()
        hits, _ = _kb_context(corpus, topic, k=5)
        prompt = (
            f"You are a research-strategy expert embedded in Microsoft Word via "
            f"the lfx Insights add-in.\n\n"
            f"Using ONLY the verified records below, identify a concrete, "
            f"testable research gap and explain why it matters. Do not invent "
            f"references; cite the listed records with their bracketed numbers.\n\n"
            f"Topic: {topic}\n\n"
        )
        _, ctx = _kb_context(corpus, topic, k=5)
        prompt += f"{ctx}\n"
        if req.instruction:
            prompt += f"\nAdditional instructions: {req.instruction}\n"
        raw = _chat(prompt)
        intext, references, cited_ids = render_citations(
            raw, [p for p, _ in hits], corpus
        )
        return InsightsResponse(
            action=action,
            result=intext,
            model=_llm_model,
            evidence=_evidence_items(hits),
            citations=CitationsBlock(intext=intext, references=references),
            cited_ids=cited_ids,
        )

    if action == "evidence":
        resp = _synthesize(topic, req.instruction, "Explain the literature", "evidence")
        resp.action = "evidence"
        resp.elapsed_seconds = round(time.monotonic() - t0, 2)
        return resp

    if action == "citations":
        resp = _synthesize(
            topic, req.instruction, "Generate insertion-ready cited text",
            "citations",
        )
        resp.action = "citations"
        resp.elapsed_seconds = round(time.monotonic() - t0, 2)
        return resp

    if action == "verify":
        if not req.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Provide text containing citations to verify.",
            )
        corpus = _get_corpus()
        if not corpus.papers:
            raise HTTPException(
                status_code=404,
                detail="Knowledge base is not available; cannot verify citations.",
            )
        report = verify_text_citations(req.text, corpus)
        lines = []
        if report["all_exist"] and report["cited_ids"]:
            lines.append("All cited references match verified knowledge-base records.")
        elif not report["cited_ids"]:
            lines.append("No recognizable citations were found in the text.")
        else:
            lines.append("Some citations could not be verified against the corpus:")
            lines.extend(f"  - {i}" for i in report["issues"])
        return InsightsResponse(
            action=action,
            result="\n".join(lines),
            model=_llm_model,
            evidence=[EvidenceItem(**e) for e in report["evidence"]],
            verified=report["verified"],
            issues=report["issues"],
            cited_ids=report["cited_ids"],
        )

    raise HTTPException(status_code=400, detail=f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Entry point:  python -m lfx_insights.api
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the API server with uvicorn."""
    import uvicorn

    uvicorn.run(
        "lfx_insights.api:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
