"""Theme labeling via the LLM (structured output)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from lfx_insights.llm.client import LLMClient
    from lfx_insights.models import Corpus, Theme

_PROMPT_TEMPLATE = (
    "You are labeling a cluster of research papers discovered by unsupervised "
    "clustering. Give a short, specific theme label (<= 8 words), a one-sentence "
    "rationale, and up to 6 keywords.\n\n"
    "Salient terms: {keywords}\n\n"
    "Representative paper titles:\n{titles}\n"
)


class ThemeLabel(BaseModel):
    label: str
    rationale: str = ""
    keywords: list[str] = Field(default_factory=list)


def _prompt_for(theme: Theme, corpus: Corpus, max_titles: int = 6) -> str:
    titles = []
    for pid in theme.paper_ids[:max_titles]:
        paper = corpus.by_id(pid)
        if paper:
            titles.append(f"- {paper.title}")
    return _PROMPT_TEMPLATE.format(
        keywords=", ".join(theme.keywords) or "(none)",
        titles="\n".join(titles) or "(none)",
    )


def label_themes(themes: list[Theme], corpus: Corpus, llm: LLMClient) -> list[Theme]:
    """Fill ``label``/``rationale``/``keywords`` for each theme via the LLM."""
    for theme in themes:
        prompt = _prompt_for(theme, corpus)
        result = llm.complete_structured(prompt, ThemeLabel)
        theme.label = result.label
        theme.rationale = result.rationale
        if result.keywords:
            merged = list(dict.fromkeys([*theme.keywords, *result.keywords]))
            theme.keywords = merged[:8]
    return themes
