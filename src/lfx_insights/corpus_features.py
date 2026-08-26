"""Shared corpus/theme feature primitives.

Extracted to one place so the year-anchor, theme-paper resolution, and
keyword-homogeneity logic cannot drift out of sync across the scoring and
life-science modules (they previously each reimplemented these).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from lfx_insights.scoring.common import clamp01

if TYPE_CHECKING:
    from lfx_insights.models import Corpus, Paper, Theme


def max_year(corpus: Corpus) -> int | None:
    """The most recent publication year present in the corpus, or None."""
    years = [p.year for p in corpus.papers if p.year is not None]
    return max(years) if years else None


def theme_papers(theme: Theme, corpus: Corpus) -> list[Paper]:
    """Resolve a theme's paper ids to Paper objects, skipping unknown ids."""
    out: list[Paper] = []
    for pid in theme.paper_ids:
        paper = corpus.by_id(pid)
        if paper is not None:
            out.append(paper)
    return out


def theme_years(theme: Theme, corpus: Corpus) -> list[int]:
    """Publication years of a theme's resolved papers (undated papers excluded)."""
    return [p.year for p in theme_papers(theme, corpus) if p.year is not None]


def keyword_homogeneity(theme: Theme, corpus: Corpus) -> float:
    """Mean per-paper share of the theme's keywords present in the paper text.

    Word-boundary matching (not substring). Neutral 0.5 when there are no
    keywords or no resolvable papers (no signal -> do not invent structure).
    """
    if not theme.keywords:
        return 0.5
    papers = theme_papers(theme, corpus)
    if not papers:
        return 0.5
    patterns = [re.compile(rf"(?<!\w){re.escape(k.lower())}(?!\w)") for k in theme.keywords]
    shares: list[float] = []
    for paper in papers:
        text = paper.text().lower()
        present = sum(1 for pat in patterns if pat.search(text))
        shares.append(present / len(patterns))
    return clamp01(sum(shares) / len(shares))
