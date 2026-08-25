"""Reproducibility auditing: how reproducible is each paper, from its text?

Correctness is paramount. This is a deterministic, no-LLM text-scan heuristic over
``paper.text()`` (title + abstract), using **word-boundary** regexes so that, e.g.,
"github" is not matched inside an unrelated token. The audit scores six *weighted*
dimensions (a weighted mean, **not** an equal average) drawn from the FAIR /
open-science and methodological-rigor literature:

- ``data_availability``    (0.25): a data-availability statement or repository accession;
- ``code_availability``    (0.20): a code repository or code-availability statement;
- ``sample_size_adequacy`` (0.15): an *explicit* reported ``n`` that is also **>= 30**
  (presence of an ``n`` alone is deliberately not enough — that was an old-tool bug);
- ``statistical_rigor``    (0.15): p-values, confidence intervals, effect sizes, or
  multiple-comparison correction;
- ``validation_strategy``  (0.15): cross-validation, held-out / independent cohort, or
  explicit replication;
- ``controls``             (0.10): a control group, negative/positive control, or placebo.

Each dimension is binary (``1.0`` met / ``0.0`` not). The per-paper Score is built from
these six :class:`~consilium.models.ScoreComponent` weights via the shared honest-scoring
kernel, so it always carries its components, method, interpretation band and weights — no
bare magic number.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from consilium.models import EvidenceRef, Insight, ScoreComponent
from consilium.scoring.common import make_score

if TYPE_CHECKING:
    from consilium.models import Corpus, Paper

# Minimum reported sample size we treat as "adequate". Below this, an explicit n is
# *not* enough — a paper that only says "small sample (n=5)" must not score the point.
_MIN_ADEQUATE_N = 30

# --- Dimension keyword patterns (word-boundary, case-insensitive) ---------------------
# Each entry maps a dimension name to a compiled regex whose presence sets value=1.0.

_DATA_AVAILABILITY = re.compile(
    r"\b(?:"
    r"data\s+(?:availability|available|access(?:ion)?|deposited)"
    r"|available\s+at"
    r"|deposited\s+(?:in|at)"
    r"|accession(?:\s+(?:number|code|id))?"
    r"|GEO|G(?:SE|SM|DS)\d+|SRA|SRR\d+|ArrayExpress|ENA|PRIDE|MetaboLights"
    r"|dbGaP|phs\d+|EGA[SD]\d+"
    r"|zenodo|figshare|dryad"
    r")\b",
    re.IGNORECASE,
)

_CODE_AVAILABILITY = re.compile(
    r"\b(?:"
    r"github|gitlab|bitbucket"
    r"|code\s+(?:availability|available|repository|repositories)"
    r"|source\s+code"
    r"|software\s+(?:availability|available)"
    r"|zenodo\s+(?:code|software|doi)"
    r")\b",
    re.IGNORECASE,
)

# An explicit sample size: "n=120", "N = 120", "n of 120". We capture the integer and
# require it to be >= _MIN_ADEQUATE_N. \b before n avoids matching the 'n' inside words.
_SAMPLE_SIZE = re.compile(
    r"\bn\b\s*(?:=|of|:)?\s*(\d+)",
    re.IGNORECASE,
)

_STATISTICAL_RIGOR = re.compile(
    r"(?:"
    r"\bp\s*[<>=]\s*0?\.\d+"  # p<0.05, p = .01, etc.
    r"|\bp[\s-]*values?\b"
    r"|\bconfidence\s+intervals?\b"
    r"|\bCI\b"  # 95% CI
    r"|\beffect\s+sizes?\b"
    r"|\b(?:cohen'?s\s+d|odds\s+ratio|hazard\s+ratio)\b"
    r"|\b(?:bonferroni|benjamini[\s-]*hochberg|fdr|holm)\b"
    r"|\bmultiple\s+(?:comparisons?|testing)\b"
    r"|\b(?:corrected|adjusted)\s+(?:for\s+)?multiple\b"
    r")",
    re.IGNORECASE,
)

_VALIDATION_STRATEGY = re.compile(
    r"\b(?:"
    r"cross[\s-]*validation"
    r"|\d+[\s-]*fold"
    r"|held[\s-]*out"
    r"|hold[\s-]*out"
    r"|independent\s+(?:cohort|dataset|test\s+set|validation|sample)"
    r"|external\s+validation"
    r"|replicat(?:ion|ions|ed|es|ing|e)"
    r"|test\s+set"
    r")\b",
    re.IGNORECASE,
)

_CONTROLS = re.compile(
    r"\b(?:"
    r"control\s+(?:group|condition|arm|cohort)"
    r"|(?:negative|positive)\s+controls?"
    r"|placebo"
    r"|sham\b"
    r"|vehicle\s+control"
    r"|controls?"
    r")\b",
    re.IGNORECASE,
)

# Dimension name -> weight. Order is the canonical reporting order. Weights sum to 1.0.
_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("data_availability", 0.25),
    ("code_availability", 0.20),
    ("sample_size_adequacy", 0.15),
    ("statistical_rigor", 0.15),
    ("validation_strategy", 0.15),
    ("controls", 0.10),
)


def _has_adequate_sample(text: str) -> bool:
    """True iff an explicit ``n`` >= :data:`_MIN_ADEQUATE_N` is reported.

    Presence of *any* reported n is intentionally not sufficient: a paper that only
    mentions a small sample (e.g. "n=5") must score 0 here. We scan every ``n=NN`` and
    require at least one to clear the threshold.
    """
    return any(int(m.group(1)) >= _MIN_ADEQUATE_N for m in _SAMPLE_SIZE.finditer(text))


def _dimension_values(text: str) -> dict[str, float]:
    """Binary value (0.0/1.0) for each scored dimension, from a single paper's text."""
    return {
        "data_availability": 1.0 if _DATA_AVAILABILITY.search(text) else 0.0,
        "code_availability": 1.0 if _CODE_AVAILABILITY.search(text) else 0.0,
        "sample_size_adequacy": 1.0 if _has_adequate_sample(text) else 0.0,
        "statistical_rigor": 1.0 if _STATISTICAL_RIGOR.search(text) else 0.0,
        "validation_strategy": 1.0 if _VALIDATION_STRATEGY.search(text) else 0.0,
        "controls": 1.0 if _CONTROLS.search(text) else 0.0,
    }


def _audit_paper(paper: Paper, text: str, source: str) -> Insight:
    """Score one paper's reproducibility into an :class:`Insight`."""
    values = _dimension_values(text)
    components = [
        ScoreComponent(name=name, value=values[name], weight=weight) for name, weight in _WEIGHTS
    ]
    score = make_score(components, method="weighted_mean")

    met = [name for name, _ in _WEIGHTS if values[name] >= 1.0]
    met_clause = ", ".join(met) if met else "none"
    caveat = "" if source == "full text" else " (abstract-only; may under-report.)"
    reasoning = (
        f"Deterministic text-scan heuristic over {source} (word-boundary regex; no LLM). "
        "Six weighted dimensions; sample_size_adequacy requires an explicit reported "
        f"n >= {_MIN_ADEQUATE_N}, not merely the presence of an n. "
        f"Dimensions met: {met_clause}.{caveat}"
    )
    return Insight(
        statement=f"Reproducibility of '{paper.title}' is {score.interpretation}.",
        evidence=[EvidenceRef(paper_id=paper.id)],
        is_synthesized=True,
        reasoning=reasoning,
        tags=["reproducibility"],
        score=score,
    )


def audit_reproducibility(
    corpus: Corpus, full_texts: dict[str, str] | None = None
) -> list[Insight]:
    """Audit each paper's reproducibility from its text.

    Returns one :class:`Insight` per paper (in corpus order), each carrying a six-component
    weighted :class:`~consilium.models.Score`. The weighting is **not** an equal mean:
    data availability (0.25) and code availability (0.20) dominate, followed by sample-size
    adequacy, statistical rigor and validation strategy (0.15 each) and controls (0.10).

    By default the scan runs over ``title + abstract``, which usually omits data/code
    availability statements and so UNDER-reports; pass ``full_texts``
    (``{paper_id: full text}``, e.g. fetched from Perspicacité) for a fair audit. Each
    Insight's ``reasoning`` discloses which source was used. Returns ``[]`` for an empty corpus.
    """
    texts = full_texts or {}
    if not corpus.papers:
        return []
    out: list[Insight] = []
    for paper in corpus.papers:
        full = texts.get(paper.id)
        out.append(
            _audit_paper(paper, full or paper.text(), "full text" if full else "title+abstract")
        )
    return out
