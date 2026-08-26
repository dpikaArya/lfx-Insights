"""Lean pydantic working models.

These are lfx Insights *internal* representations. They are serialized to the
Holobiomics standards (indicium / ASTRA / asb-schema) at output boundaries by the
``lfx_insights.standards`` exporters — they deliberately do not subclass the generated
LinkML classes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Author(BaseModel):
    name: str
    affiliation: str | None = None
    orcid: str | None = None


class Paper(BaseModel):
    id: str
    title: str
    doi: str | None = None
    authors: list[Author] = Field(default_factory=list)
    year: int | None = None
    abstract: str | None = None
    source: str | None = None
    url: str | None = None

    def text(self) -> str:
        """Title + abstract, for embedding/keywording."""
        return f"{self.title}\n\n{self.abstract or ''}".strip()


class Passage(BaseModel):
    paper_id: str
    text: str
    location: str | None = None


class Corpus(BaseModel):
    kb_id: str
    papers: list[Paper] = Field(default_factory=list)

    def by_id(self, paper_id: str) -> Paper | None:
        return next((p for p in self.papers if p.id == paper_id), None)

    def dois(self) -> list[str]:
        return [p.doi for p in self.papers if p.doi]

    def __len__(self) -> int:
        return len(self.papers)


class ScoreComponent(BaseModel):
    name: str
    value: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0)


class Score(BaseModel):
    """A score that is honest about how it was produced.

    No bare magic numbers: a Score always carries its components, the combination
    ``method``, an ``interpretation`` band, and (optionally) an ``uncertainty``.
    """

    value: float = Field(ge=0.0, le=1.0)
    components: list[ScoreComponent] = Field(default_factory=list, validate_default=True)
    method: str
    interpretation: str
    uncertainty: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("components")
    @classmethod
    def _components_present(cls, v: list[ScoreComponent]) -> list[ScoreComponent]:
        # A composite score with no components is exactly the magic-number we ban.
        if not v:
            raise ValueError(
                "a Score must expose its components; "
                "a component-less composite is the magic number we ban"
            )
        return v


class Theme(BaseModel):
    id: int
    label: str = ""
    paper_ids: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    rationale: str = ""

    def size(self) -> int:
        return len(self.paper_ids)


class Provenance(BaseModel):
    generated_by: str = "lfx-insights"
    model: str | None = None
    git_commit: str | None = None
    artifact_path: str | None = None


class EvidenceRef(BaseModel):
    """A reference to supporting evidence in the corpus (anchored to a quote)."""

    paper_id: str
    quote: str | None = None
    location: str | None = None


class Insight(BaseModel):
    """A finding, mappable to an ASTRA Insight."""

    statement: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    is_synthesized: bool = False
    conditions: str | None = None
    reasoning: str | None = None
    tags: list[str] = Field(default_factory=list)
    score: Score | None = None
    provenance: Provenance = Field(default_factory=Provenance)


# The 20-term Bucur et al. (K-CAP 2021) SuperPattern qualifier vocabulary,
# mirroring indicium's SuperPatternQualifier enum (used when exporting a
# Hypothesis to an indicium draft Claim).
SUPERPATTERN_QUALIFIERS: tuple[str, ...] = (
    "causes",
    "prevents",
    "inhibits",
    "activates",
    "increases",
    "decreases",
    "correlates_with",
    "is_associated_with",
    "predicts",
    "interacts_with",
    "produces",
    "requires",
    "enables",
    "treats",
    "enhances",
    "reduces",
    "has_property",
    "is_part_of",
    "is_a",
    "consistent_with",
)


# Common natural-language qualifiers mapped onto the closest Bucur SuperPattern
# term, so a Hypothesis built from free-text relations doesn't silently collapse
# every unknown verb to the weak ``is_associated_with`` default.
QUALIFIER_SYNONYMS: dict[str, str] = {
    "upregulates": "increases",
    "up-regulates": "increases",
    "up_regulates": "increases",
    "upregulate": "increases",
    "raises": "increases",
    "elevates": "increases",
    "downregulates": "decreases",
    "down-regulates": "decreases",
    "down_regulates": "decreases",
    "downregulate": "decreases",
    "suppresses": "inhibits",
    "suppress": "inhibits",
    "represses": "inhibits",
    "blocks": "inhibits",
    "lowers": "decreases",
    "modulates": "is_associated_with",
    "modulate": "is_associated_with",
    "regulates": "is_associated_with",
    "affects": "is_associated_with",
    "induces": "causes",
    "induce": "causes",
    "triggers": "causes",
    "leads_to": "causes",
    "associated_with": "is_associated_with",
    "correlated_with": "correlates_with",
    "correlates": "correlates_with",
    "interacts": "interacts_with",
    "stimulates": "activates",
    "boosts": "enhances",
}


class Hypothesis(BaseModel):
    """A testable hypothesis, exportable to an indicium draft Claim (Bucur 5-slot)."""

    subject: str
    qualifier: str
    object: str
    statement: str
    rationale: str = ""
    independent_var: str | None = None
    dependent_var: str | None = None
    methodology: str | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)
    score: Score | None = None
    status: str = "draft"
    provenance: Provenance = Field(default_factory=Provenance)

    @field_validator("qualifier")
    @classmethod
    def _normalize_qualifier(cls, v: str) -> str:
        norm = v.strip().lower().replace("-", "_").replace(" ", "_")
        if norm in SUPERPATTERN_QUALIFIERS:
            return norm
        if norm in QUALIFIER_SYNONYMS:
            return QUALIFIER_SYNONYMS[norm]
        return "is_associated_with"


class ResearchQuestion(BaseModel):
    question: str
    rationale: str = ""
    score: Score | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)


class GeneratedSection(BaseModel):
    """A drafted manuscript or grant section with verified in-corpus citations."""

    name: str
    text: str
    citations: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)


class SectionBundle(BaseModel):
    """A self-contained, serializable export unit: drafted sections plus the
    papers they cite (deduped, first-citation order). Consumed by docx export."""

    title: str
    sections: list[GeneratedSection] = Field(default_factory=list)
    references: list[Paper] = Field(default_factory=list)


class ReviewComment(BaseModel):
    severity: str
    section: str
    comment: str
    suggestion: str | None = None


class StatRecommendation(BaseModel):
    """A statistical recommendation with a correct, named method (STATO-aligned)."""

    design: str
    method: str
    stato_term: str | None = None
    effect_size: float | None = None
    alpha: float = 0.05
    power: float | None = None
    n_per_group: int | None = None
    total_n: int | None = None
    notes: str = ""


class Protocol(BaseModel):
    """A lab or bioinformatics protocol with ordered steps and a QC checklist."""

    name: str
    kind: str
    steps: list[str] = Field(default_factory=list)
    qc_checklist: list[str] = Field(default_factory=list)
    notes: str = ""
