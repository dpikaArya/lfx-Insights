"""Configuration: YAML + environment, via pydantic-settings.

Precedence (highest first): explicit init kwargs > environment > YAML file > defaults.
Environment variables use the ``LFX_INSIGHTS_`` prefix and ``__`` to nest, e.g.
``LFX_INSIGHTS_LLM__MODEL=ollama/qwen2.5-coder:7b``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

if TYPE_CHECKING:
    from pathlib import Path


class LLMSettings(BaseModel):
    model: str = "ollama/qwen2.5-coder:7b"
    fallback: list[str] = Field(default_factory=list)
    temperature: float = 0.2
    cache: bool = True
    mock: bool = False
    ollama_base_url: str = "http://localhost:11434"


class PerspicaciteSettings(BaseModel):
    transport: str = "mcp_http"
    url: str = "http://localhost:8002/mcp"
    timeout: int = 60


class EmbeddingSettings(BaseModel):
    # "tfidf" -> deterministic offline embedder; "all-MiniLM-L6-v2" or any
    # local sentence-transformers model -> local; "<provider>/<model>" with an
    # explicit provider prefix (openai/, cohere/, voyage/, etc.) -> hosted tier
    # via LiteLLM (needs the provider API key).  See lfx_insights.themes.discover.default_embedder.
    model: str = "all-MiniLM-L6-v2"


class EvalSettings(BaseModel):
    """ScholarQABench eval harness knobs (see lfx_insights.eval)."""

    # Retrieval conditions for the ablation, in report order.
    conditions: list[str] = Field(default_factory=lambda: ["null", "tfidf", "perspicacite"])
    # Entailment judge for citation precision/recall: "lexical" (deterministic,
    # offline default), "grounding" (indicium verify_quote gate), or "llm".
    judge: str = "lexical"
    # Token-overlap fraction above which the lexical entailer counts a passage as
    # supporting a sentence.
    lexical_threshold: float = 0.55
    # Max papers retrieved per question per condition.
    retrieval_k: int = 10
    # Cap cases evaluated (0 = all); the runner logs any cap as a caveat.
    max_cases: int = 0
    # Self-ground citations at generation time (drop misattributed + add uncited-but-
    # supported). CRAG/Self-RAG + ALCE-style. See lfx_insights.eval.answer._selfground.
    ground_generation: bool = False
    # Entailer used by the generation self-grounding pass: "lexical" (cheap, deterministic)
    # or "llm" (matches a semantic citation judge, avoids dropping good paraphrases).
    generation_judge: str = "lexical"


_SCORING_STAGES = [
    "themes",
    "evidence_strength",
    "novelty",
    "opportunity",
    "funding",
    "meta_analysis",
]

_LIFE_SCIENCE_STAGES = [
    *_SCORING_STAGES,
    "study_design",
    "bioinformatics",
    "reproducibility",
    "datasets",
]

_NEW_STAGES = [
    "self_evaluation",
    "learning",
    "gap_evolution",
]

# Aggregation stages run last (they read prior stages' artifacts from the run dir).
_AGGREGATION_STAGES = [
    "kb_snapshot",
    "explainability",
    "dashboard",
    "brief",
    "capsule",
    "project",
    "memory",
]


class PipelineSettings(BaseModel):
    quick: list[str] = Field(
        default_factory=lambda: [*_SCORING_STAGES, *_NEW_STAGES, *_AGGREGATION_STAGES]
    )
    life_science: list[str] = Field(
        default_factory=lambda: [*_LIFE_SCIENCE_STAGES, *_NEW_STAGES, *_AGGREGATION_STAGES]
    )
    # Fetch each paper's full text from the backend for the datasets/reproducibility
    # stages (better recall — accessions & data/code statements live in full text, not
    # abstracts). Off by default: it is N extra backend calls per run.
    full_text: bool = False


class FeaturesSettings(BaseModel):
    """Compact feature flags for research intelligence capabilities."""

    self_evaluation: bool = True
    self_learning: bool = True
    gap_evolution: bool = True
    paper_chat: bool = True
    paper_comparison: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LFX_INSIGHTS_",
        env_nested_delimiter="__",
        extra="ignore",
        yaml_file=None,
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    perspicacite: PerspicaciteSettings = Field(default_factory=PerspicaciteSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    eval: EvalSettings = Field(default_factory=EvalSettings)
    features: FeaturesSettings = Field(default_factory=FeaturesSettings)
    output_dir: str = "outputs"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # init > env > yaml (yaml is the lowest-priority source)
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls))


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings, optionally layering a YAML file under env + defaults."""
    Settings.model_config["yaml_file"] = str(path) if path is not None else None
    return Settings()
