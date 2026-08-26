"""Render an :class:`~consilium.eval.models.AblationReport` as Markdown."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lfx_insights.eval.models import AblationReport, ConditionReport
    from lfx_insights.models import Score


def _cell(score: Score | None) -> str:
    if score is None:
        return "â€”"
    return f"{score.value:.3f} ({score.interpretation})"


def _condition_row(report: ConditionReport) -> str:
    return (
        f"| {report.condition} | {report.n_cases} | {_cell(report.citation)} | "
        f"{_cell(report.correctness)} | {_cell(report.quality)} | {_cell(report.retrieval)} |"
    )


def render_markdown(report: AblationReport) -> str:
    """Render the ablation as a Markdown report (table + lift + caveats)."""
    lines: list[str] = [
        f"# ScholarQABench ablation â€” {report.dataset}",
        "",
        "PerspicacitÃ©â†’Consilium pipeline: retrieve â†’ synthesise grounded answer â†’ score.",
        "",
        "| condition | cases | citation F1 | correctness | quality | retrieval |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(_condition_row(c) for c in report.conditions)
    lines.append("")

    if report.lift:
        lines.append("## Lift (perspicacite - tfidf)")
        lines.append("")
        lines.extend(f"- **{k}**: {v:+.4f}" for k, v in report.lift.items())
        lines.append("")

    caveats = list(report.caveats)
    for cond in report.conditions:
        caveats.extend(f"[{cond.condition}] {c}" for c in cond.caveats)
    if caveats:
        lines.append("## Caveats")
        lines.append("")
        lines.extend(f"- {c}" for c in dict.fromkeys(caveats))  # de-dupe, keep order
        lines.append("")

    lines.append(
        f"_Aggregates are honest Scores (components/weights/uncertainty). "
        f"Model: {report.provenance.model or 'n/a'}._"
    )
    return "\n".join(lines)
