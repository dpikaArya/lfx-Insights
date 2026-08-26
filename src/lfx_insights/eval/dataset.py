"""Eval dataset loaders and the TF-IDF candidate-pool builder.

Normalises three benchmark shapes into the frozen
:class:`~lfx_insights.eval.models.EvalCase` contract, auto-detected per record:

- **ScholarQABench** ``{id, subject, input, output, ctxs}`` â€” multi-paper long-form QA.
- **ExpertQA** (Malaviya et al., NAACL 2024) ``{question, metadata.field, answers{model->
  {answer_string, revised_answer_string, claims[{claim_string, evidence}]}}}`` â€” expert,
  multi-domain attributed QA.
- **LitSearch** (Ajith et al., EMNLP 2024) ``{query, corpusids, specificity, quality}`` â€”
  scientific literature-search *retrieval* (gold corpus ids â†’ intrinsic recall@k/nDCG).

Sources: a bundled fixture (``"bundled"`` / ``"expertqa-bundled"`` / ``"litsearch-bundled"``)
or a filesystem path to a ``.jsonl`` (one JSON object per line) or ``.json`` (a JSON array).
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from lfx_insights.errors import InsightsError
from lfx_insights.eval.models import EvalCase, RetrievedDoc
from lfx_insights.models import Paper

# Short answer labels that mark a record as a short-form (classification) task.
_SHORT_LABELS = frozenset({"true", "false", "yes", "no"})

# Bundled fixtures, addressable by name for offline demos and tests.
_BUNDLED = {
    "bundled": "scholarqa_sample.jsonl",
    "expertqa-bundled": "expertqa_sample.jsonl",
    "litsearch-bundled": "litsearch_sample.jsonl",
}
# Cap on evidence passages pulled into ExpertQA contexts (bounds prompt/pool size).
_EXPERTQA_MAX_CTXS = 25


def _load_records(source: str) -> list[dict[str, Any]]:
    """Read raw records from ``source`` (bundled fixture, ``.jsonl``, or ``.json``)."""
    if source in _BUNDLED:
        resource = files("lfx_insights.eval") / "fixtures" / _BUNDLED[source]
        return _parse_jsonl(resource.read_text(encoding="utf-8"))

    path = Path(source)
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(raw)
        if not isinstance(data, list):
            raise InsightsError(f"Expected a JSON array in {source!r}, got {type(data).__name__}")
        return data
    return _parse_jsonl(raw)


def _parse_jsonl(raw: str) -> list[dict[str, Any]]:
    """Parse one JSON object per non-blank line."""
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _build_ctxs(rec: dict[str, Any], case_id: str) -> list[RetrievedDoc]:
    """Build :class:`RetrievedDoc`s from a record's context list.

    A context carrying a ``corpusid`` (LitSearch) keeps it as the ``doc_id`` so
    retrieved papers can be matched against gold corpus ids; otherwise the doc is
    numbered ``<case_id>-D<i>``.
    """
    raw_ctxs = rec.get("ctxs") or rec.get("gold_ctx") or rec.get("docs") or []
    docs: list[RetrievedDoc] = []
    for i, c in enumerate(raw_ctxs):
        doc_id = str(c["corpusid"]) if c.get("corpusid") is not None else f"{case_id}-D{i}"
        docs.append(RetrievedDoc(doc_id=doc_id, title=c.get("title", ""), text=c.get("text", "")))
    return docs


def _detect_format(rec: dict[str, Any]) -> str:
    """Detect a record's benchmark shape from the fields present."""
    if "corpusids" in rec or rec.get("query_set") is not None:
        return "litsearch"
    if isinstance(rec.get("answers"), dict):
        return "expertqa"
    return "scholarqa"


def _normalize(rec: dict[str, Any], index: int) -> EvalCase:
    """Normalise one raw record into an :class:`EvalCase`, dispatching on its shape."""
    fmt = _detect_format(rec)
    if fmt == "litsearch":
        return _normalize_litsearch(rec, index)
    if fmt == "expertqa":
        return _normalize_expertqa(rec, index)
    return _normalize_scholarqa(rec, index)


def _normalize_scholarqa(rec: dict[str, Any], index: int) -> EvalCase:
    """Normalise one raw ScholarQA-shape record into an :class:`EvalCase`."""
    question = rec.get("input") or rec.get("question")
    if not question:
        raise InsightsError(f"Record {index} has no question (missing 'input'/'question')")

    case_id = rec.get("id") or rec.get("case_id") or f"case-{index}"
    subject = rec.get("subject")
    ctxs = _build_ctxs(rec, case_id)

    raw_answer = rec.get("output") or rec.get("answer")
    if isinstance(raw_answer, str) and raw_answer.strip().lower() in _SHORT_LABELS:
        return EvalCase(
            id=case_id,
            question=question,
            gold_label=raw_answer.strip().lower(),
            reference_answer=None,
            ctxs=ctxs,
            subject=subject,
            task="short_form",
            metrics=["match", "citation"],
        )

    reference_answer = raw_answer or None
    metrics = ["citation", "rouge", "quality"] if reference_answer else ["citation"]
    return EvalCase(
        id=case_id,
        question=question,
        reference_answer=reference_answer,
        ctxs=ctxs,
        subject=subject,
        task="long_form",
        metrics=metrics,
    )


def _coerce_evidence(item: Any) -> tuple[str, str]:
    """Coerce one ExpertQA evidence entry into ``(title, text)``.

    Entries are URL-with-passage dicts or bare strings, depending on the dump.
    """
    if isinstance(item, dict):
        title = str(item.get("url") or item.get("title") or "")
        text = str(item.get("text") or item.get("passage") or item.get("snippet") or "")
        return title, text
    return "", str(item)


def _expertqa_reference(answers: dict[str, Any]) -> str | None:
    """Pick a reference answer: the first human-revised string, else the first answer."""
    fallback: str | None = None
    for ans in answers.values():
        if not isinstance(ans, dict):
            continue
        revised = (ans.get("revised_answer_string") or "").strip()
        if revised:
            return revised
        if fallback is None and (ans.get("answer_string") or "").strip():
            fallback = ans["answer_string"].strip()
    return fallback


def _normalize_expertqa(rec: dict[str, Any], index: int) -> EvalCase:
    """Normalise one ExpertQA record into an :class:`EvalCase`.

    The reference is the human-revised answer; contexts are the evidence passages
    attached to the answers' claims (deduplicated, capped).
    """
    question = rec.get("question")
    if not question:
        raise InsightsError(f"ExpertQA record {index} has no 'question'")
    case_id = rec.get("id") or f"expertqa-{index}"
    meta = rec.get("metadata") or {}
    subject = meta.get("field") or meta.get("specific_field")
    answers = rec.get("answers") or {}

    docs: list[RetrievedDoc] = []
    seen: set[str] = set()
    for ans in answers.values():
        if not isinstance(ans, dict):
            continue
        for claim in ans.get("claims") or []:
            for item in (claim.get("evidence") or []) if isinstance(claim, dict) else []:
                title, text = _coerce_evidence(item)
                key = text.strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                docs.append(RetrievedDoc(doc_id=f"{case_id}-D{len(docs)}", title=title, text=text))
                if len(docs) >= _EXPERTQA_MAX_CTXS:
                    break

    reference_answer = _expertqa_reference(answers)
    metrics = ["citation", "rouge", "quality"] if reference_answer else ["citation"]
    return EvalCase(
        id=case_id,
        question=question,
        reference_answer=reference_answer,
        ctxs=docs,
        subject=subject,
        task="long_form",
        metrics=metrics,
    )


def _normalize_litsearch(rec: dict[str, Any], index: int) -> EvalCase:
    """Normalise one LitSearch query record into a retrieval :class:`EvalCase`.

    ``corpusids`` become the gold doc set for intrinsic recall@k/nDCG. Any attached
    ``ctxs`` (with ``corpusid``) form the candidate pool for the TF-IDF condition.
    """
    question = rec.get("query") or rec.get("question")
    if not question:
        raise InsightsError(f"LitSearch record {index} has no 'query'")
    case_id = rec.get("id") or f"litsearch-{index}"
    gold = [str(c) for c in rec.get("corpusids") or []]
    ctxs = _build_ctxs(rec, case_id)
    return EvalCase(
        id=case_id,
        question=question,
        gold_docs=gold,
        ctxs=ctxs,
        subject=rec.get("query_set"),
        task="retrieval",
        metrics=["retrieval"],
    )


def load_dataset(source: str) -> list[EvalCase]:
    """Load and normalise an eval dataset into :class:`EvalCase`s (format auto-detected).

    Args:
        source: a bundled fixture name (``"bundled"`` for ScholarQABench,
            ``"expertqa-bundled"``, ``"litsearch-bundled"``), or a filesystem path to a
            ``.jsonl`` (one JSON object per line) or ``.json`` (a JSON array) file. Each
            record's shape (ScholarQABench / ExpertQA / LitSearch) is detected per record.

    Returns:
        The normalised cases, in source order.

    Raises:
        InsightsError: If a record lacks its question/query, or a ``.json`` source is
            not a JSON array.
    """
    return [_normalize(rec, i) for i, rec in enumerate(_load_records(source))]


def candidate_pool(cases: list[EvalCase]) -> list[Paper]:
    """Build a deduplicated :class:`Paper` pool over all case contexts.

    Every unique context document (by ``doc_id``) becomes a Paper for the TF-IDF
    retrieval baseline, with its text carried in the abstract. First occurrence of
    a ``doc_id`` wins; later duplicates are skipped.

    Args:
        cases: The cases whose ``ctxs`` form the candidate pool.

    Returns:
        One Paper per unique context document, in first-seen order.
    """
    seen: set[str] = set()
    pool: list[Paper] = []
    for case in cases:
        for doc in case.ctxs:
            if doc.doc_id in seen:
                continue
            seen.add(doc.doc_id)
            pool.append(
                Paper(
                    id=doc.doc_id,
                    title=doc.title,
                    abstract=doc.text,
                    source="eval-pool",
                )
            )
    return pool
