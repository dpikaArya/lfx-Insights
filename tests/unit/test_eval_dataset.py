"""Unit tests for the ScholarQABench dataset loader and candidate-pool builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from consilium.errors import ConsiliumError
from consilium.eval.dataset import candidate_pool, load_dataset

pytestmark = pytest.mark.unit


def test_load_bundled_returns_five_long_form_cases() -> None:
    cases = load_dataset("bundled")
    assert len(cases) == 5
    for case in cases:
        assert case.task == "long_form"
        assert case.reference_answer is not None
        assert case.gold_label is None
        assert case.metrics == ["citation", "rouge", "quality"]
        assert case.ctxs, "bundled cases must carry contexts"


def test_bundled_doc_ids_are_case_scoped() -> None:
    cases = load_dataset("bundled")
    first = cases[0]
    assert first.id == "fix-cs-1"
    assert first.ctxs[0].doc_id == "fix-cs-1-D0"
    assert first.ctxs[0].title
    assert first.ctxs[0].text


def test_short_form_record(tmp_path: Path) -> None:
    path = tmp_path / "short.jsonl"
    rec = {
        "input": "Does aspirin reduce fever?",
        "answer": "yes",
        "ctxs": [{"title": "Aspirin", "text": "Aspirin reduces fever."}],
    }
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    cases = load_dataset(str(path))
    assert len(cases) == 1
    case = cases[0]
    assert case.task == "short_form"
    assert case.gold_label == "yes"
    assert case.reference_answer is None
    assert case.metrics == ["match", "citation"]
    assert case.id == "case-0"
    assert case.ctxs[0].doc_id == "case-0-D0"


def test_short_label_is_case_insensitive(tmp_path: Path) -> None:
    path = tmp_path / "short_upper.jsonl"
    path.write_text(
        json.dumps({"input": "Is it true?", "output": "True", "ctxs": []}) + "\n",
        encoding="utf-8",
    )
    case = load_dataset(str(path))[0]
    assert case.task == "short_form"
    assert case.gold_label == "true"


def test_long_form_without_reference(tmp_path: Path) -> None:
    path = tmp_path / "no_ref.jsonl"
    path.write_text(
        json.dumps({"input": "Open question with no reference?", "ctxs": []}) + "\n",
        encoding="utf-8",
    )
    case = load_dataset(str(path))[0]
    assert case.task == "long_form"
    assert case.reference_answer is None
    assert case.metrics == ["citation"]


def test_json_array_source(tmp_path: Path) -> None:
    path = tmp_path / "arr.json"
    payload = [
        {"id": "a", "input": "Q one?", "output": "A long-form answer.", "ctxs": []},
        {"input": "Q two?", "answer": "no", "ctxs": []},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    cases = load_dataset(str(path))
    assert [c.id for c in cases] == ["a", "case-1"]
    assert cases[0].task == "long_form"
    assert cases[1].task == "short_form"
    assert cases[1].gold_label == "no"


def test_alternate_field_names(tmp_path: Path) -> None:
    path = tmp_path / "alt.jsonl"
    rec = {
        "case_id": "alt-1",
        "question": "Asked via 'question' key?",
        "answer": "A reference long-form answer.",
        "subject": "physics",
        "gold_ctx": [{"title": "G", "text": "gold context"}],
    }
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    case = load_dataset(str(path))[0]
    assert case.id == "alt-1"
    assert case.question == "Asked via 'question' key?"
    assert case.subject == "physics"
    assert case.reference_answer == "A reference long-form answer."
    assert case.ctxs[0].doc_id == "alt-1-D0"
    assert case.ctxs[0].text == "gold context"


def test_record_with_no_question_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"output": "answer but no question", "ctxs": []}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ConsiliumError):
        load_dataset(str(path))


def test_json_array_source_must_be_array(tmp_path: Path) -> None:
    path = tmp_path / "obj.json"
    path.write_text(json.dumps({"input": "not an array"}), encoding="utf-8")
    with pytest.raises(ConsiliumError):
        load_dataset(str(path))


def test_candidate_pool_dedupes_and_sets_abstract() -> None:
    cases = load_dataset("bundled")
    total_ctxs = sum(len(c.ctxs) for c in cases)
    pool = candidate_pool(cases)

    assert len(pool) == total_ctxs
    ids = [p.id for p in pool]
    assert len(set(ids)) == len(ids), "doc_ids must be unique across the pool"
    for paper in pool:
        assert paper.source == "eval-pool"
        assert paper.abstract
        assert paper.text()  # title + abstract is non-empty


def test_candidate_pool_dedupes_shared_doc_ids() -> None:
    cases = load_dataset("bundled")
    # Re-using the same cases twice must not double the pool.
    pool = candidate_pool(cases + cases)
    assert len(pool) == sum(len(c.ctxs) for c in cases)


def test_load_expertqa_bundled() -> None:
    cases = load_dataset("expertqa-bundled")
    assert len(cases) == 3
    subjects = {c.subject for c in cases}
    assert subjects == {"Biology", "Computer Science", "Physics"}
    for c in cases:
        assert c.task == "long_form"
        assert c.reference_answer  # the human-revised answer
        assert c.metrics == ["citation", "rouge", "quality"]
        assert c.ctxs and all(d.text for d in c.ctxs)  # evidence passages


def test_load_litsearch_bundled() -> None:
    cases = load_dataset("litsearch-bundled")
    assert len(cases) == 3
    first = cases[0]
    assert first.task == "retrieval"
    assert first.metrics == ["retrieval"]
    assert first.gold_docs == ["101"]
    # ctxs keep the corpusid as the doc_id so retrieved papers match gold ids.
    assert any(d.doc_id == "101" for d in first.ctxs)


def test_format_autodetect(tmp_path: Path) -> None:
    # One file mixing all three shapes; each row detected independently.
    f = tmp_path / "mixed.jsonl"
    f.write_text(
        "\n".join(
            [
                '{"input": "scholar q", "output": "ans", "ctxs": []}',
                '{"question": "expert q", "metadata": {"field": "Bio"}, '
                '"answers": {"a": {"answer_string": "x", "claims": []}}}',
                '{"query": "lit q", "corpusids": [7], "query_set": "demo"}',
            ]
        ),
        encoding="utf-8",
    )
    cases = load_dataset(str(f))
    assert [c.task for c in cases] == ["long_form", "long_form", "retrieval"]
    assert cases[2].gold_docs == ["7"]


def test_expertqa_missing_question_raises(tmp_path: Path) -> None:
    f = tmp_path / "bad.jsonl"
    f.write_text('{"answers": {"a": {"answer_string": "x"}}, "metadata": {}}', encoding="utf-8")
    with pytest.raises(ConsiliumError):
        load_dataset(str(f))


def test_litsearch_missing_query_raises(tmp_path: Path) -> None:
    f = tmp_path / "bad.jsonl"
    f.write_text('{"corpusids": [1], "query_set": "demo"}', encoding="utf-8")
    with pytest.raises(ConsiliumError):
        load_dataset(str(f))
