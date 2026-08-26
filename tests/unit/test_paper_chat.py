"""Tests for paper_chat module."""

from __future__ import annotations

import pytest

from lfx_insights.llm.client import MockLLM
from lfx_insights.projects.paper_chat import PaperChatEntry, PaperChatSession

pytestmark = pytest.mark.unit


class TestPaperChatEntryModel:
    def test_defaults(self) -> None:
        e = PaperChatEntry(
            entry_id="abc12345",
            paper_id="p1",
            question="What?",
            answer="This.",
            confidence=0.8,
            created_at="2025-01-01T00:00:00Z",
        )
        assert e.page is None
        assert e.section is None
        assert e.citations == []


class TestPaperChatSession:
    def test_ask_basic(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        session = PaperChatSession("paper_001", base)
        llm = MockLLM()
        entry = session.ask(
            "What is the main finding?",
            chunks=["The main finding is that gene X inhibits pathway Y."],
            llm=llm,
        )
        assert entry.paper_id == "paper_001"
        assert entry.question == "What is the main finding?"
        assert entry.answer
        assert entry.confidence > 0.0

    def test_ask_empty_chunks(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        session = PaperChatSession("p1", base)
        llm = MockLLM()
        entry = session.ask("Question?", chunks=[], llm=llm)
        assert entry.confidence == 0.1

    def test_history_persists(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        session1 = PaperChatSession("p1", base)
        llm = MockLLM()
        session1.ask("Q1", chunks=["context"], llm=llm)
        session1.ask("Q2", chunks=["context"], llm=llm)

        session2 = PaperChatSession("p1", base)
        assert len(session2.history) == 2

    def test_clear(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        session = PaperChatSession("p1", base)
        llm = MockLLM()
        session.ask("Q1", chunks=["c"], llm=llm)
        session.clear()
        assert len(session.history) == 0

    def test_get_history(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        session = PaperChatSession("p1", base)
        assert session.get_history() == []
