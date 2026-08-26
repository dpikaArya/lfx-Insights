from __future__ import annotations

import pytest

from lfx_insights.config import Settings, load_settings
from lfx_insights.llm.client import MockLLM
from lfx_insights.sources.fake import FakeBackend


@pytest.fixture
def settings() -> Settings:
    return load_settings(None)


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()
