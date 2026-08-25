from __future__ import annotations

import pytest

from consilium.config import Settings, load_settings
from consilium.llm.client import MockLLM
from consilium.sources.fake import FakeBackend


@pytest.fixture
def settings() -> Settings:
    return load_settings(None)


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()
