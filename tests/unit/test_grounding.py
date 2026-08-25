from __future__ import annotations

import pytest

from consilium.errors import GroundingError
from consilium.standards.grounding import (
    _normalize,
    require_grounded,
    text_quote_selector,
    verify_quote_in,
)

pytestmark = pytest.mark.unit

SOURCE = "Graph neural networks predict molecular properties for drug discovery."


def test_text_quote_selector_found() -> None:
    sel = text_quote_selector("molecular properties", SOURCE)
    assert sel["exact"] == "molecular properties"
    assert "predict " in sel["prefix"]
    assert sel["suffix"].startswith(" for")


def test_verify_quote_in_true_false() -> None:
    assert verify_quote_in("molecular properties", [SOURCE]) is True
    assert verify_quote_in("quantum teleportation of cats", [SOURCE]) is False
    assert verify_quote_in("", [SOURCE]) is False


def test_require_grounded_raises() -> None:
    require_grounded("drug discovery", [SOURCE])  # no raise
    with pytest.raises(GroundingError):
        require_grounded("fabricated nonsense claim", [SOURCE])


def test_normalize_is_case_sensitive() -> None:
    # _normalize must NOT lowercase: the fallback containment has to mirror
    # indicium.verify_quote (case-sensitive) so strictness does not depend on
    # whether indicium is installed.
    assert _normalize("Molecular  Properties") == "Molecular Properties"
    assert _normalize("  Drug   Discovery  ") == "Drug Discovery"
    # Differing case yields differing normalized forms (no containment match).
    nq = _normalize("MOLECULAR PROPERTIES")
    assert nq not in _normalize(SOURCE)


def test_verify_quote_in_fallback_is_case_sensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exercise the no-indicium fallback directly so the case-sensitivity parity
    # is asserted even in environments where indicium is installed. Force the
    # ImportError branch by making the in-function ``import indicium.verify`` fail.
    import builtins
    from collections.abc import Mapping, Sequence
    from types import ModuleType

    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name.startswith("indicium"):
            raise ImportError("indicium not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Exact case matches in the fallback.
    assert verify_quote_in("molecular properties", [SOURCE]) is True
    # Wrong case does NOT match in the fallback (case-sensitive parity).
    assert verify_quote_in("MOLECULAR PROPERTIES", [SOURCE]) is False
