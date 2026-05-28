"""DeepSeek thinking-mode content/reasoning short-circuit tests for track_a.

``track_a/llm.py`` uses the openai SDK pointed at OpenRouter. Some
reasoning-mode models (DeepSeek-v4-pro in particular) put the visible
output in ``message.reasoning`` rather than ``message.content``. The
unified short-circuit ``text = content or reasoning or ""`` mirrors the
daycare verifier idiom.

We mock the underlying ``OpenAI`` client so we never hit the network.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from unittest import mock

import pytest

from skill_evolve.track_a.llm import LLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_choice(*, content, reasoning=None) -> SimpleNamespace:
    """Build a SimpleNamespace shaped like OpenAI ChatCompletion choice."""
    msg = SimpleNamespace(content=content, reasoning=reasoning)
    return SimpleNamespace(message=msg)


def _fake_response(*, content, reasoning=None) -> SimpleNamespace:
    return SimpleNamespace(choices=[_fake_choice(content=content, reasoning=reasoning)])


def _build_client_with_fake(resp) -> LLMClient:
    """Construct a non-synthetic LLMClient with the openai backend stubbed."""
    # We bypass __post_init__'s openai import + key check by injecting the
    # stub client after construction.
    with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "dummy"}):
        with mock.patch("openai.OpenAI") as m_openai:
            fake_client = mock.MagicMock()
            fake_client.chat.completions.create.return_value = resp
            m_openai.return_value = fake_client
            client = LLMClient(synthetic=False)
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_content_present_returns_content() -> None:
    resp = _fake_response(content="answer", reasoning=None)
    client = _build_client_with_fake(resp)
    out = client.complete(system="sys", user="usr")
    assert out == "answer"


def test_content_none_reasoning_present_returns_reasoning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    resp = _fake_response(content=None, reasoning="thoughts")
    client = _build_client_with_fake(resp)
    with caplog.at_level(logging.WARNING, logger="skill_evolve.track_a.llm"):
        out = client.complete(system="sys", user="usr")
    assert out == "thoughts"
    # Warning emitted noting the fallback.
    assert any("reasoning" in rec.message.lower() for rec in caplog.records)


def test_both_empty_returns_empty_string() -> None:
    resp = _fake_response(content="", reasoning="")
    client = _build_client_with_fake(resp)
    out = client.complete(system="sys", user="usr")
    assert out == ""


def test_dict_shape_message_also_supported() -> None:
    """OpenRouter raw-dict shape (msg is a dict, not an SDK object) works."""
    # Build a response where message is a plain dict.
    msg_dict = {"content": None, "reasoning": "from-dict"}
    choice = SimpleNamespace(message=msg_dict)
    resp = SimpleNamespace(choices=[choice])
    client = _build_client_with_fake(resp)
    out = client.complete(system="sys", user="usr")
    assert out == "from-dict"
