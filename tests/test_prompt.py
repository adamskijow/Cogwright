# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Tests for the shape of the prompt handed to the model."""

from __future__ import annotations

from cogwright.core.models import BlockKind, CodeRef, ScoredChunk
from cogwright.core.prompt import NOT_FOUND_MESSAGE, PromptBuilder

from .builders import chunk


def _scored() -> list[ScoredChunk]:
    a = chunk(
        "a1b2c3d4",
        text="1. Engage disconnect.\n2. Press start.",
        page=2,
        section="STARTUP PROCEDURE",
        kind=BlockKind.STEP,
    )
    b = chunk(
        "e5f6a7b8",
        text="Alarm 204 indicates low coolant.",
        page=3,
        section="ALARM REFERENCE",
        codes=(CodeRef("alarm", "AL-204", "Alarm 204"),),
    )
    return [
        ScoredChunk(chunk=a, score=1.4, match_type="hybrid"),
        ScoredChunk(chunk=b, score=0.8, match_type="semantic"),
    ]


def test_system_message_states_the_grounding_rules() -> None:
    messages = PromptBuilder().build("how do I start it", _scored())
    system = messages[0]

    assert system.role == "system"
    assert "ONLY" in system.content
    assert "numbered steps" in system.content
    assert NOT_FOUND_MESSAGE in system.content


def test_user_message_includes_question_and_every_passage() -> None:
    scored = _scored()
    messages = PromptBuilder().build("how do I start it", scored)
    user = messages[1]

    assert user.role == "user"
    assert "how do I start it" in user.content
    for sc in scored:
        assert f"[{sc.chunk.chunk_id}]" in user.content
        assert f"page {sc.chunk.page}" in user.content
        assert sc.chunk.section is not None
        assert sc.chunk.section in user.content
    # The identifier on the alarm passage is surfaced to the model.
    assert "AL-204" in user.content


def test_message_order_is_system_then_user() -> None:
    messages = PromptBuilder().build("q", _scored())
    assert [m.role for m in messages] == ["system", "user"]
