"""Tests for the prefill marquee helpers in ``swival._msg``."""

import types

from swival._msg import (
    _MARQUEE_PIECE_BUDGET,
    _marquee_text_for_turn,
    _trim_for_marquee,
)


class TestTrimForMarquee:
    def test_short_text_unchanged(self):
        assert _trim_for_marquee("hello") == "hello"

    def test_exact_budget_unchanged(self):
        text = "a" * _MARQUEE_PIECE_BUDGET
        assert _trim_for_marquee(text) == text

    def test_one_over_budget_truncates(self):
        text = "a" * (_MARQUEE_PIECE_BUDGET + 1)
        result = _trim_for_marquee(text)
        assert len(result) == _MARQUEE_PIECE_BUDGET
        assert result.endswith("…")
        assert result[:-1] == "a" * (_MARQUEE_PIECE_BUDGET - 1)

    def test_custom_budget(self):
        assert _trim_for_marquee("abcdef", budget=4) == "abc…"

    def test_preserves_beginning(self):
        text = "BEGINNING" + ("x" * _MARQUEE_PIECE_BUDGET) + "END"
        result = _trim_for_marquee(text)
        assert result.startswith("BEGINNING")
        assert "END" not in result

    def test_budget_zero_returns_empty(self):
        assert _trim_for_marquee("hello", budget=0) == ""

    def test_negative_budget_returns_empty(self):
        assert _trim_for_marquee("hello", budget=-5) == ""

    def test_budget_one_returns_just_ellipsis(self):
        assert _trim_for_marquee("hello", budget=1) == "…"

    def test_budget_one_short_text_unchanged(self):
        assert _trim_for_marquee("a", budget=1) == "a"


class TestMarqueeTextForTurn:
    def test_empty_messages_returns_none(self):
        assert _marquee_text_for_turn([]) is None

    def test_tail_ending_in_assistant_returns_none(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
        assert _marquee_text_for_turn(messages) is None

    def test_single_user_tail(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "hello world"},
        ]
        assert _marquee_text_for_turn(messages) == "hello world"

    def test_tool_tail_uses_name_prefix(self):
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "let me check"},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "name": "read_file",
                "content": "file contents",
            },
        ]
        assert _marquee_text_for_turn(messages) == "read_file: file contents"

    def test_multiple_tool_tail_joined_with_separator(self):
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "checking"},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "name": "tool_a",
                "content": "result a",
            },
            {
                "role": "tool",
                "tool_call_id": "c2",
                "name": "tool_b",
                "content": "result b",
            },
        ]
        result = _marquee_text_for_turn(messages)
        assert result == "tool_a: result a   ·   tool_b: result b"

    def test_tool_without_name_falls_back(self):
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]
        assert _marquee_text_for_turn(messages) == "tool: result"

    def test_all_blank_tail_returns_none(self):
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "tool_call_id": "c1", "name": "t", "content": ""},
            {"role": "user", "content": "   "},
        ]
        assert _marquee_text_for_turn(messages) is None

    def test_long_content_truncated_per_piece(self):
        big = "x" * (_MARQUEE_PIECE_BUDGET * 2)
        messages = [
            {"role": "assistant", "content": "a"},
            {"role": "tool", "tool_call_id": "c1", "name": "big_tool", "content": big},
        ]
        result = _marquee_text_for_turn(messages)
        assert result is not None
        assert result.endswith("…")
        assert "big_tool: " in result
        assert len(result) < _MARQUEE_PIECE_BUDGET + 32

    def test_stops_at_system_boundary(self):
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "only user"},
        ]
        assert _marquee_text_for_turn(messages) == "only user"

    def test_namespace_message_shape(self):
        messages = [
            types.SimpleNamespace(role="assistant", content="a"),
            types.SimpleNamespace(
                role="tool",
                tool_call_id="c1",
                name="ns_tool",
                content="ns result",
            ),
        ]
        assert _marquee_text_for_turn(messages) == "ns_tool: ns result"

    def test_mixed_dict_and_namespace(self):
        messages = [
            {"role": "assistant", "content": "a"},
            types.SimpleNamespace(
                role="tool", tool_call_id="c1", name="t1", content="r1"
            ),
            {"role": "tool", "tool_call_id": "c2", "name": "t2", "content": "r2"},
        ]
        assert _marquee_text_for_turn(messages) == "t1: r1   ·   t2: r2"

    def test_synthetic_user_included(self):
        messages = [
            {"role": "assistant", "content": "a"},
            {
                "role": "user",
                "content": "Reminder: do not forget tools",
                "_swival_synthetic": True,
            },
        ]
        assert _marquee_text_for_turn(messages) == "Reminder: do not forget tools"

    def test_assistant_in_middle_stops_walk(self):
        messages = [
            {"role": "user", "content": "old user"},
            {"role": "assistant", "content": "old assistant"},
            {"role": "user", "content": "newer user"},
        ]
        assert _marquee_text_for_turn(messages) == "newer user"

    def test_list_content_flattens(self):
        messages = [
            {"role": "assistant", "content": "a"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"},
                ],
            },
        ]
        assert _marquee_text_for_turn(messages) == "part one part two"
