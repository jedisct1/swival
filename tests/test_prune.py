"""Tests for swival/prune.py — transcript pruning for token savings."""

import types

import pytest

from swival.prune import (
    RECAP_PREFIXES,
    REPAIR_FEEDBACK_SENTINEL,
    PruneMetrics,
    _build_think_recap,
    _build_todo_recap,
    _build_snapshot_recap,
    _canonicalize_tool_calls,
    _fold_state_turns,
    _gc_synthetic_messages,
    _is_recap_message,
    _strip_repair_feedback,
    prune_transcript_for_llm,
)
from swival.thinking import ThinkingState
from swival.todo import TodoState
from swival.snapshot import SnapshotState


# -- Helpers -----------------------------------------------------------------


def _user(content):
    return {"role": "user", "content": content}


def _assistant(content):
    return {"role": "assistant", "content": content}


def _tool(tool_call_id, content):
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _tc(tc_id, name, arguments="{}"):
    return {
        "id": tc_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _assistant_tc(tool_calls, content=""):
    return {"role": "assistant", "content": content, "tool_calls": tool_calls}


def _system(content="You are helpful."):
    return {"role": "system", "content": content}


# -- Rule 1: Synthetic Message GC -------------------------------------------


def _synthetic_user(content):
    """Create a synthetic user message with the _swival_synthetic marker."""
    return {"role": "user", "content": content, "_swival_synthetic": True}


class TestSyntheticGC:
    def test_removes_marked_synthetic_after_assistant_reply(self):
        msgs = [
            _system(),
            _user("do something"),
            _assistant(""),
            _synthetic_user(
                "Your response was empty. Please continue working on the task using the available tools."
            ),
            _assistant("Here is the answer."),
        ]
        saved, mutated = _gc_synthetic_messages(msgs)
        assert saved > 0
        assert mutated
        assert not any(
            "Your response was empty" in (m.get("content", "") or "")
            for m in msgs
            if m.get("role") == "user"
        )

    def test_keeps_synthetic_without_later_assistant(self):
        msgs = [
            _system(),
            _user("do something"),
            _assistant(""),
            _synthetic_user("Your response was empty. Please answer."),
        ]
        saved, mutated = _gc_synthetic_messages(msgs)
        assert saved == 0
        assert not mutated
        assert len(msgs) == 4

    def test_removes_marked_guardrail(self):
        msgs = [
            _system(),
            _user("fix it"),
            _synthetic_user(
                "IMPORTANT: You have called `edit_file` 2 times with the same error."
            ),
            _assistant("OK, fixed."),
        ]
        saved, mutated = _gc_synthetic_messages(msgs)
        assert saved > 0
        assert mutated

    def test_preserves_real_user_message_with_important_prefix(self):
        """Real user messages starting with IMPORTANT: must NOT be removed."""
        msgs = [
            _system(),
            _user("IMPORTANT: do not modify package.json"),
            _assistant("OK."),
        ]
        _gc_synthetic_messages(msgs)
        assert any(
            "do not modify package.json" in m.get("content", "")
            for m in msgs
            if m.get("role") == "user"
        )

    def test_preserves_real_user_message_with_stop_prefix(self):
        msgs = [
            _system(),
            _user("STOP: don't delete that file"),
            _assistant("OK."),
        ]
        _gc_synthetic_messages(msgs)
        assert any("don't delete that file" in m.get("content", "") for m in msgs)

    def test_removes_tip_and_reminder_when_marked(self):
        msgs = [
            _system(),
            _synthetic_user("Tip: Consider using the `think` tool."),
            _synthetic_user("Reminder: You have 3 unfinished todo items."),
            _assistant("OK."),
        ]
        saved, mutated = _gc_synthetic_messages(msgs)
        assert saved > 0
        assert mutated
        assert len([m for m in msgs if m.get("role") == "user"]) == 0

    def test_preserves_real_user_messages(self):
        msgs = [
            _system(),
            _user("Please fix the bug in main.py"),
            _assistant("Done."),
        ]
        _gc_synthetic_messages(msgs)
        assert any("fix the bug" in m.get("content", "") for m in msgs)

    def test_removes_cut_off_nudge_when_marked(self):
        msgs = [
            _system(),
            _synthetic_user(
                "Your response was cut off. Please use the provided tools."
            ),
            _assistant("Continuing."),
        ]
        saved, mutated = _gc_synthetic_messages(msgs)
        assert saved > 0
        assert len([m for m in msgs if m.get("role") == "user"]) == 0

    def test_namespace_synthetic_messages_are_gc_eligible(self):
        """Namespace-style messages with _swival_synthetic are GC'd too."""
        ns_msg = types.SimpleNamespace(
            role="user",
            content="Reminder: You have 2 unfinished items.",
            _swival_synthetic=True,
        )
        msgs = [_system(), ns_msg, _assistant("OK.")]
        saved, mutated = _gc_synthetic_messages(msgs)
        assert saved > 0
        assert mutated

    def test_tool_call_only_response_counts_as_assistant(self):
        """A tool-calls-only assistant turn (no text) should expire synthetic messages."""
        msgs = [
            _system(),
            _synthetic_user("Tip: Consider using the `think` tool."),
            _assistant_tc([_tc("tc1", "read_file", '{"file_path":"x.py"}')]),
            _tool("tc1", "file contents"),
        ]
        saved, mutated = _gc_synthetic_messages(msgs)
        assert saved > 0
        assert mutated


# -- Rule 2: State-Tool Folding ---------------------------------------------


class TestStateFolding:
    def test_folds_think_turn_after_assistant_followup(self):
        msgs = [
            _system(),
            _user("analyze this"),
            _assistant_tc(
                [
                    _tc(
                        "tc1",
                        "think",
                        '{"thought":"step 1","thought_number":1,"total_thoughts":3,"next_thought_needed":true}',
                    )
                ]
            ),
            _tool("tc1", '{"thought_number":1}'),
            _assistant("Based on my analysis..."),
        ]
        ts = ThinkingState()
        ts.think_calls = 1
        ts.history = [
            types.SimpleNamespace(
                thought="step 1",
                thought_number=1,
                total_thoughts=3,
                next_thought_needed=True,
                is_revision=False,
                revises_thought=None,
                branch_from_thought=None,
                branch_id=None,
            )
        ]

        saved, mutated = _fold_state_turns(msgs, thinking_state=ts)
        assert saved > 0
        assert mutated
        # The think tool call turn should be gone
        assert not any(
            m.get("role") == "assistant" and m.get("tool_calls") for m in msgs
        )
        # A recap should be inserted
        assert any(
            m.get("content", "").startswith("[think state]")
            for m in msgs
            if m.get("role") == "user"
        )

    def test_folds_todo_turn(self):
        msgs = [
            _system(),
            _user("plan the work"),
            _assistant_tc([_tc("tc1", "todo", '{"action":"add","tasks":["task1"]}')]),
            _tool("tc1", '{"action":"add","total":1,"remaining":1}'),
            _assistant("Added task."),
        ]
        ts = TodoState()
        ts.items = [types.SimpleNamespace(text="task1", done=False)]
        ts._total_actions = 1
        ts.add_count = 1

        saved, mutated = _fold_state_turns(msgs, todo_state=ts)
        assert saved > 0
        assert mutated
        assert any(
            m.get("content", "").startswith("[todo state]")
            for m in msgs
            if m.get("role") == "user"
        )

    def test_skips_mixed_turn(self):
        msgs = [
            _system(),
            _user("do it"),
            _assistant_tc(
                [
                    _tc("tc1", "think", '{"thought":"plan"}'),
                    _tc("tc2", "edit_file", '{"file_path":"x.py"}'),
                ]
            ),
            _tool("tc1", '{"thought_number":1}'),
            _tool("tc2", "ok"),
            _assistant("Done."),
        ]
        ts = ThinkingState()
        saved, mutated = _fold_state_turns(msgs, thinking_state=ts)
        assert saved == 0

    def test_skips_turn_without_later_assistant(self):
        msgs = [
            _system(),
            _user("think about this"),
            _assistant_tc([_tc("tc1", "think", '{"thought":"hmm"}')]),
            _tool("tc1", '{"thought_number":1}'),
        ]
        ts = ThinkingState()
        saved, mutated = _fold_state_turns(msgs, thinking_state=ts)
        assert saved == 0

    def test_folds_when_later_assistant_is_tool_call_only(self):
        """An assistant turn with tool_calls but no text still counts as a response."""
        msgs = [
            _system(),
            _user("start"),
            _assistant_tc([_tc("tc1", "think", '{"thought":"plan"}')]),
            _tool("tc1", '{"thought_number":1}'),
            # Later assistant with tool_calls but no text content
            _assistant_tc([_tc("tc2", "read_file", '{"file_path":"x.py"}')]),
            _tool("tc2", "file contents"),
        ]
        ts = ThinkingState()
        ts.think_calls = 1
        ts.history = [
            types.SimpleNamespace(
                thought="plan",
                thought_number=1,
                total_thoughts=1,
                next_thought_needed=False,
                is_revision=False,
                revises_thought=None,
                branch_from_thought=None,
                branch_id=None,
            )
        ]
        saved, mutated = _fold_state_turns(msgs, thinking_state=ts)
        assert saved > 0
        assert mutated

    def test_preserves_snapshot_save_anchor(self):
        msgs = [
            _system(),
            _user("investigate"),
            _assistant_tc([_tc("tc1", "snapshot", '{"action":"save","label":"auth"}')]),
            _tool("tc1", '{"action":"save","status":"checkpoint_set"}'),
            _assistant("Investigating..."),
        ]
        ss = SnapshotState()
        ss.explicit_active = True
        ss.explicit_label = "auth"
        ss.explicit_begin_tool_call_id = "tc1"

        saved, mutated = _fold_state_turns(msgs, snapshot_state=ss)
        assert saved == 0
        assert not mutated
        # The snapshot save turn should still be there
        assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in msgs)

    def test_recap_placed_before_final_message(self):
        msgs = [
            _system(),
            _user("start"),
            _assistant_tc([_tc("tc1", "think", '{"thought":"x"}')]),
            _tool("tc1", "ok"),
            _assistant("middle"),
            _user("continue"),
        ]
        ts = ThinkingState()
        ts.think_calls = 1
        ts.history = [
            types.SimpleNamespace(
                thought="x",
                thought_number=1,
                total_thoughts=1,
                next_thought_needed=False,
                is_revision=False,
                revises_thought=None,
                branch_from_thought=None,
                branch_id=None,
            )
        ]

        _fold_state_turns(msgs, thinking_state=ts)
        # Recap should be right before the last message
        recap_indices = [
            i
            for i, m in enumerate(msgs)
            if m.get("content", "").startswith("[think state]")
        ]
        assert recap_indices
        assert recap_indices[0] == len(msgs) - 2  # before the last "continue"

    def test_stale_recap_cleanup(self):
        msgs = [
            _system(),
            _user("start"),
            _user("[think state]\nold recap"),  # stale recap
            _assistant_tc([_tc("tc1", "think", '{"thought":"new"}')]),
            _tool("tc1", "ok"),
            _assistant("result"),
        ]
        ts = ThinkingState()
        ts.think_calls = 2
        ts.history = [
            types.SimpleNamespace(
                thought="new",
                thought_number=2,
                total_thoughts=3,
                next_thought_needed=True,
                is_revision=False,
                revises_thought=None,
                branch_from_thought=None,
                branch_id=None,
            )
        ]

        _fold_state_turns(msgs, thinking_state=ts)
        # Should have exactly one think recap
        recaps = [m for m in msgs if m.get("content", "").startswith("[think state]")]
        assert len(recaps) == 1
        assert "new" in recaps[0]["content"]

    def test_recap_does_not_split_tool_call_turn(self):
        """Recaps must not land between an assistant(tool_calls) and its tool results."""
        msgs = [
            _system(),
            _user("start"),
            _assistant_tc([_tc("tc1", "think", '{"thought":"plan"}')]),
            _tool("tc1", '{"thought_number":1}'),
            _assistant("middle"),
            # Tail is a tool-call turn: assistant + tool results
            _assistant_tc([_tc("tc2", "read_file", '{"file_path":"x.py"}')]),
            _tool("tc2", "file contents here"),
        ]
        ts = ThinkingState()
        ts.think_calls = 1
        ts.history = [
            types.SimpleNamespace(
                thought="plan",
                thought_number=1,
                total_thoughts=2,
                next_thought_needed=True,
                is_revision=False,
                revises_thought=None,
                branch_from_thought=None,
                branch_id=None,
            )
        ]

        _fold_state_turns(msgs, thinking_state=ts)

        # Find the recap and the read_file tool-call turn
        for i, m in enumerate(msgs):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                tc_name = m["tool_calls"][0]["function"]["name"]
                if tc_name == "read_file":
                    # The next message must be the matching tool result, not a recap
                    assert i + 1 < len(msgs)
                    assert msgs[i + 1].get("role") == "tool"
                    assert msgs[i + 1].get("tool_call_id") == "tc2"
                    break
        else:
            pytest.fail("read_file tool-call turn not found")

    def test_messages_mutated_flag_on_recap_refresh(self):
        """Stale recap removal + re-emission should set messages_mutated even
        if net token savings are zero."""
        msgs = [
            _system(),
            _user("start"),
            _user("[think state]\nold recap"),  # stale recap
            _assistant("done"),
        ]
        ts = ThinkingState()
        ts.think_calls = 1
        ts.history = [
            types.SimpleNamespace(
                thought="x",
                thought_number=1,
                total_thoughts=1,
                next_thought_needed=False,
                is_revision=False,
                revises_thought=None,
                branch_from_thought=None,
                branch_id=None,
            )
        ]

        _, mutated = _fold_state_turns(msgs, thinking_state=ts)
        assert mutated


# -- Rule 3: Repair Feedback Expiry -----------------------------------------


class TestRepairFeedback:
    def test_strips_sentinel_from_old_tool_results(self):
        msgs = [
            _system(),
            _tool(
                "tc1", f"result1{REPAIR_FEEDBACK_SENTINEL}\n[Syntax correction] fix it"
            ),
            _tool(
                "tc2",
                f"result2{REPAIR_FEEDBACK_SENTINEL}\n[Syntax correction] fix it too",
            ),
            _tool("tc3", "result3"),
        ]
        saved = _strip_repair_feedback(msgs)
        assert saved > 0
        assert msgs[1]["content"] == "result1"
        # tc3 is within last 2, but tc2 is also within last 2
        # tc1 is the only one outside last 2

    def test_keeps_last_two_intact(self):
        msgs = [
            _system(),
            _tool("tc1", f"result1{REPAIR_FEEDBACK_SENTINEL}\nfeedback1"),
            _tool("tc2", f"result2{REPAIR_FEEDBACK_SENTINEL}\nfeedback2"),
            _tool("tc3", f"result3{REPAIR_FEEDBACK_SENTINEL}\nfeedback3"),
        ]
        _strip_repair_feedback(msgs)
        # tc2 and tc3 are last two tool messages, should keep feedback
        assert REPAIR_FEEDBACK_SENTINEL in msgs[2]["content"]
        assert REPAIR_FEEDBACK_SENTINEL in msgs[3]["content"]
        # tc1 should be stripped
        assert REPAIR_FEEDBACK_SENTINEL not in msgs[1]["content"]

    def test_no_sentinel_no_change(self):
        msgs = [
            _system(),
            _tool("tc1", "clean result"),
            _tool("tc2", "another clean result"),
            _tool("tc3", "third clean result"),
        ]
        saved = _strip_repair_feedback(msgs)
        assert saved == 0


# -- Rule 4: Tool-Call Canonicalization --------------------------------------


class TestToolCallCanonicalization:
    def test_strips_extra_fields(self):
        msgs = [
            _system(),
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"file_path":"x.py"}',
                        },
                        "index": 0,
                        "extra_field": "junk",
                    }
                ],
            },
            _tool("tc1", "file contents"),
            # A later tool-call turn so the first one is not the "last"
            _assistant_tc([_tc("tc2", "edit_file", '{"file_path":"x.py"}')]),
            _tool("tc2", "ok"),
        ]
        _canonicalize_tool_calls(msgs)
        tc = msgs[1]["tool_calls"][0]
        assert "index" not in tc
        assert "extra_field" not in tc
        assert tc["id"] == "tc1"
        assert tc["function"]["name"] == "read_file"

    def test_skips_last_tool_call_turn(self):
        msgs = [
            _system(),
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                        "index": 0,
                    }
                ],
            },
            _tool("tc1", "file contents"),
        ]
        _canonicalize_tool_calls(msgs)
        # Last tc turn — should still have index
        assert "index" in msgs[1]["tool_calls"][0]

    def test_converts_namespace_to_dict(self):
        tc_ns = types.SimpleNamespace(
            id="tc1",
            type="function",
            function=types.SimpleNamespace(name="think", arguments='{"thought":"x"}'),
        )
        msgs = [
            _system(),
            {"role": "assistant", "content": "", "tool_calls": [tc_ns]},
            _tool("tc1", "ok"),
            # A later tool-call turn so the first one is not the "last"
            _assistant_tc([_tc("tc2", "read_file", "{}")]),
            _tool("tc2", "content"),
        ]
        _canonicalize_tool_calls(msgs)
        tc = msgs[1]["tool_calls"][0]
        assert isinstance(tc, dict)
        assert tc["function"]["name"] == "think"


# -- Recap Builders ----------------------------------------------------------


class TestRecapBuilders:
    def test_think_recap(self):
        ts = ThinkingState()
        ts.think_calls = 3
        ts.history = [
            types.SimpleNamespace(
                thought="first thought",
                thought_number=1,
                total_thoughts=5,
                next_thought_needed=True,
                is_revision=False,
                revises_thought=None,
                branch_from_thought=None,
                branch_id=None,
            ),
            types.SimpleNamespace(
                thought="second thought",
                thought_number=2,
                total_thoughts=5,
                next_thought_needed=True,
                is_revision=False,
                revises_thought=None,
                branch_from_thought=None,
                branch_id="retry",
            ),
        ]
        ts.branches = {"retry": [ts.history[1]]}

        recap = _build_think_recap(ts)
        assert recap is not None
        assert recap.startswith("[think state]")
        assert "retry" in recap
        assert "#1" in recap
        assert "#2 [retry]" in recap

    def test_think_recap_none_when_empty(self):
        ts = ThinkingState()
        assert _build_think_recap(ts) is None

    def test_todo_recap(self):
        ts = TodoState()
        ts._total_actions = 2
        ts.add_count = 2
        ts.items = [
            types.SimpleNamespace(text="fix auth", done=True),
            types.SimpleNamespace(text="write tests", done=False),
        ]

        recap = _build_todo_recap(ts)
        assert recap is not None
        assert recap.startswith("[todo state]")
        assert "- [x] fix auth" in recap
        assert "- [ ] write tests" in recap
        assert "1 remaining" in recap

    def test_todo_recap_none_when_empty(self):
        ts = TodoState()
        assert _build_todo_recap(ts) is None

    def test_snapshot_recap_active(self):
        ss = SnapshotState()
        ss.explicit_active = True
        ss.explicit_label = "auth investigation"

        recap = _build_snapshot_recap(ss)
        assert recap is not None
        assert recap.startswith("[snapshot state]")
        assert "auth investigation" in recap
        assert "clean" in recap

    def test_snapshot_recap_dirty(self):
        ss = SnapshotState()
        ss.explicit_active = True
        ss.explicit_label = "test"
        ss.dirty_tools.add("edit_file")

        recap = _build_snapshot_recap(ss)
        assert "dirty" in recap

    def test_snapshot_recap_none_when_inactive(self):
        ss = SnapshotState()
        assert _build_snapshot_recap(ss) is None


# -- Recap Message Detection -------------------------------------------------


class TestRecapDetection:
    def test_detects_recap_prefixes(self):
        for prefix in RECAP_PREFIXES:
            msg = _user(f"{prefix}\nsome content")
            assert _is_recap_message(msg)

    def test_rejects_non_recap(self):
        assert not _is_recap_message(_user("hello"))
        assert not _is_recap_message(_assistant("[think state] not a recap"))


# -- End-to-End Pruning -----------------------------------------------------


class TestPruneTranscript:
    def test_full_pass(self):
        msgs = [
            _system(),
            _user("start"),
            _synthetic_user("Your response was empty. Please continue."),
            _assistant_tc(
                [
                    _tc(
                        "tc1",
                        "think",
                        '{"thought":"planning","thought_number":1,"total_thoughts":2,"next_thought_needed":true}',
                    )
                ]
            ),
            _tool("tc1", '{"thought_number":1}'),
            _assistant("Here is the plan."),
            _user("ok go"),
        ]
        ts = ThinkingState()
        ts.think_calls = 1
        ts.history = [
            types.SimpleNamespace(
                thought="planning",
                thought_number=1,
                total_thoughts=2,
                next_thought_needed=True,
                is_revision=False,
                revises_thought=None,
                branch_from_thought=None,
                branch_id=None,
            )
        ]

        metrics = prune_transcript_for_llm(
            msgs,
            thinking_state=ts,
        )
        assert metrics.net_savings > 0
        assert metrics.synthetic_gc > 0
        assert metrics.state_folding > 0
        assert metrics.messages_mutated

    def test_noop_on_short_transcript(self):
        msgs = [_system(), _user("hi")]
        metrics = prune_transcript_for_llm(msgs)
        assert metrics.net_savings == 0
        assert not metrics.messages_mutated

    def test_metrics_summary_format(self):
        m = PruneMetrics(tokens_before=100, tokens_after=74, synthetic_gc=26)
        s = m.summary()
        assert s is not None
        assert "26" in s
        assert "synthetic_gc" in s

    def test_metrics_summary_none_when_no_savings(self):
        m = PruneMetrics(tokens_before=100, tokens_after=100)
        assert m.summary() is None

    def test_messages_mutated_on_recap_refresh_no_net_savings(self):
        """Checkpoint invalidation must fire even when recap refresh has
        net_savings=0 (removed stale recap and re-emitted a new one)."""
        msgs = [
            _system(),
            _user("start"),
            _user("[think state]\nold recap data"),
            _assistant("working on it"),
            _user("continue"),
        ]
        ts = ThinkingState()
        ts.think_calls = 1
        ts.history = [
            types.SimpleNamespace(
                thought="x",
                thought_number=1,
                total_thoughts=1,
                next_thought_needed=False,
                is_revision=False,
                revises_thought=None,
                branch_from_thought=None,
                branch_id=None,
            )
        ]
        metrics = prune_transcript_for_llm(msgs, thinking_state=ts)
        assert metrics.messages_mutated
