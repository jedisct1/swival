"""Pre-flight transcript pruning for token savings.

Runs in-place on the live messages list every turn, after snapshot history
injection and before token estimation / the outbound LLM call.

The pass permanently shrinks the replayed suffix so later turns benefit too.
It never touches the first (system) message or tool schemas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ._msg import (
    _estimate_tokens,
    _msg_content,
    _msg_get,
    _msg_role,
    _msg_tool_call_id,
    _msg_tool_calls,
    _set_msg_content,
)

REPAIR_FEEDBACK_SENTINEL = "\n[swival:repair-feedback]"

RECAP_PREFIXES = ("[think state]", "[todo state]", "[snapshot state]")

_STATE_TOOLS = frozenset({"think", "todo", "snapshot"})


@dataclass
class PruneMetrics:
    tokens_before: int = 0
    tokens_after: int = 0
    synthetic_gc: int = 0
    state_folding: int = 0
    repair_feedback: int = 0
    tool_canonicalization: int = 0
    messages_mutated: bool = False

    @property
    def net_savings(self) -> int:
        return self.tokens_before - self.tokens_after

    def summary(self) -> str | None:
        if self.net_savings <= 0:
            return None
        parts = []
        if self.synthetic_gc:
            parts.append(f"synthetic_gc={self.synthetic_gc}")
        if self.state_folding:
            parts.append(f"state_folding={self.state_folding}")
        if self.repair_feedback:
            parts.append(f"repair_feedback={self.repair_feedback}")
        if self.tool_canonicalization:
            parts.append(f"tool_canon={self.tool_canonicalization}")
        detail = ", ".join(parts)
        return f"pruned ~{self.net_savings} tokens ({detail})"


def _msg_tokens(msg) -> int:
    content = _msg_content(msg)
    tcs = _msg_tool_calls(msg)
    extra = 0
    if tcs:
        for tc in tcs:
            fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
            name = fn.name if hasattr(fn, "name") else fn.get("name", "")
            args = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", "")
            extra += _estimate_tokens(name + (args or ""))
    return _estimate_tokens(content) + extra


def _has_later_assistant(messages: list, after_idx: int) -> bool:
    """True if there is a later assistant turn (text or tool-call)."""
    for i in range(after_idx + 1, len(messages)):
        if _msg_role(messages[i]) == "assistant":
            c = _msg_content(messages[i])
            if (c and c.strip()) or _msg_tool_calls(messages[i]):
                return True
    return False


def _tc_ids_for(msg) -> set[str]:
    """Extract the set of tool_call IDs from an assistant message."""
    ids: set[str] = set()
    tcs = _msg_tool_calls(msg)
    if tcs:
        for tc in tcs:
            tc_id = tc.id if hasattr(tc, "id") else tc.get("id", "")
            if tc_id:
                ids.add(tc_id)
    return ids


def _tool_names_in_turn(assistant_msg) -> set[str]:
    names = set()
    tcs = _msg_tool_calls(assistant_msg)
    if tcs:
        for tc in tcs:
            fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
            name = fn.name if hasattr(fn, "name") else fn.get("name", "")
            if name:
                names.add(name)
    return names


def _is_recap_message(msg) -> bool:
    if _msg_role(msg) != "user":
        return False
    content = _msg_content(msg)
    return content.startswith(RECAP_PREFIXES)


def _gc_synthetic_messages(messages: list) -> tuple[int, bool]:
    """Remove expired synthetic user messages.

    Only removes messages explicitly marked with ``_swival_synthetic: True``
    by the agent loop, so real user messages are never touched regardless of
    their content.

    Returns (estimated tokens saved, whether any messages were removed).
    """
    saved = 0
    to_remove = []

    for i, msg in enumerate(messages):
        if _msg_role(msg) != "user":
            continue
        if not _msg_get(msg, "_swival_synthetic"):
            continue

        if _has_later_assistant(messages, i):
            to_remove.append(i)
            saved += _msg_tokens(msg)

    for i in reversed(to_remove):
        messages.pop(i)

    return saved, bool(to_remove)


@dataclass
class _FoldCandidate:
    """An assistant+tool turn that is eligible for folding."""

    assistant_idx: int
    tool_indices: list[int] = field(default_factory=list)
    tool_names: set[str] = field(default_factory=set)
    tokens: int = 0


def _find_foldable_state_turns(
    messages: list,
    snapshot_state=None,
) -> list[_FoldCandidate]:
    """Identify pure state-tool turns that can be folded.

    A turn is foldable when it contains only think/todo/snapshot calls, there
    is at least one later assistant turn, and it is not the snapshot save that
    anchors an active explicit checkpoint.
    """
    candidates = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if _msg_role(msg) == "assistant" and _msg_tool_calls(msg):
            names = _tool_names_in_turn(msg)
            if names and names.issubset(_STATE_TOOLS):
                tc_ids = _tc_ids_for(msg)

                if (
                    snapshot_state is not None
                    and snapshot_state.explicit_active
                    and "snapshot" in names
                    and snapshot_state.explicit_begin_tool_call_id in tc_ids
                ):
                    i += 1
                    continue

                cand = _FoldCandidate(
                    assistant_idx=i, tool_names=names, tokens=_msg_tokens(msg)
                )
                j = i + 1
                while j < len(messages):
                    nxt = messages[j]
                    if _msg_role(nxt) == "tool" and _msg_tool_call_id(nxt) in tc_ids:
                        cand.tool_indices.append(j)
                        cand.tokens += _msg_tokens(nxt)
                        j += 1
                    else:
                        break

                last_idx = cand.tool_indices[-1] if cand.tool_indices else i
                if _has_later_assistant(messages, last_idx):
                    candidates.append(cand)

                i = j
                continue
        i += 1
    return candidates


def _build_think_recap(thinking_state) -> str | None:
    if thinking_state is None or thinking_state.think_calls == 0:
        return None
    lines = ["[think state]"]
    lines.append(f"thought {len(thinking_state.history)}/{thinking_state.think_calls}")
    if thinking_state.branches:
        branch_names = ", ".join(sorted(thinking_state.branches.keys()))
        lines.append(f"branches: {branch_names}")
    for entry in thinking_state.history[-4:]:
        text = entry.thought[:120].replace("\n", " ").strip()
        prefix = f"  #{entry.thought_number}"
        if entry.branch_id:
            prefix += f" [{entry.branch_id}]"
        lines.append(f"{prefix}: {text}")
    return "\n".join(lines)


def _build_todo_recap(todo_state) -> str | None:
    if todo_state is None or todo_state._total_actions == 0:
        return None
    lines = ["[todo state]"]
    for item in todo_state.items:
        marker = "x" if item.done else " "
        lines.append(f"- [{marker}] {item.text}")
    lines.append(f"{todo_state.remaining_count} remaining")
    return "\n".join(lines)


def _build_snapshot_recap(snapshot_state) -> str | None:
    if snapshot_state is None:
        return None
    if not snapshot_state.explicit_active and not snapshot_state.history:
        return None
    parts = ["[snapshot state]"]
    if snapshot_state.explicit_active:
        scope = "dirty" if snapshot_state.dirty else "clean"
        parts.append(
            f"active checkpoint: {snapshot_state.explicit_label} (scope: {scope})"
        )
    elif snapshot_state.history:
        last = snapshot_state.history[-1]
        parts.append(f"last collapse: {last.get('label', '?')}")
    return " ".join(parts)


def _safe_recap_insert_pos(messages: list) -> int:
    """Find a position near the tail where recap messages can be inserted
    without splitting an assistant+tool-result turn.

    Scans backward past any trailing tool-result messages to find the start
    of the tail turn, then returns that index so recaps land just before it.
    """
    n = len(messages)
    if n <= 1:
        return n

    i = n - 1
    while i > 0 and _msg_role(messages[i]) == "tool":
        i -= 1

    if i > 0 and _msg_role(messages[i]) == "assistant" and _msg_tool_calls(messages[i]):
        return i

    return max(i, 1)


def _remove_stale_recaps(messages: list) -> set[str]:
    """Remove existing recap messages. Returns the set of recap types removed."""
    removed_types: set[str] = set()
    to_remove = []
    for i, msg in enumerate(messages):
        if _is_recap_message(msg):
            content = _msg_content(msg)
            if content.startswith("[think state]"):
                removed_types.add("think")
            elif content.startswith("[todo state]"):
                removed_types.add("todo")
            elif content.startswith("[snapshot state]"):
                removed_types.add("snapshot")
            to_remove.append(i)
    for i in reversed(to_remove):
        messages.pop(i)
    return removed_types


def _fold_state_turns(
    messages: list,
    thinking_state=None,
    todo_state=None,
    snapshot_state=None,
) -> tuple[int, bool]:
    """Fold old state-tool turns and insert current-state recaps near the tail.

    Returns (estimated net tokens saved, whether messages were mutated).
    """
    candidates = _find_foldable_state_turns(messages, snapshot_state)
    if not candidates:
        removed = _remove_stale_recaps(messages)
        mutated = bool(removed)
        if removed:
            recaps = _emit_recaps(removed, thinking_state, todo_state, snapshot_state)
            if recaps:
                insert_pos = _safe_recap_insert_pos(messages)
                for recap in reversed(recaps):
                    messages.insert(insert_pos, recap)
        return 0, mutated

    tokens_removed = sum(c.tokens for c in candidates)

    folded_types: set[str] = set()
    for c in candidates:
        folded_types.update(c.tool_names)

    all_indices = []
    for c in candidates:
        all_indices.append(c.assistant_idx)
        all_indices.extend(c.tool_indices)
    for i in sorted(all_indices, reverse=True):
        messages.pop(i)

    removed_recap_types = _remove_stale_recaps(messages)
    recap_types = folded_types | removed_recap_types

    recaps = _emit_recaps(recap_types, thinking_state, todo_state, snapshot_state)

    tokens_added = 0
    if recaps:
        insert_pos = _safe_recap_insert_pos(messages)
        for recap in reversed(recaps):
            messages.insert(insert_pos, recap)
            tokens_added += _msg_tokens(recap)

    return tokens_removed - tokens_added, True


def _emit_recaps(
    types: set[str],
    thinking_state,
    todo_state,
    snapshot_state,
) -> list[dict]:
    """Build recap user messages for the requested tool types."""
    recaps = []
    if "think" in types:
        text = _build_think_recap(thinking_state)
        if text:
            recaps.append({"role": "user", "content": text})
    if "todo" in types:
        text = _build_todo_recap(todo_state)
        if text:
            recaps.append({"role": "user", "content": text})
    if "snapshot" in types:
        text = _build_snapshot_recap(snapshot_state)
        if text:
            recaps.append({"role": "user", "content": text})
    return recaps


def _strip_repair_feedback(messages: list) -> int:
    """Strip [swival:repair-feedback] suffixes from older tool results.

    Keeps the last 2 tool-result messages intact.
    Returns estimated tokens saved.
    """
    saved = 0
    tool_indices = [i for i, m in enumerate(messages) if _msg_role(m) == "tool"]
    if len(tool_indices) <= 2:
        return 0

    for i in tool_indices[:-2]:
        msg = messages[i]
        content = _msg_content(msg)
        idx = content.find(REPAIR_FEEDBACK_SENTINEL)
        if idx != -1:
            before = _estimate_tokens(content)
            new_content = content[:idx]
            _set_msg_content(msg, new_content)
            saved += before - _estimate_tokens(new_content)

    return saved


def _canonicalize_tool_calls(messages: list) -> int:
    """Rewrite historical assistant tool_calls to minimal shape.

    Strips provider extras (index, etc.) keeping only id, type,
    function.name, function.arguments. Skips the last assistant message
    with tool_calls.

    Returns estimated tokens saved.
    """
    saved = 0

    last_tc_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if _msg_role(messages[i]) == "assistant" and _msg_tool_calls(messages[i]):
            last_tc_idx = i
            break

    for i, msg in enumerate(messages):
        if i == last_tc_idx:
            continue
        if not isinstance(msg, dict):
            continue
        if _msg_role(msg) != "assistant":
            continue
        tcs = msg.get("tool_calls")
        if not tcs or not isinstance(tcs, list):
            continue

        new_tcs = []
        changed = False
        for tc in tcs:
            if isinstance(tc, dict):
                fn = tc.get("function", {})
                canonical = {
                    "id": tc.get("id", ""),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", ""),
                    },
                }
                if canonical != tc:
                    before = _estimate_tokens(json.dumps(tc))
                    after = _estimate_tokens(json.dumps(canonical))
                    saved += before - after
                    changed = True
                new_tcs.append(canonical)
            elif hasattr(tc, "function"):
                fn = tc.function
                fn_name = fn.name if hasattr(fn, "name") else ""
                fn_args = fn.arguments if hasattr(fn, "arguments") else ""
                canonical = {
                    "id": tc.id if hasattr(tc, "id") else "",
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": fn_args or "",
                    },
                }
                before = _estimate_tokens(fn_name + (fn_args or ""))
                after = _estimate_tokens(
                    canonical["function"]["name"] + canonical["function"]["arguments"]
                )
                changed = True
                new_tcs.append(canonical)
            else:
                new_tcs.append(tc)

        if changed:
            msg["tool_calls"] = new_tcs

    return saved


def prune_transcript_for_llm(
    messages: list,
    *,
    thinking_state=None,
    todo_state=None,
    snapshot_state=None,
    estimate_tokens_fn=None,
) -> PruneMetrics:
    """Run all pruning rules on the live message list.

    This mutates ``messages`` in place. Call after snapshot history injection
    and before token estimation.
    """
    if len(messages) <= 2:
        return PruneMetrics()

    est = estimate_tokens_fn or (
        lambda msgs, tools=None: sum(_msg_tokens(m) for m in msgs)
    )

    metrics = PruneMetrics()
    metrics.tokens_before = est(messages)

    mutated = False

    metrics.synthetic_gc, gc_mutated = _gc_synthetic_messages(messages)
    mutated = mutated or gc_mutated

    metrics.state_folding, fold_mutated = _fold_state_turns(
        messages, thinking_state, todo_state, snapshot_state
    )
    mutated = mutated or fold_mutated

    metrics.repair_feedback = _strip_repair_feedback(messages)
    metrics.tool_canonicalization = _canonicalize_tool_calls(messages)

    any_changes = (
        mutated
        or metrics.synthetic_gc
        or metrics.state_folding
        or metrics.repair_feedback
        or metrics.tool_canonicalization
    )
    metrics.tokens_after = est(messages) if any_changes else metrics.tokens_before
    metrics.messages_mutated = mutated

    return metrics
