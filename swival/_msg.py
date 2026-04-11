"""Message accessor helpers for dict-or-namespace messages."""

# Image token budget: worst-case high-detail (85 base + 170 * 16 tiles = 2805)
IMAGE_TOKEN_ESTIMATE = 2805


def _msg_get(msg, key, default=None):
    return (
        msg.get(key, default) if isinstance(msg, dict) else getattr(msg, key, default)
    )


def _msg_role(msg) -> str | None:
    return _msg_get(msg, "role")


def _msg_content(msg) -> str:
    c = _msg_get(msg, "content", "")
    if isinstance(c, list):
        return " ".join(
            part.get("text", "")
            for part in c
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return c or ""


def _msg_tool_calls(msg):
    return _msg_get(msg, "tool_calls")


def _msg_tool_call_id(msg) -> str | None:
    return _msg_get(msg, "tool_call_id")


def _msg_name(msg) -> str:
    return _msg_get(msg, "name") or ""


def _set_msg_content(msg, value: str) -> None:
    if isinstance(msg, dict):
        msg["content"] = value
    else:
        msg.content = value


def _estimate_tokens(text: str) -> int:
    """Rough token estimate without importing tiktoken."""
    return len(text) // 4


def _canonicalize_tool_calls(messages: list) -> None:
    """Rewrite historical assistant tool_calls to minimal shape.

    Strips provider extras (index, etc.) keeping only id, type,
    function.name, function.arguments.  Skips the most recent assistant
    message with tool_calls so in-flight calls are untouched.
    """
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
                    changed = True
                new_tcs.append(canonical)
            elif hasattr(tc, "function"):
                fn = tc.function
                canonical = {
                    "id": tc.id if hasattr(tc, "id") else "",
                    "type": "function",
                    "function": {
                        "name": fn.name if hasattr(fn, "name") else "",
                        "arguments": (fn.arguments if hasattr(fn, "arguments") else "")
                        or "",
                    },
                }
                changed = True
                new_tcs.append(canonical)
            else:
                new_tcs.append(tc)

        if changed:
            msg["tool_calls"] = new_tcs


def _has_image_content(messages: list) -> bool:
    """Check if any message contains image_url parts."""
    return any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for msg in messages
        if isinstance(msg, dict) and isinstance(msg.get("content"), list)
        for part in msg["content"]
    )
