"""Tests for swival.a2a_server: A2aServer, A2aTask, build_agent_card."""

import time
import uuid

import pytest

from swival.a2a_server import A2aServer, A2aTask, build_agent_card
from swival.session import Result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jsonrpc(method, params, req_id=1):
    """Build a JSON-RPC 2.0 request."""
    return {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}


def _send_message(client, message, context_id=None, task_id=None):
    """Send a SendMessage JSON-RPC request."""
    msg = {"role": "user", "parts": [{"type": "text", "text": message}]}
    if context_id:
        msg["contextId"] = context_id
    if task_id:
        msg["taskId"] = task_id
    params = {"message": msg}
    return client.post("/", json=_jsonrpc("SendMessage", params))


def _get_task(client, task_id, req_id=1):
    """Send a GetTask JSON-RPC request."""
    return client.post("/", json=_jsonrpc("GetTask", {"id": task_id}, req_id))


def _list_tasks(client, context_id=None, req_id=1):
    """Send a ListTasks JSON-RPC request."""
    params = {}
    if context_id:
        params["contextId"] = context_id
    return client.post("/", json=_jsonrpc("ListTasks", params, req_id))


def _make_result(answer="Hello from the agent", exhausted=False):
    """Build a canned Result for mocking Session.ask()."""
    return Result(
        answer=answer,
        exhausted=exhausted,
        messages=[
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": answer},
        ],
        report=None,
    )


def _make_input_required_result():
    """Build a Result that signals input-required (exhausted with no answer)."""
    return Result(
        answer=None,
        exhausted=True,
        messages=[{"role": "user", "content": "test"}],
        report=None,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _patch_session(monkeypatch):
    """Patch Session._setup to no-op and Session.ask to return a canned result."""
    from swival import session as session_mod

    monkeypatch.setattr(session_mod.Session, "_setup", lambda self: None)
    monkeypatch.setattr(
        session_mod.Session,
        "ask",
        lambda self, q: _make_result(f"answer to: {q}"),
    )


@pytest.fixture()
def server(_patch_session):
    """Create an A2aServer with mocked Session internals."""
    return A2aServer(
        session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
        host="127.0.0.1",
        port=0,
    )


@pytest.fixture()
def client(server):
    """Starlette TestClient wrapping the server's ASGI app."""
    from starlette.testclient import TestClient

    return TestClient(server.app)


# ---------------------------------------------------------------------------
# 1. Single-shot SendMessage
# ---------------------------------------------------------------------------


class TestSendMessageSingleShot:
    def test_creates_task_and_returns_completed(self, client):
        resp = _send_message(client, "Hello")
        assert resp.status_code == 200
        body = resp.json()
        assert "result" in body
        task = body["result"]
        assert task["status"]["state"] == "completed"
        assert task["id"]
        assert task["contextId"]

    def test_response_contains_agent_text(self, client):
        resp = _send_message(client, "Hello")
        task = resp.json()["result"]
        # The answer should appear somewhere in artifacts or status message
        texts = []
        for artifact in task.get("artifacts", []):
            for part in artifact.get("parts", []):
                if part.get("type") == "text":
                    texts.append(part["text"])
        status_msg = task.get("status", {}).get("message", {})
        for part in status_msg.get("parts", []):
            if part.get("type") == "text":
                texts.append(part["text"])
        combined = " ".join(texts)
        assert "answer to: Hello" in combined

    def test_jsonrpc_id_echoed(self, client):
        resp = client.post(
            "/",
            json=_jsonrpc(
                "SendMessage",
                {
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": "hi"}],
                    },
                },
                req_id=42,
            ),
        )
        body = resp.json()
        assert body.get("id") == 42


# ---------------------------------------------------------------------------
# 2. Multi-turn: same contextId shares session
# ---------------------------------------------------------------------------


class TestMultiTurn:
    def test_same_context_reuses_session(self, server, client):
        ctx = str(uuid.uuid4())
        resp1 = _send_message(client, "First question", context_id=ctx)
        assert resp1.status_code == 200

        resp2 = _send_message(client, "Follow-up", context_id=ctx)
        assert resp2.status_code == 200

        # Both tasks should share the same contextId
        t1 = resp1.json()["result"]
        t2 = resp2.json()["result"]
        assert t1["contextId"] == ctx
        assert t2["contextId"] == ctx
        # But they get distinct task IDs
        assert t1["id"] != t2["id"]

    def test_different_context_gets_different_session(self, server, client):
        resp1 = _send_message(client, "Question A", context_id="ctx-a")
        resp2 = _send_message(client, "Question B", context_id="ctx-b")
        t1 = resp1.json()["result"]
        t2 = resp2.json()["result"]
        assert t1["contextId"] == "ctx-a"
        assert t2["contextId"] == "ctx-b"


# ---------------------------------------------------------------------------
# 3. input-required resumption
# ---------------------------------------------------------------------------


class TestInputRequired:
    def test_input_required_task_is_non_terminal(self, monkeypatch, client, server):
        from swival import session as session_mod

        call_count = [0]

        def mock_ask(self, q):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_input_required_result()
            return _make_result("resumed answer")

        monkeypatch.setattr(session_mod.Session, "ask", mock_ask)

        ctx = str(uuid.uuid4())
        resp1 = _send_message(client, "Start task", context_id=ctx)
        t1 = resp1.json()["result"]
        assert t1["status"]["state"] == "input-required"
        task_id = t1["id"]

        # Follow-up with taskId resumes
        resp2 = _send_message(client, "More info", context_id=ctx, task_id=task_id)
        t2 = resp2.json()["result"]
        assert t2["status"]["state"] == "completed"


# ---------------------------------------------------------------------------
# 4. TTL expiry
# ---------------------------------------------------------------------------


class TestTTLExpiry:
    def test_expired_session_gets_fresh_session(self, monkeypatch, _patch_session):
        """An idle session past TTL is cleaned up; follow-up gets a new one."""
        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            ttl=1,  # 1s TTL
        )
        from starlette.testclient import TestClient

        tc = TestClient(srv.app)

        ctx = str(uuid.uuid4())
        resp1 = _send_message(tc, "Hello", context_id=ctx)
        assert resp1.status_code == 200

        # Wait for TTL to expire, then trigger cleanup manually
        time.sleep(1.1)
        srv._cleanup_expired()

        # Even if the session is gone, a follow-up should succeed with a fresh session
        resp2 = _send_message(tc, "Hello again", context_id=ctx)
        assert resp2.status_code == 200
        t2 = resp2.json()["result"]
        assert t2["status"]["state"] == "completed"


# ---------------------------------------------------------------------------
# 5. Concurrent contexts
# ---------------------------------------------------------------------------


class TestConcurrentContexts:
    def test_multiple_contexts_independent(self, client):
        contexts = [str(uuid.uuid4()) for _ in range(5)]
        responses = []
        for ctx in contexts:
            resp = _send_message(client, f"Question for {ctx}", context_id=ctx)
            assert resp.status_code == 200
            responses.append(resp.json()["result"])

        # Each should have its own contextId
        returned_contexts = {r["contextId"] for r in responses}
        assert returned_contexts == set(contexts)

        # All task IDs should be unique
        task_ids = [r["id"] for r in responses]
        assert len(set(task_ids)) == len(task_ids)


# ---------------------------------------------------------------------------
# 6. Agent Card
# ---------------------------------------------------------------------------


class TestAgentCard:
    def test_card_served_at_well_known_path(self, client):
        resp = client.get("/.well-known/agent-card.json")
        assert resp.status_code == 200
        card = resp.json()
        assert "name" in card
        assert "version" in card

    def test_card_has_skills(self, client):
        resp = client.get("/.well-known/agent-card.json")
        card = resp.json()
        assert "skills" in card
        assert isinstance(card["skills"], list)

    def test_card_content_type(self, client):
        resp = client.get("/.well-known/agent-card.json")
        assert "application/json" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# 7. Auth
# ---------------------------------------------------------------------------


class TestAuth:
    def test_no_auth_required_by_default(self, client):
        resp = _send_message(client, "Hello")
        assert resp.status_code == 200

    def test_bearer_token_accepted(self, _patch_session):
        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            auth_token="secret-token-123",
        )
        from starlette.testclient import TestClient

        tc = TestClient(srv.app)

        resp = tc.post(
            "/",
            json=_jsonrpc(
                "SendMessage",
                {
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": "hi"}],
                    },
                },
            ),
            headers={"Authorization": "Bearer secret-token-123"},
        )
        assert resp.status_code == 200
        assert "result" in resp.json()

    def test_missing_token_rejected(self, _patch_session):
        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            auth_token="secret-token-123",
        )
        from starlette.testclient import TestClient

        tc = TestClient(srv.app)

        resp = tc.post(
            "/",
            json=_jsonrpc(
                "SendMessage",
                {
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": "hi"}],
                    },
                },
            ),
        )
        # Should be 401 or a JSON-RPC error
        assert resp.status_code in (401, 403) or "error" in resp.json()

    def test_wrong_token_rejected(self, _patch_session):
        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            auth_token="secret-token-123",
        )
        from starlette.testclient import TestClient

        tc = TestClient(srv.app)

        resp = tc.post(
            "/",
            json=_jsonrpc(
                "SendMessage",
                {
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": "hi"}],
                    },
                },
            ),
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code in (401, 403) or "error" in resp.json()

    def test_agent_card_accessible_without_auth(self, _patch_session):
        """The agent card endpoint should be publicly accessible even with auth."""
        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            auth_token="secret-token-123",
        )
        from starlette.testclient import TestClient

        tc = TestClient(srv.app)
        resp = tc.get("/.well-known/agent-card.json")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 8. GetTask
# ---------------------------------------------------------------------------


class TestGetTask:
    def test_get_existing_task(self, client):
        # Create a task first
        send_resp = _send_message(client, "Hello")
        task_id = send_resp.json()["result"]["id"]

        # Now retrieve it
        get_resp = _get_task(client, task_id)
        assert get_resp.status_code == 200
        body = get_resp.json()
        assert "result" in body
        assert body["result"]["id"] == task_id

    def test_get_nonexistent_task(self, client):
        resp = _get_task(client, "nonexistent-task-id")
        body = resp.json()
        # Should be a JSON-RPC error (task not found)
        assert "error" in body
        err = body["error"]
        assert err.get("code") is not None

    def test_get_task_preserves_status(self, client):
        send_resp = _send_message(client, "Hello")
        task = send_resp.json()["result"]
        task_id = task["id"]

        get_resp = _get_task(client, task_id)
        retrieved = get_resp.json()["result"]
        assert retrieved["status"]["state"] == task["status"]["state"]


# ---------------------------------------------------------------------------
# 9. ListTasks
# ---------------------------------------------------------------------------


class TestListTasks:
    def test_list_tasks_empty(self, client):
        resp = _list_tasks(client, context_id="nonexistent-ctx")
        assert resp.status_code == 200
        body = resp.json()
        assert "result" in body
        tasks = body["result"]
        assert isinstance(tasks, list)
        assert len(tasks) == 0

    def test_list_tasks_by_context(self, client):
        ctx = str(uuid.uuid4())
        _send_message(client, "Q1", context_id=ctx)
        _send_message(client, "Q2", context_id=ctx)
        _send_message(client, "Q3", context_id="other-ctx")

        resp = _list_tasks(client, context_id=ctx)
        tasks = resp.json()["result"]
        assert len(tasks) == 2
        assert all(t["contextId"] == ctx for t in tasks)

    def test_list_all_tasks(self, client):
        _send_message(client, "Q1", context_id="ctx-a")
        _send_message(client, "Q2", context_id="ctx-b")

        resp = _list_tasks(client)
        tasks = resp.json()["result"]
        assert len(tasks) >= 2


# ---------------------------------------------------------------------------
# 10. Unknown method
# ---------------------------------------------------------------------------


class TestUnknownMethod:
    def test_unknown_method_returns_error(self, client):
        resp = client.post("/", json=_jsonrpc("NonExistentMethod", {}))
        assert resp.status_code == 200
        body = resp.json()
        assert "error" in body
        err = body["error"]
        # JSON-RPC method not found code is -32601
        assert err["code"] == -32601

    def test_invalid_json_returns_error(self, client):
        resp = client.post(
            "/",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        body = resp.json()
        assert "error" in body


# ---------------------------------------------------------------------------
# 11. build_agent_card
# ---------------------------------------------------------------------------


class TestBuildAgentCard:
    def test_basic_card(self):
        card = build_agent_card(
            session_kwargs={"provider": "lmstudio", "model": "my-model"},
            host="0.0.0.0",
            port=8080,
        )
        assert card["name"]
        assert card["version"]
        assert isinstance(card["skills"], list)

    def test_card_includes_url(self):
        card = build_agent_card(
            session_kwargs={},
            host="0.0.0.0",
            port=9090,
        )
        # Should have some URL or supportedInterfaces
        has_url = False
        if "url" in card:
            has_url = True
        for iface in card.get("supportedInterfaces", []):
            if "url" in iface:
                has_url = True
        assert has_url

    def test_card_name_includes_provider(self):
        card = build_agent_card(
            session_kwargs={"provider": "openrouter", "model": "qwen3"},
            host="127.0.0.1",
            port=5000,
        )
        assert "openrouter" in card["name"]
        assert "qwen3" in card["name"]

    def test_card_has_description(self):
        card = build_agent_card(
            session_kwargs={},
            host="127.0.0.1",
            port=5000,
        )
        assert card["description"]

    def test_card_protocol_version_on_interface(self):
        card = build_agent_card(
            session_kwargs={},
            host="127.0.0.1",
            port=5000,
        )
        # protocolVersion must be on the interface entry, not top-level
        assert "protocolVersion" not in card
        ifaces = card.get("supportedInterfaces", [])
        assert len(ifaces) == 1
        assert ifaces[0]["protocolVersion"] == "1.0"

    def test_custom_name(self):
        card = build_agent_card(
            session_kwargs={"provider": "openrouter", "model": "qwen3"},
            host="127.0.0.1",
            port=5000,
            name="My Custom Agent",
        )
        assert card["name"] == "My Custom Agent"

    def test_custom_description(self):
        card = build_agent_card(
            session_kwargs={},
            host="127.0.0.1",
            port=5000,
            description="A specialized agent for code review",
        )
        assert card["description"] == "A specialized agent for code review"

    def test_custom_name_none_uses_default(self):
        card = build_agent_card(
            session_kwargs={"provider": "openrouter", "model": "qwen3"},
            host="127.0.0.1",
            port=5000,
            name=None,
        )
        assert "openrouter" in card["name"]
        assert "qwen3" in card["name"]

    def test_custom_description_none_uses_default(self):
        card = build_agent_card(
            session_kwargs={},
            host="127.0.0.1",
            port=5000,
            description=None,
        )
        assert "coding agent" in card["description"]

    def test_skills_in_card(self):
        skills = [
            {
                "id": "review",
                "name": "Code Review",
                "description": "Analyze code",
                "examples": ["Review this PR"],
            },
            {
                "id": "explain",
                "name": "Code Explanation",
                "description": "Explain code",
            },
        ]
        card = build_agent_card(
            session_kwargs={},
            host="127.0.0.1",
            port=5000,
            skills=skills,
        )
        assert len(card["skills"]) == 2
        assert card["skills"][0]["id"] == "review"
        assert card["skills"][0]["name"] == "Code Review"
        assert card["skills"][0]["examples"] == ["Review this PR"]
        assert card["skills"][1]["id"] == "explain"
        assert "examples" not in card["skills"][1]

    def test_skills_none_gives_empty_list(self):
        card = build_agent_card(
            session_kwargs={},
            host="127.0.0.1",
            port=5000,
            skills=None,
        )
        assert card["skills"] == []

    def test_skills_empty_list(self):
        card = build_agent_card(
            session_kwargs={},
            host="127.0.0.1",
            port=5000,
            skills=[],
        )
        assert card["skills"] == []

    def test_all_customizations_together(self):
        skills = [{"id": "ask", "name": "Ask", "description": "Ask a question"}]
        card = build_agent_card(
            session_kwargs={"provider": "openrouter", "model": "qwen3"},
            host="127.0.0.1",
            port=5000,
            auth_token="secret",
            name="My Agent",
            description="Does things",
            skills=skills,
        )
        assert card["name"] == "My Agent"
        assert card["description"] == "Does things"
        assert len(card["skills"]) == 1
        assert "securitySchemes" in card


# ---------------------------------------------------------------------------
# 12. A2aTask dataclass
# ---------------------------------------------------------------------------


class TestA2aTask:
    def test_create_task(self):
        task = A2aTask(
            id="t1",
            context_id="c1",
            status="completed",
        )
        assert task.id == "t1"
        assert task.context_id == "c1"
        assert task.status == "completed"

    def test_task_defaults(self):
        task = A2aTask(id="t2", context_id="c2")
        assert task.messages == []
        assert task.artifacts == []
        assert task.status == "working"

    def test_task_timestamps(self):
        task = A2aTask(id="t3", context_id="c3")
        assert task.created_at is not None
        assert task.updated_at is not None
        assert task.updated_at >= task.created_at


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_message_text(self, client):
        resp = _send_message(client, "")
        # Should still succeed or return a sensible error
        assert resp.status_code == 200

    def test_missing_message_in_params(self, client):
        resp = client.post("/", json=_jsonrpc("SendMessage", {}))
        body = resp.json()
        # Should return an error for missing required params
        assert "error" in body

    def test_params_as_list_returns_error(self, client):
        """params as a JSON array should return INVALID_PARAMS, not crash."""
        resp = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "SendMessage",
                "params": [],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == -32602

    def test_params_as_string_returns_error(self, client):
        """params as a string should return INVALID_PARAMS."""
        resp = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "SendMessage",
                "params": "bad",
            },
        )
        body = resp.json()
        assert "error" in body

    def test_post_to_agent_card_path(self, client):
        """POST to the agent card path should not crash."""
        resp = client.post("/.well-known/agent-card.json", content=b"{}")
        # Could be 405 or just ignored; should not be 500
        assert resp.status_code != 500

    def test_get_to_jsonrpc_endpoint(self, client):
        """GET to / should not crash (maybe 405)."""
        resp = client.get("/")
        assert resp.status_code != 500


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestLRUEviction:
    def test_eviction_on_capacity(self, _patch_session):
        """When max_sessions is reached, LRU session is evicted (not rejected)."""
        srv = A2aServer(
            session_kwargs={"provider": "lmstudio", "base_dir": "/tmp"},
            host="127.0.0.1",
            port=0,
            max_sessions=2,
        )
        from starlette.testclient import TestClient

        tc = TestClient(srv.app)

        # Fill to capacity
        resp1 = _send_message(tc, "Q1", context_id="ctx-1")
        assert resp1.json()["result"]["status"]["state"] == "completed"
        resp2 = _send_message(tc, "Q2", context_id="ctx-2")
        assert resp2.json()["result"]["status"]["state"] == "completed"
        assert len(srv._sessions) == 2

        # Third context should succeed by evicting the LRU (ctx-1)
        resp3 = _send_message(tc, "Q3", context_id="ctx-3")
        assert resp3.json()["result"]["status"]["state"] == "completed"
        assert len(srv._sessions) == 2
        assert "ctx-3" in srv._sessions
        assert "ctx-1" not in srv._sessions
