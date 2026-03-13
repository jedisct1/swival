"""Tests for swival.a2a_types: wire constants, dataclasses, serialization."""

import pytest

from swival.a2a_types import (
    INTERRUPTED_STATES,
    TERMINAL_STATES,
    AgentCard,
    Message,
    SendMessageConfiguration,
    Task,
    TaskStatus,
    extract_task_text,
    extract_text_from_parts,
    from_wire,
    jsonrpc_request,
    parse_jsonrpc_response,
    sanitize_skill_id,
    to_wire,
    validate_server_name,
    _to_camel,
    _to_snake,
)


class TestCamelSnakeConversion:
    def test_to_camel(self):
        assert _to_camel("context_id") == "contextId"
        assert _to_camel("return_immediately") == "returnImmediately"
        assert _to_camel("name") == "name"

    def test_to_snake(self):
        assert _to_snake("contextId") == "context_id"
        assert _to_snake("returnImmediately") == "return_immediately"
        assert _to_snake("name") == "name"

    def test_to_wire_nested(self):
        d = {"context_id": "abc", "nested": {"return_immediately": True}}
        result = to_wire(d)
        assert result == {"contextId": "abc", "nested": {"returnImmediately": True}}

    def test_from_wire_nested(self):
        d = {"contextId": "abc", "nested": {"returnImmediately": True}}
        result = from_wire(d)
        assert result == {"context_id": "abc", "nested": {"return_immediately": True}}

    def test_to_wire_list(self):
        d = {"items": [{"skill_id": "a"}, {"skill_id": "b"}]}
        result = to_wire(d)
        assert result == {"items": [{"skillId": "a"}, {"skillId": "b"}]}

    def test_roundtrip(self):
        original = {"context_id": "x", "task_id": "y", "parts": [{"type": "text"}]}
        assert from_wire(to_wire(original)) == original


class TestValidateServerName:
    def test_valid(self):
        validate_server_name("my-agent")
        validate_server_name("agent123")
        validate_server_name("a")

    def test_invalid_chars(self):
        with pytest.raises(Exception, match="invalid"):
            validate_server_name("my agent")

    def test_double_underscore(self):
        with pytest.raises(Exception, match="double underscores"):
            validate_server_name("my__agent")


class TestSanitizeSkillId:
    def test_basic(self):
        assert sanitize_skill_id("web-search") == "web-search"

    def test_special_chars(self):
        assert sanitize_skill_id("my.skill/v2") == "my_skill_v2"

    def test_empty(self):
        assert sanitize_skill_id("") == "ask"

    def test_double_underscore(self):
        assert sanitize_skill_id("a__b") == "a_b"


class TestMessage:
    def test_to_wire_minimal(self):
        msg = Message(role="user", parts=[{"type": "text", "text": "hello"}])
        wire = msg.to_wire()
        assert wire == {"role": "user", "parts": [{"type": "text", "text": "hello"}]}
        assert "contextId" not in wire
        assert "taskId" not in wire

    def test_to_wire_with_ids(self):
        msg = Message(
            role="user",
            parts=[{"type": "text", "text": "hello"}],
            context_id="ctx1",
            task_id="task1",
        )
        wire = msg.to_wire()
        assert wire["contextId"] == "ctx1"
        assert wire["taskId"] == "task1"


class TestSendMessageConfiguration:
    def test_defaults(self):
        config = SendMessageConfiguration()
        wire = config.to_wire()
        assert wire["returnImmediately"] is False
        assert "text/plain" in wire["acceptedOutputModes"]


class TestTaskStatus:
    def test_from_wire(self):
        ts = TaskStatus.from_wire({"state": "completed"})
        assert ts.state == "completed"
        assert ts.message is None

    def test_from_wire_with_message(self):
        ts = TaskStatus.from_wire(
            {
                "state": "input-required",
                "message": {
                    "role": "agent",
                    "parts": [{"type": "text", "text": "Need more info"}],
                    "contextId": "ctx1",
                },
            }
        )
        assert ts.state == "input-required"
        assert ts.message is not None
        assert ts.message.context_id == "ctx1"


class TestTask:
    def test_from_wire_completed(self):
        data = {
            "id": "t1",
            "contextId": "c1",
            "status": {"state": "completed"},
            "artifacts": [
                {"parts": [{"type": "text", "text": "Done!"}]},
            ],
        }
        task = Task.from_wire(data)
        assert task.id == "t1"
        assert task.context_id == "c1"
        assert task.state == "completed"
        assert task.is_terminal
        assert not task.is_interrupted
        assert len(task.artifacts) == 1

    def test_from_wire_input_required(self):
        data = {
            "id": "t2",
            "contextId": "c2",
            "status": {
                "state": "input-required",
                "message": {
                    "role": "agent",
                    "parts": [{"type": "text", "text": "What file?"}],
                },
            },
        }
        task = Task.from_wire(data)
        assert task.state == "input-required"
        assert task.is_interrupted
        assert not task.is_terminal

    def test_terminal_states(self):
        for state in TERMINAL_STATES:
            task = Task.from_wire({"status": {"state": state}})
            assert task.is_terminal

    def test_interrupted_states(self):
        for state in INTERRUPTED_STATES:
            task = Task.from_wire({"status": {"state": state}})
            assert task.is_interrupted


class TestAgentCard:
    def test_from_wire(self):
        data = {
            "name": "test-agent",
            "description": "A test agent",
            "version": "1.0.0",
            "supportedInterfaces": [
                {
                    "url": "https://example.com",
                    "protocolBinding": "JSONRPC",
                    "protocolVersion": "1.0",
                },
            ],
            "capabilities": {"streaming": True},
            "skills": [
                {"id": "search", "name": "Search", "description": "Search the web"},
            ],
        }
        card = AgentCard.from_wire(data)
        assert card.name == "test-agent"
        assert card.url == "https://example.com"
        assert len(card.skills) == 1
        assert card.skills[0].id == "search"

    def test_from_wire_no_interfaces(self):
        card = AgentCard.from_wire({"name": "agent"})
        assert card.url == ""


class TestJsonRpc:
    def test_request(self):
        req = jsonrpc_request("SendMessage", {"message": {}})
        assert req["jsonrpc"] == "2.0"
        assert req["method"] == "SendMessage"
        assert req["params"] == {"message": {}}
        assert req["id"] == 1

    def test_parse_response_success(self):
        data = {"jsonrpc": "2.0", "id": 1, "result": {"id": "t1"}}
        result = parse_jsonrpc_response(data)
        assert result == {"id": "t1"}

    def test_parse_response_error(self):
        data = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32600, "message": "Invalid Request"},
        }
        with pytest.raises(ValueError, match="Invalid Request"):
            parse_jsonrpc_response(data)


class TestExtractText:
    def test_extract_text_from_parts(self):
        parts = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        assert extract_text_from_parts(parts) == "hello\nworld"

    def test_extract_text_empty(self):
        assert extract_text_from_parts([]) == ""

    def test_extract_task_text_artifacts(self):
        task = Task.from_wire(
            {
                "id": "t1",
                "status": {"state": "completed"},
                "artifacts": [{"parts": [{"type": "text", "text": "result"}]}],
            }
        )
        assert extract_task_text(task) == "result"

    def test_extract_task_text_status_message(self):
        task = Task.from_wire(
            {
                "id": "t1",
                "status": {
                    "state": "input-required",
                    "message": {
                        "role": "agent",
                        "parts": [{"type": "text", "text": "need input"}],
                    },
                },
            }
        )
        assert extract_task_text(task) == "need input"

    def test_extract_task_text_empty(self):
        task = Task.from_wire({"id": "t1", "status": {"state": "completed"}})
        assert extract_task_text(task) == "(empty response)"
