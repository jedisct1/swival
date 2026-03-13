"""A2A (Agent-to-Agent) server for swival.

Exposes a swival Session as an A2A endpoint so other agents can call it.
Uses starlette + uvicorn for lightweight async HTTP serving.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .a2a_types import (
    A2A_VERSION,
    AGENT_CARD_PATH,
    METHOD_GET_TASK,
    METHOD_LIST_TASKS,
    METHOD_SEND_MESSAGE,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_INPUT_REQUIRED,
    STATE_WORKING,
    TERMINAL_STATES,
    extract_text_from_parts,
)
from .session import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TTL = 3600  # 1 hour
DEFAULT_MAX_SESSIONS = 100
CLEANUP_INTERVAL = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Task dataclass (server-side, not the same as a2a_types.Task which is
# the client-side wire representation)
# ---------------------------------------------------------------------------


@dataclass
class A2aTask:
    """Server-side task record."""

    id: str
    context_id: str
    status: str = STATE_WORKING
    messages: list[dict] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)

    def to_wire(self) -> dict:
        """Serialize to A2A v1.0 wire format (camelCase)."""
        result: dict[str, Any] = {
            "id": self.id,
            "contextId": self.context_id,
            "status": {"state": self.status},
            "artifacts": self.artifacts,
        }
        # Attach the last agent message to the status if present
        for msg in reversed(self.messages):
            if msg.get("role") == "agent":
                result["status"]["message"] = msg
                break
        return result


# ---------------------------------------------------------------------------
# Agent Card generation
# ---------------------------------------------------------------------------


def _build_skills_list(skills: list[dict] | None) -> list[dict]:
    """Convert skill dicts to A2A AgentSkill wire format (camelCase keys)."""
    if not skills:
        return []
    result = []
    for s in skills:
        entry: dict[str, Any] = {"id": s["id"]}
        if "name" in s:
            entry["name"] = s["name"]
        if "description" in s:
            entry["description"] = s["description"]
        if "examples" in s:
            entry["examples"] = s["examples"]
        result.append(entry)
    return result


def build_agent_card(
    session_kwargs: dict,
    host: str,
    port: int,
    *,
    auth_token: str | None = None,
    name: str | None = None,
    description: str | None = None,
    skills: list[dict] | None = None,
) -> dict:
    """Auto-generate an A2A Agent Card from session config.

    Returns a dict ready to be served as JSON at /.well-known/agent-card.json.
    """
    if name is None:
        provider = session_kwargs.get("provider", "lmstudio")
        model = session_kwargs.get("model") or "default"
        name = f"swival ({provider}/{model})"
    if description is None:
        description = (
            "A coding agent powered by swival. Accepts natural-language tasks "
            "and executes them using tool-augmented LLM reasoning."
        )

    url = f"http://{host}:{port}/"

    card: dict[str, Any] = {
        "name": name,
        "description": description,
        "version": "0.1.0",
        "url": url,
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "extendedAgentCard": False,
        },
        "supportedInterfaces": [
            {
                "protocolBinding": "JSONRPC",
                "protocolVersion": A2A_VERSION,
                "url": url,
            }
        ],
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": _build_skills_list(skills),
    }

    if auth_token:
        card["securitySchemes"] = {
            "bearer": {
                "type": "http",
                "scheme": "bearer",
            }
        }
        card["securityRequirements"] = [{"bearer": []}]

    return card


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------


def _jsonrpc_error(req_id: Any, code: int, message: str) -> dict:
    """Build a JSON-RPC 2.0 error response."""
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


def _jsonrpc_result(req_id: Any, result: Any) -> dict:
    """Build a JSON-RPC 2.0 success response."""
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": result,
    }


# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
# A2A-specific error codes
TASK_NOT_FOUND = -32001
CONTEXT_NOT_FOUND = -32002


# ---------------------------------------------------------------------------
# A2aServer
# ---------------------------------------------------------------------------


class A2aServer:
    """Serves a swival Session over A2A protocol.

    Each unique contextId gets its own Session instance with persistent
    conversation state (via Session.ask()). Tasks are tracked in memory.
    """

    def __init__(
        self,
        session_kwargs: dict,
        host: str = "0.0.0.0",
        port: int = 8080,
        auth_token: str | None = None,
        ttl: int = DEFAULT_TTL,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        name: str | None = None,
        description: str | None = None,
        skills: list[dict] | None = None,
    ):
        self.session_kwargs = dict(session_kwargs)
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self.ttl = ttl
        self.max_sessions = max_sessions

        # context_id -> Session
        self._sessions: dict[str, Session] = {}
        # context_id -> last-access monotonic time
        self._session_access: dict[str, float] = {}
        # context_id -> asyncio.Lock (serialize per-context calls)
        self._context_locks: dict[str, asyncio.Lock] = {}

        # task_id -> A2aTask
        self._tasks: dict[str, A2aTask] = {}
        # context_id -> [task_id, ...]
        self._context_tasks: dict[str, list[str]] = {}

        # Agent card (built once)
        self._agent_card = build_agent_card(
            session_kwargs,
            host,
            port,
            auth_token=auth_token,
            name=name,
            description=description,
            skills=skills,
        )

        self._cleanup_task: asyncio.Task | None = None
        self._app: Starlette | None = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _create_session(self, context_id: str) -> Session:
        """Create a new Session for a context, respecting max_sessions via LRU."""
        if len(self._sessions) >= self.max_sessions:
            self._evict_lru()
        session = Session(**self.session_kwargs)
        self._sessions[context_id] = session
        self._session_access[context_id] = time.monotonic()
        logger.info("Created session for context %s", context_id)
        return session

    def _get_or_create_session(self, context_id: str) -> Session:
        """Get existing session or create a new one."""
        session = self._sessions.get(context_id)
        if session is None:
            session = self._create_session(context_id)
        self._session_access[context_id] = time.monotonic()
        return session

    def _get_context_lock(self, context_id: str) -> asyncio.Lock:
        """Get or create the per-context asyncio lock."""
        if context_id not in self._context_locks:
            self._context_locks[context_id] = asyncio.Lock()
        return self._context_locks[context_id]

    def _evict_lru(self) -> None:
        """Evict the least-recently-used session to make room."""
        if not self._session_access:
            return
        oldest_ctx = min(self._session_access, key=self._session_access.get)  # type: ignore[arg-type]
        self._remove_context(oldest_ctx)
        logger.info("Evicted LRU session for context %s", oldest_ctx)

    def _remove_context(self, context_id: str) -> None:
        """Remove a context and all its associated state."""
        session = self._sessions.pop(context_id, None)
        if session is not None:
            # Best-effort cleanup
            try:
                session.__exit__(None, None, None)
            except Exception:
                logger.debug(
                    "Error closing session for context %s", context_id, exc_info=True
                )
        self._session_access.pop(context_id, None)
        self._context_locks.pop(context_id, None)
        # Remove associated tasks
        task_ids = self._context_tasks.pop(context_id, [])
        for tid in task_ids:
            self._tasks.pop(tid, None)

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def _create_task(self, context_id: str) -> A2aTask:
        """Create a new task for a context."""
        task_id = str(uuid.uuid4())
        now = time.monotonic()
        task = A2aTask(
            id=task_id,
            context_id=context_id,
            status="working",
            created_at=now,
            updated_at=now,
        )
        self._tasks[task_id] = task
        self._context_tasks.setdefault(context_id, []).append(task_id)
        return task

    # ------------------------------------------------------------------
    # TTL cleanup
    # ------------------------------------------------------------------

    async def _cleanup_loop(self) -> None:
        """Background coroutine that periodically removes expired sessions/tasks."""
        while True:
            await asyncio.sleep(CLEANUP_INTERVAL)
            try:
                self._cleanup_expired()
            except Exception:
                logger.debug("Cleanup error", exc_info=True)

    def _cleanup_expired(self) -> None:
        """Remove sessions and tasks older than TTL, plus orphaned locks."""
        now = time.monotonic()
        expired = [
            ctx_id
            for ctx_id, last_access in self._session_access.items()
            if now - last_access > self.ttl
        ]
        for ctx_id in expired:
            self._remove_context(ctx_id)
            logger.info("Expired session for context %s", ctx_id)
        # Clean up orphaned locks (contexts that were removed but locks linger)
        orphaned = set(self._context_locks) - set(self._sessions)
        for ctx_id in orphaned:
            self._context_locks.pop(ctx_id, None)

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _check_auth(self, request: Request) -> str | None:
        """Check bearer auth if configured. Returns error message or None."""
        if not self.auth_token:
            return None
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return "Missing or invalid Authorization header"
        token = auth[7:]
        if not hmac.compare_digest(token, self.auth_token):
            return "Invalid bearer token"
        return None

    # ------------------------------------------------------------------
    # A2A message extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_message_text(message: dict) -> str:
        """Extract plain text from an A2A message dict."""
        return extract_text_from_parts(message.get("parts", []))

    # ------------------------------------------------------------------
    # JSON-RPC method handlers
    # ------------------------------------------------------------------

    async def _handle_send_message(self, params: dict, req_id: Any) -> dict:
        """Handle SendMessage JSON-RPC method."""
        # Extract message
        message = params.get("message")
        if not message:
            return _jsonrpc_error(req_id, INVALID_PARAMS, "Missing 'message' in params")

        message_text = self._extract_message_text(message)
        if not message_text.strip():
            return _jsonrpc_error(req_id, INVALID_PARAMS, "Empty message text")

        # Determine context
        context_id = message.get("contextId") or params.get("contextId")
        task_id = message.get("taskId") or params.get("taskId")

        # If resuming an existing task, use its context
        existing_task: A2aTask | None = None
        if task_id:
            existing_task = self._tasks.get(task_id)
            if existing_task is not None:
                context_id = existing_task.context_id

        if not context_id:
            context_id = str(uuid.uuid4())

        # Acquire per-context lock
        lock = self._get_context_lock(context_id)
        async with lock:
            session = self._get_or_create_session(context_id)

            # Create or reuse task
            if existing_task and existing_task.status not in TERMINAL_STATES:
                task = existing_task
                task.status = STATE_WORKING
                task.updated_at = time.monotonic()
            else:
                task = self._create_task(context_id)

            # Record the user message
            user_msg = {
                "role": "user",
                "parts": message.get("parts", [{"type": "text", "text": message_text}]),
                "contextId": context_id,
                "taskId": task.id,
            }
            task.messages.append(user_msg)

            # Run session.ask() in thread pool (Session is synchronous)
            try:
                result = await asyncio.to_thread(session.ask, message_text)
            except Exception as exc:
                logger.error(
                    "Session.ask() failed for context %s: %s",
                    context_id,
                    exc,
                    exc_info=True,
                )
                task.status = STATE_FAILED
                task.updated_at = time.monotonic()
                agent_msg = {
                    "role": "agent",
                    "parts": [{"type": "text", "text": f"Internal error: {exc}"}],
                    "contextId": context_id,
                    "taskId": task.id,
                }
                task.messages.append(agent_msg)
                return _jsonrpc_result(req_id, task.to_wire())

            # Determine outcome
            answer_text = result.answer or ""
            if result.exhausted and result.answer is None:
                # Exhausted with no answer: treat as input-required
                # (the agent couldn't complete without more input)
                task.status = STATE_INPUT_REQUIRED
            elif result.exhausted:
                task.status = STATE_FAILED
            else:
                task.status = STATE_COMPLETED

            task.updated_at = time.monotonic()

            # Record agent message
            agent_msg = {
                "role": "agent",
                "parts": [{"type": "text", "text": answer_text}],
                "contextId": context_id,
                "taskId": task.id,
            }
            task.messages.append(agent_msg)

            # Record artifact
            if answer_text:
                task.artifacts.append(
                    {
                        "parts": [{"type": "text", "text": answer_text}],
                    }
                )

        return _jsonrpc_result(req_id, task.to_wire())

    async def _handle_get_task(self, params: dict, req_id: Any) -> dict:
        """Handle GetTask JSON-RPC method."""
        task_id = params.get("id")
        if not task_id:
            return _jsonrpc_error(req_id, INVALID_PARAMS, "Missing 'id' in params")

        task = self._tasks.get(task_id)
        if task is None:
            return _jsonrpc_error(req_id, TASK_NOT_FOUND, f"Task not found: {task_id}")

        return _jsonrpc_result(req_id, task.to_wire())

    async def _handle_list_tasks(self, params: dict, req_id: Any) -> dict:
        """Handle ListTasks JSON-RPC method."""
        context_id = params.get("contextId")
        if context_id:
            task_ids = self._context_tasks.get(context_id, [])
            tasks = [
                self._tasks[tid].to_wire() for tid in task_ids if tid in self._tasks
            ]
        else:
            tasks = [t.to_wire() for t in self._tasks.values()]
        return _jsonrpc_result(req_id, tasks)

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    async def _handle_agent_card(self, request: Request) -> JSONResponse:
        """Serve the Agent Card at /.well-known/agent-card.json."""
        return JSONResponse(self._agent_card)

    async def _handle_jsonrpc(self, request: Request) -> JSONResponse:
        """Handle JSON-RPC 2.0 requests at POST /."""
        # Auth check
        auth_err = self._check_auth(request)
        if auth_err:
            return JSONResponse(
                _jsonrpc_error(None, -32000, auth_err),
                status_code=401,
            )

        # Parse body
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                _jsonrpc_error(None, PARSE_ERROR, "Invalid JSON"),
                status_code=400,
            )

        # Validate JSON-RPC envelope
        if not isinstance(body, dict):
            return JSONResponse(
                _jsonrpc_error(None, INVALID_REQUEST, "Request must be a JSON object"),
                status_code=400,
            )

        req_id = body.get("id")
        method = body.get("method")
        params = body.get("params", {})

        if not isinstance(params, dict):
            return JSONResponse(
                _jsonrpc_error(req_id, INVALID_PARAMS, "params must be a JSON object"),
                status_code=200,
            )

        if body.get("jsonrpc") != "2.0":
            return JSONResponse(
                _jsonrpc_error(
                    req_id, INVALID_REQUEST, "Missing or invalid jsonrpc version"
                ),
                status_code=400,
            )

        if not method:
            return JSONResponse(
                _jsonrpc_error(req_id, INVALID_REQUEST, "Missing method"),
                status_code=400,
            )

        # Route to handler
        if method == METHOD_SEND_MESSAGE:
            result = await self._handle_send_message(params, req_id)
        elif method == METHOD_GET_TASK:
            result = await self._handle_get_task(params, req_id)
        elif method == METHOD_LIST_TASKS:
            result = await self._handle_list_tasks(params, req_id)
        else:
            result = _jsonrpc_error(
                req_id, METHOD_NOT_FOUND, f"Unknown method: {method}"
            )

        return JSONResponse(result)

    # ------------------------------------------------------------------
    # App construction and serve()
    # ------------------------------------------------------------------

    def _build_app(self) -> Starlette:
        """Build the Starlette ASGI application."""
        routes = [
            Route(AGENT_CARD_PATH, self._handle_agent_card, methods=["GET"]),
            Route("/", self._handle_jsonrpc, methods=["POST"]),
        ]

        @contextlib.asynccontextmanager
        async def lifespan(app):
            # Startup
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info(
                "A2A server starting on %s:%d (TTL=%ds, max_sessions=%d)",
                self.host,
                self.port,
                self.ttl,
                self.max_sessions,
            )
            yield
            # Shutdown
            if self._cleanup_task is not None:
                self._cleanup_task.cancel()
                self._cleanup_task = None
            for ctx_id in list(self._sessions):
                self._remove_context(ctx_id)
            logger.info("A2A server shut down, all sessions closed")

        app = Starlette(routes=routes, lifespan=lifespan)
        return app

    @property
    def app(self) -> Starlette:
        """The ASGI application (for testing or external ASGI servers)."""
        if self._app is None:
            self._app = self._build_app()
        return self._app

    def serve(self) -> None:
        """Start the A2A server (blocking)."""
        import uvicorn

        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
