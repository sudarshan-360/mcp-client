from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# Roles
# ─────────────────────────────────────────────────────────────

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ─────────────────────────────────────────────────────────────
# Chat Messages
# ─────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: Role
    content: str
    tool_call_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# Tool Call Tracking
# ─────────────────────────────────────────────────────────────

class ToolCallInfo(BaseModel):
    tool_name: str
    arguments: dict
    result: Optional[str] = None
    error: Optional[str] = None

class ToolInfo(BaseModel):
    name: str
    description: Optional[str] = None

# ─────────────────────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────────────────────

from datetime import datetime
from typing import Any

class Session(BaseModel):
    session_id: str
    messages: list[Message] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    metadata: dict[str, Any] = Field(default_factory=dict)
# ─────────────────────────────────────────────────────────────
# Incoming Request
# ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


# ─────────────────────────────────────────────────────────────
# SSE Events
# ─────────────────────────────────────────────────────────────

class SSEEventType(str, Enum):
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_RESULT = "tool_call_result"
    TEXT_DELTA = "text_delta"
    DONE = "done"


class SSEEvent(BaseModel):
    event: SSEEventType
    data: str