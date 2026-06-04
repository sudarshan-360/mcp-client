"""
app/models/chat.py
──────────────────
Deterministic-routing version.

Flow:

Frontend Form
      ↓
ChatRequest
      ↓
CancerType Validation
      ↓
Tool Router
      ↓
MCP Tool
      ↓
LLM Formatter
      ↓
SSE Stream
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, ConfigDict

from app.core.tool_router import CancerType


# =============================================================================
# Roles
# =============================================================================

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# =============================================================================
# Messages
# =============================================================================

class Message(BaseModel):
    role: Role
    content: str
    tool_call_id: Optional[str] = None


# =============================================================================
# Tool Tracking
# =============================================================================

class ToolCallInfo(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    result: Optional[str] = None
    error: Optional[str] = None


class ToolInfo(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Optional[dict[str, Any]] = None


# =============================================================================
# Session
# =============================================================================

class Session(BaseModel):
    session_id: str

    messages: list[Message] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Clinical Parameters
# =============================================================================

class ClinicalParameters(BaseModel):
    """
    Accept arbitrary cancer-specific fields.

    Example:

    {
        "age": 45,
        "sex": "female",
        "menopausal_status": "postmenopausal",
        "t_stage": "T2",
        "n_stage": "N1",
        "her2_status": "positive"
    }
    """

    model_config = ConfigDict(
        extra="allow"
    )


# =============================================================================
# Incoming Request
# =============================================================================

class ChatRequest(BaseModel):
    """
    Frontend sends structured form data.

    Example:

    {
        "cancer_type": "breast",
        "patient_data": {
            "age": 45,
            "t_stage": "T2",
            "n_stage": "N1"
        },
        "request_context": "Summarize treatment options"
    }
    """

    cancer_type: CancerType = Field(
        ...,
        description="Cancer type used for deterministic routing"
    )

    patient_data: ClinicalParameters = Field(
        ...,
        description="Structured clinical parameters"
    )

    request_context: Optional[str] = Field(
        default=None,
        description="Optional clinician question"
    )

    session_id: Optional[str] = Field(
        default=None,
        description="Optional session id"
    )


# =============================================================================
# SSE Events
# =============================================================================

class SSEEventType(str, Enum):
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_RESULT = "tool_call_result"
    TEXT_DELTA = "text_delta"
    DONE = "done"
    ERROR = "error"


class SSEEvent(BaseModel):
    event: SSEEventType
    data: str