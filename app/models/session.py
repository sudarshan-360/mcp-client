"""
app/models/session.py
─────────────────────
Extended session model with case context for multi-turn conversations.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field

from app.models.chat import Message


class CaseContextData(BaseModel):
    """
    Stores the original case and the initial decision-support output.
    """
    
    cancer_type: str
    patient_data: dict[str, Any]
    request_context: Optional[str] = None
    
    initial_tool_output: str
    initial_llm_response: str
    
    tool_name: str
    parameters_sent: int


class ExtendedSession(BaseModel):
    """
    Extended session with case context for multi-turn conversations.
    """
    
    session_id: str
    
    messages: list[Message] = Field(default_factory=list)
    
    # Store the original case context
    case_context: Optional[CaseContextData] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    @property
    def has_case_context(self) -> bool:
        """Check if we have an active case to reference."""
        return self.case_context is not None