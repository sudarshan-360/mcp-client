"""
app/core/session_extended.py
──────────────────────────────
Enhanced session management that stores case context for multi-turn conversations.

Flow:
  1. Initial /chat/stream → stores case context in session
  2. Follow-up /chat/follow-up → retrieves case context, constrains response
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.models.chat import Message, Role
from app.models.session import ExtendedSession, CaseContextData


COLLECTION = "sessions_extended"


async def get_or_create_extended_session(
    db: AsyncIOMotorDatabase,
    session_id: str | None = None,
) -> ExtendedSession:
    """
    Load or create an extended session with case context support.
    
    Args:
        db: MongoDB database instance
        session_id: Optional session ID (creates new if None)
    
    Returns:
        ExtendedSession with case context if available
    """
    if session_id:
        doc = await db[COLLECTION].find_one({"session_id": session_id})
        if doc:
            doc.pop("_id", None)
            messages = [Message(**m) for m in doc.get("messages", [])]
            case_data = doc.get("case_context")
            case_context = CaseContextData(**case_data) if case_data else None
            
            return ExtendedSession(
                session_id=doc["session_id"],
                messages=messages,
                case_context=case_context,
                created_at=doc.get("created_at", datetime.utcnow()),
                updated_at=doc.get("updated_at", datetime.utcnow()),
                metadata=doc.get("metadata", {}),
            )

    # Create new session
    new_id = session_id or str(uuid.uuid4())
    session = ExtendedSession(session_id=new_id)
    await db[COLLECTION].insert_one(_to_doc(session))
    return session


async def store_case_context(
    db: AsyncIOMotorDatabase,
    session_id: str,
    case_context: CaseContextData,
) -> None:
    """
    Store the case context (initial tool output + patient data) in the session.
    Called after the initial decision-support query.
    
    Args:
        db: MongoDB database instance
        session_id: Session ID
        case_context: Case data to store
    """
    await db[COLLECTION].update_one(
        {"session_id": session_id},
        {
            "$set": {
                "case_context": case_context.model_dump(),
                "updated_at": datetime.utcnow(),
            },
        },
        upsert=True,
    )


async def append_messages_extended(
    db: AsyncIOMotorDatabase,
    session_id: str,
    messages: list[Message],
) -> None:
    """
    Append messages to the session history.
    
    Args:
        db: MongoDB database instance
        session_id: Session ID
        messages: Messages to append
    """
    await db[COLLECTION].update_one(
        {"session_id": session_id},
        {
            "$push": {"messages": {"$each": [m.model_dump() for m in messages]}},
            "$set": {"updated_at": datetime.utcnow()},
        },
        upsert=True,
    )


async def get_extended_session(
    db: AsyncIOMotorDatabase, session_id: str
) -> ExtendedSession | None:
    """
    Fetch an existing extended session.
    
    Args:
        db: MongoDB database instance
        session_id: Session ID
    
    Returns:
        ExtendedSession or None if not found
    """
    doc = await db[COLLECTION].find_one({"session_id": session_id})
    if not doc:
        return None
    
    doc.pop("_id", None)
    messages = [Message(**m) for m in doc.get("messages", [])]
    case_data = doc.get("case_context")
    case_context = CaseContextData(**case_data) if case_data else None
    
    return ExtendedSession(
        session_id=doc["session_id"],
        messages=messages,
        case_context=case_context,
        created_at=doc.get("created_at", datetime.utcnow()),
        updated_at=doc.get("updated_at", datetime.utcnow()),
        metadata=doc.get("metadata", {}),
    )


async def delete_extended_session(
    db: AsyncIOMotorDatabase, session_id: str
) -> bool:
    """
    Delete an extended session.
    
    Args:
        db: MongoDB database instance
        session_id: Session ID
    
    Returns:
        True if deleted, False if not found
    """
    result = await db[COLLECTION].delete_one({"session_id": session_id})
    return result.deleted_count > 0


def _to_doc(session: ExtendedSession) -> dict[str, Any]:
    """Convert ExtendedSession to MongoDB document."""
    d = session.model_dump()
    d["messages"] = [m.model_dump() for m in session.messages]
    if session.case_context:
        d["case_context"] = session.case_context.model_dump()
    return d