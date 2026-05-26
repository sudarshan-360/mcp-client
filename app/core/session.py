"""
app/core/session.py — Session CRUD backed by MongoDB
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.models.chat import Message, Role, Session


COLLECTION = "sessions"


async def get_or_create_session(
    db: AsyncIOMotorDatabase,
    session_id: str | None = None,
) -> Session:
    if session_id:
        doc = await db[COLLECTION].find_one({"session_id": session_id})
        if doc:
            doc.pop("_id", None)
            messages = [Message(**m) for m in doc.get("messages", [])]
            return Session(
                session_id=doc["session_id"],
                messages=messages,
                created_at=doc.get("created_at", datetime.utcnow()),
                updated_at=doc.get("updated_at", datetime.utcnow()),
                metadata=doc.get("metadata", {}),
            )

    # Create new session
    new_id = session_id or str(uuid.uuid4())
    session = Session(session_id=new_id)
    await db[COLLECTION].insert_one(_to_doc(session))
    return session


async def append_messages(
    db: AsyncIOMotorDatabase,
    session_id: str,
    messages: list[Message],
) -> None:
    await db[COLLECTION].update_one(
        {"session_id": session_id},
        {
            "$push": {"messages": {"$each": [m.model_dump() for m in messages]}},
            "$set": {"updated_at": datetime.utcnow()},
        },
        upsert=True,
    )


async def get_session(
    db: AsyncIOMotorDatabase, session_id: str
) -> Session | None:
    doc = await db[COLLECTION].find_one({"session_id": session_id})
    if not doc:
        return None
    doc.pop("_id", None)
    messages = [Message(**m) for m in doc.get("messages", [])]
    return Session(
        session_id=doc["session_id"],
        messages=messages,
        created_at=doc.get("created_at", datetime.utcnow()),
        updated_at=doc.get("updated_at", datetime.utcnow()),
        metadata=doc.get("metadata", {}),
    )


async def delete_session(
    db: AsyncIOMotorDatabase, session_id: str
) -> bool:
    result = await db[COLLECTION].delete_one({"session_id": session_id})
    return result.deleted_count > 0


def _to_doc(session: Session) -> dict[str, Any]:
    d = session.model_dump()
    d["messages"] = [m.model_dump() for m in session.messages]
    return d
