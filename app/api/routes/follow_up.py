"""
app/api/routes/follow_up.py
───────────────────────────
Endpoint for follow-up questions constrained to the case scope.

Flow:
  POST /chat/follow-up
  {
    "session_id": "xyz",
    "question": "Why HER2 testing for this patient?"
  }
  
  Returns: SSE stream with case-constrained answer
"""
from __future__ import annotations

import asyncio
import json
import logging
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db.mongo import get_db
from app.llm.groq_client import LLMClient
from app.llm.case_aware import CaseAwareLLMClient
from app.core.session_extended import (
    get_extended_session,
    append_messages_extended,
)
from app.models.chat import Message, Role, SSEEvent, SSEEventType

logger = logging.getLogger(__name__)

router = APIRouter()
settings = get_settings()
llm_client = LLMClient(settings)
case_aware = CaseAwareLLMClient(llm_client)


class FollowUpRequest(BaseModel):
    """Request for a follow-up question about the case."""
    
    session_id: str = Field(..., description="Session ID for case context")
    question: str = Field(..., description="Doctor's question about the case")


@router.post("/chat/follow-up")
async def follow_up_question(
    req: FollowUpRequest,
    db=Depends(get_db),
):
    """
    Answer a follow-up question constrained to the case scope.
    
    BEHAVIOR:
      1. Load session + case context
      2. Validate case context exists (must have done initial decision-support)
      3. Stream case-constrained answer via SSE
      4. Save Q&A to session history
    
    Returns:
        SSE stream with follow-up answer
    """

    async def event_generator():
        try:
            # ─────────────────────────────────────────────────────────────
            # Step 1: Load session and validate case context
            # ─────────────────────────────────────────────────────────────

            session = await get_extended_session(db, req.session_id)
            if not session:
                yield SSEEvent(
                    event=SSEEventType.ERROR,
                    data=json.dumps({
                        "error": f"Session not found: {req.session_id}"
                    }),
                ).model_dump_json()
                return

            if not session.case_context:
                yield SSEEvent(
                    event=SSEEventType.ERROR,
                    data=json.dumps({
                        "error": (
                            "No case context found. "
                            "Please complete the initial decision-support query first."
                        )
                    }),
                ).model_dump_json()
                return

            logger.info(
                f"Follow-up question for session {session.session_id} "
                f"({session.case_context.cancer_type})"
            )

            # ─────────────────────────────────────────────────────────────
            # Step 2: Emit start event with case context summary
            # ─────────────────────────────────────────────────────────────

            yield SSEEvent(
                event=SSEEventType.TOOL_CALL_START,
                data=json.dumps({
                    "type": "follow_up_query",
                    "cancer_type": session.case_context.cancer_type,
                    "question": req.question[:100] + ("…" if len(req.question) > 100 else ""),
                }),
            ).model_dump_json() + "\n"

            # ─────────────────────────────────────────────────────────────
            # Step 3: Stream case-constrained answer
            # ─────────────────────────────────────────────────────────────

            final_answer = ""
            buffer = ""

            async for token in case_aware.answer_follow_up(
                question=req.question,
                case_context=session.case_context,
            ):
                final_answer += token
                buffer += token

                if (
                    len(buffer) >= 100
                    or token.endswith(".")
                    or token.endswith("\n")
                    or token.endswith(":")
                ):
                    yield SSEEvent(
                        event=SSEEventType.TEXT_DELTA,
                        data=buffer,
                    ).model_dump_json() + "\n"
                    buffer = ""

                if settings.stream_chunk_delay > 0:
                    await asyncio.sleep(settings.stream_chunk_delay)

            # Flush remaining buffer
            if buffer:
                yield SSEEvent(
                    event=SSEEventType.TEXT_DELTA,
                    data=buffer,
                ).model_dump_json() + "\n"

            # ─────────────────────────────────────────────────────────────
            # Step 4: Save Q&A to session history
            # ─────────────────────────────────────────────────────────────

            user_msg = Message(
                role=Role.USER,
                content=req.question,
            )
            assistant_msg = Message(
                role=Role.ASSISTANT,
                content=final_answer,
            )

            session.messages.append(user_msg)
            session.messages.append(assistant_msg)

            await append_messages_extended(
                db,
                session.session_id,
                [user_msg, assistant_msg],
            )

            # ─────────────────────────────────────────────────────────────
            # Step 5: Emit done event
            # ─────────────────────────────────────────────────────────────

            yield SSEEvent(
                event=SSEEventType.DONE,
                data=json.dumps({
                    "type": "follow_up_complete",
                    "cancer_type": session.case_context.cancer_type,
                    "messages_in_session": len(session.messages),
                    "answer_length": len(final_answer),
                    "model": llm_client.model_name,
                }),
            ).model_dump_json() + "\n"

        except Exception as exc:
            logger.error(f"Follow-up question error: {exc}", exc_info=True)
            yield SSEEvent(
                event=SSEEventType.ERROR,
                data=json.dumps({"error": str(exc)}),
            ).model_dump_json() + "\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )