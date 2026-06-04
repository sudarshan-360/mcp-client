"""
app/api/routes/chat.py — REFACTORED
────────────────────────────────────
Removed: In-memory chat_history (use session DB instead)
Changed: Request format to accept cancer_type + structured patient_data
Simplified: Just call orchestrator.run() once (no loop)
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.db.mongo import get_db
from app.llm.groq_client import LLMClient
from app.llm.orchestrator import Orchestrator
from app.models.chat import (
    ChatRequest,
    Message,
    Role,
)
from app.core.session import get_or_create_session, append_messages

logger = logging.getLogger(__name__)

router = APIRouter()

settings = get_settings()

llm_client = LLMClient(settings)
orchestrator = Orchestrator(settings, llm_client)


@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    db=Depends(get_db),
):
    """
    REFACTORED: Deterministic tool routing with LLM formatting.
    
    Flow:
      1. Load or create session
      2. Call orchestrator.run() → direct tool call
      3. Stream events back (no tool-call loop)
      4. Save messages to session
    """

    async def event_generator():
        try:
            # ─────────────────────────────────────────────────────────────
            # Load or create session
            # ─────────────────────────────────────────────────────────────

            session = await get_or_create_session(db, req.session_id)
            logger.info(f"Session: {session.session_id}")

            # ─────────────────────────────────────────────────────────────
            # Build user message (include request context)
            # ─────────────────────────────────────────────────────────────

            user_content = (
                f"Cancer type: {req.cancer_type}\n"
                f"Patient data: {req.patient_data.model_dump_json(indent=2)}"
            )
            if req.request_context:
                user_content += f"\n\nAdditional context: {req.request_context}"

            user_msg = Message(
                role=Role.USER,
                content=user_content,
            )
            session.messages.append(user_msg)

            # ─────────────────────────────────────────────────────────────
            # Run orchestrator (deterministic, no LLM tool selection)
            # ─────────────────────────────────────────────────────────────

            final_answer = ""

            async for event in orchestrator.run(
                cancer_type=req.cancer_type,
                patient_data=req.patient_data,
                request_context=req.request_context,
            ):
                if event.event.value == "text_delta":
                    final_answer += event.data

                # Convert SSEEvent to SSE text format
                yield (
                    f"event: {event.event.value}\n"
                    f"data: {event.data}\n\n"
                )

            # ─────────────────────────────────────────────────────────────
            # Save assistant response to session
            # ─────────────────────────────────────────────────────────────

            if final_answer.strip():
                assistant_msg = Message(
                    role=Role.ASSISTANT,
                    content=final_answer,
                )
                session.messages.append(assistant_msg)

                # Persist to MongoDB
                await append_messages(
                    db,
                    session.session_id,
                    [user_msg, assistant_msg],
                )

        except Exception as exc:
            logger.error(f"Chat stream error: {exc}", exc_info=True)
            yield (
                f"event: error\n"
                f"data: {json.dumps({'error': str(exc)})}\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


import json  # For error handling