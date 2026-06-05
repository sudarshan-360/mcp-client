from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.db.mongo import get_db
from app.llm.groq_client import LLMClient
from app.llm.orchestrator_extended import OrchestratorExtended
from app.models.chat import (
    ChatRequest,
    Message,
    Role,
    SSEEvent,
)
from app.core.session_extended import (
    get_or_create_extended_session,
    append_messages_extended,
    store_case_context,
)

logger = logging.getLogger(__name__)

router = APIRouter()

settings = get_settings()

llm_client = LLMClient(settings)
orchestrator = OrchestratorExtended(settings, llm_client)


@router.post("/chat/stream-extended")
async def chat_stream_extended(
    req: ChatRequest,
    db=Depends(get_db),
):
    """
    Enhanced chat endpoint with case context storage.

    Flow:
      1. Load or create extended session
      2. Build user message from request
      3. Run orchestrator (yields events + case context)
      4. Save case context to MongoDB
      5. Stream SSE events
      6. Save messages to session

    After this completes, frontend can use /chat/follow-up endpoint.
    """

    async def event_generator():
        try:
            # -------------------------------------------------------------
            # Load or create session
            # -------------------------------------------------------------

            session = await get_or_create_extended_session(
                db,
                req.session_id,
            )

            logger.info(
                "Extended session started: %s",
                session.session_id,
            )

            # -------------------------------------------------------------
            # Build user message
            # -------------------------------------------------------------

            user_content = (
                f"Cancer type: {req.cancer_type}\n"
                f"Patient data: "
                f"{req.patient_data.model_dump_json(indent=2)}"
            )

            if req.request_context:
                user_content += (
                    f"\n\nAdditional context: "
                    f"{req.request_context}"
                )

            user_msg = Message(
                role=Role.USER,
                content=user_content,
            )

            session.messages.append(user_msg)

            # -------------------------------------------------------------
            # Run orchestrator
            # -------------------------------------------------------------

            final_answer = ""
            case_context_data = None

            async for item in orchestrator.run_with_case_context(
                cancer_type=req.cancer_type,
                patient_data=req.patient_data,
                request_context=req.request_context,
            ):

                # Case context payload
                if (
                    isinstance(item, dict)
                    and item.get("type") == "case_context"
                ):
                    case_context_data = item.get("data")

                    logger.info(
                        "Case context generated for session %s",
                        session.session_id,
                    )

                    continue

                # SSE event
                if isinstance(item, SSEEvent):

                    if item.event.value == "text_delta":
                        final_answer += str(item.data)

                    yield (
                        f"event: {item.event.value}\n"
                        f"data: {item.data}\n\n"
                    )

            # -------------------------------------------------------------
            # Store case context
            # -------------------------------------------------------------

            if case_context_data is not None:

                await store_case_context(
                    db=db,
                    session_id=session.session_id,
                    case_context=case_context_data,
                )

                logger.info(
                    "Stored case context for session %s",
                    session.session_id,
                )

            # -------------------------------------------------------------
            # Save assistant response
            # -------------------------------------------------------------

            if final_answer.strip():

                assistant_msg = Message(
                    role=Role.ASSISTANT,
                    content=final_answer,
                )

                session.messages.append(assistant_msg)

                await append_messages_extended(
                    db,
                    session.session_id,
                    [user_msg, assistant_msg],
                )

            # -------------------------------------------------------------
            # Notify frontend follow-ups are available
            # -------------------------------------------------------------

            follow_up_payload = {
                "session_id": session.session_id,
                "cancer_type": req.cancer_type,
                "message": (
                    "You can now ask follow-up questions "
                    "about this case"
                ),
            }

            yield (
                "event: follow_up_enabled\n"
                f"data: {json.dumps(follow_up_payload)}\n\n"
            )

        except Exception as exc:

            logger.exception(
                "Extended chat stream error"
            )

            error_payload = {
                "error": str(exc),
            }

            yield (
                "event: error\n"
                f"data: {json.dumps(error_payload)}\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )