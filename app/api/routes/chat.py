from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.llm.groq_client import LLMClient
from app.llm.orchestrator import Orchestrator
from app.models.chat import (
    ChatRequest,
    Message,
    Role,
)

router = APIRouter()

settings = get_settings()

llm_client = LLMClient(settings)
orchestrator = Orchestrator(settings, llm_client)

# Temporary in-memory chat history
chat_history: list[Message] = []


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):

    async def event_generator():

        try:
            # Save user message in memory
            user_msg = Message(
                role=Role.USER,
                content=req.message,
            )

            chat_history.append(user_msg)

            final_answer = ""

            async for event in orchestrator.run(
                user_message=req.message,
                history=chat_history,
            ):

                if event.event.value == "text_delta":
                    final_answer += event.data

                yield (
                    f"event: {event.event.value}\n"
                    f"data: {event.data}\n\n"
                )

            # Save assistant reply in memory
            if final_answer.strip():

                assistant_msg = Message(
                    role=Role.ASSISTANT,
                    content=final_answer,
                )

                chat_history.append(assistant_msg)

        except Exception as exc:

            yield (
                f"event: error\n"
                f"data: {str(exc)}\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )