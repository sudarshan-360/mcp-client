"""
app/llm/groq_client.py
──────────────────────
Groq SDK wrapper with OpenRouter fallback.
Both use the OpenAI-compatible chat completions API,
so the same call signature works for both.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx
from openai import AsyncOpenAI          # Groq uses openai-compatible SDK
from groq import AsyncGroq

from app.config import Settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert oncology clinical decision support assistant.

Your job is to collect the clinical parameters needed to call the appropriate
oncology decision-support tool, then present the tool's recommendation clearly.

Available cancer sites and their tools:
- Cervical cancer         → cervix_cancer
- Head & Neck SCC         → hnscc_decision
- Breast cancer           → breast_cancer
- Prostate cancer         → gu_prostate
- Bladder cancer          → gu_bladder
- Testicular cancer       → gu_testicular
- GI cancers (esophagus/stomach/rectum/anal/pancreas/colon) → gi_cancer
- Lymphoma                → lymphoma
- CNS tumours             → cns_tumor

Rules:
1. If the user's query already contains enough parameters, call the tool immediately.
2. If parameters are missing, ask for them clearly and specifically.
3. After receiving tool output, present it clearly with key treatment highlights.
4. Always end with: "⚠ This is decision-support only. All recommendations require clinical judgment and MDT discussion where indicated."
5. Never invent clinical parameters — only use what the user provides.
"""


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._groq: AsyncGroq | None = None
        self._openrouter: AsyncOpenAI | None = None

        if settings.use_groq:
            self._groq = AsyncGroq(api_key=settings.groq_api_key)
            logger.info("LLM: using Groq model '%s'", settings.groq_model)

        elif settings.use_openrouter:
            self._openrouter = AsyncOpenAI(
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
            )
            logger.info("LLM: using OpenRouter model '%s'", settings.openrouter_model)
        else:
            logger.warning("No LLM API key configured — tool calls will fail.")

    @property
    def model_name(self) -> str:
        if self._groq:
            return self._settings.groq_model
        return self._settings.openrouter_model

    # ── Non-streaming (tool-call loop) ────────────────────────────────────────

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Single non-streaming call.  Returns the raw message dict from the API
        (may contain tool_calls or plain content).
        """
        full_messages = [{"role": "system", "content": _SYSTEM_PROMPT}] + messages

        if self._groq:
            response = await self._groq.chat.completions.create(
                model=self._settings.groq_model,
                messages=full_messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                max_tokens=512,
                temperature=0.1,
            )
        elif self._openrouter:
            response = await self._openrouter.chat.completions.create(
                model=self._settings.openrouter_model,
                messages=full_messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                max_tokens=512,
                temperature=0.1,
            )
        else:
            raise RuntimeError("No LLM client configured.")

        choice = response.choices[0]
        msg = choice.message

        # Normalise to plain dict for the orchestrator
        result: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [],
        }

        if msg.tool_calls:
            for tc in msg.tool_calls:

                result["tool_calls"].append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,

                        # IMPORTANT:
                        # Keep arguments as RAW JSON STRING
                        # Do NOT json.loads() here
                        "arguments": tc.function.arguments,
                    },
                })

        return result

    # ── Streaming (final answer only, after tool-call loop) ──────────────────

    async def stream_answer(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncIterator[str]:
        """
        Stream the final assistant answer token-by-token.
        Call this AFTER the tool-call loop has finished and
        `messages` contains all tool results.
        """
        full_messages = [{"role": "system", "content": _SYSTEM_PROMPT}] + messages

        if self._groq:
            stream = await self._groq.chat.completions.create(
                model=self._settings.groq_model,
                messages=full_messages,
                stream=True,
                max_tokens=512,
                temperature=0.1,
            )
        elif self._openrouter:
            stream = await self._openrouter.chat.completions.create(
                model=self._settings.openrouter_model,
                messages=full_messages,
                stream=True,
                max_tokens=512,
                temperature=0.1,
            )
        else:
            raise RuntimeError("No LLM client configured.")

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
