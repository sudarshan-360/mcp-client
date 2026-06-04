"""
app/llm/groq_client.py — REFACTORED
────────────────────────────────────
Removed: Tool-selection logic (chat_with_tools)
Kept: stream_answer (now used for formatting tool output)
Added: format_tool_output (specialized for explaining MCP results)
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from groq import AsyncGroq
from openai import AsyncOpenAI

from app.config import Settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# System prompts (specialized)
# ──────────────────────────────────────────────────────────────────────────────

_FORMAT_SYSTEM_PROMPT = """You are a medical report formatter.

CRITICAL RULES:

1. Preserve every line exactly.
2. Preserve every recommendation exactly.
3. Preserve every dose exactly.
4. Preserve every stage exactly.
5. Preserve every protocol exactly.
6. Do not summarize.
7. Do not explain.
8. Do not infer.
9. Do not rewrite clinical content.
10. Convert only to markdown formatting.

Return the content with identical medical meaning and wording.
"""


class LLMClient:
    """
    REFACTORED: No tool selection. Only for formatting/explaining tool output.
    """

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
            logger.warning("No LLM API key configured — formatting will fail.")

    @property
    def model_name(self) -> str:
        if self._groq:
            return self._settings.groq_model
        return self._settings.openrouter_model

    # ── Format tool output for presentation ────────────────────────────────────

    async def format_tool_output(
        self,
        tool_output: str,
        cancer_type: str,
        patient_context: str = "",
    ) -> AsyncIterator[str]:
        """
        Stream formatted explanation of the MCP tool output.
        
        Args:
            tool_output: Raw output from MCP tool (treatment recommendation)
            cancer_type: Type of cancer (for context)
            patient_context: Optional additional clinical context
        
        Yields:
            Formatted tokens one by one
        """
        
        messages = [
            {
                "role": "user",
                "content": (
                    f"Please present this {cancer_type} cancer decision-support output "
                    f"clearly and professionally.\n\n"
                    f"Tool output:\n{tool_output}"
                    + (f"\n\nAdditional context: {patient_context}" if patient_context else "")
                ),
            }
        ]

        if self._groq:
            stream = await self._groq.chat.completions.create(
                model=self._settings.groq_model,
                messages=[
                    {"role": "system", "content": _FORMAT_SYSTEM_PROMPT}
                ] + messages,
                stream=True,
                max_tokens=512,
                temperature=0.1,
            )
        elif self._openrouter:
            stream = await self._openrouter.chat.completions.create(
                model=self._settings.openrouter_model,
                messages=[
                    {"role": "system", "content": _FORMAT_SYSTEM_PROMPT}
                ] + messages,
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

    # ── Fallback: Direct streaming (if LLM fails) ──────────────────────────────

    async def stream_answer(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncIterator[str]:
        """
        General streaming for any message list.
        Used as fallback if format_tool_output needs more control.
        """
        if self._groq:
            stream = await self._groq.chat.completions.create(
                model=self._settings.groq_model,
                messages=[
                    {"role": "system", "content": _FORMAT_SYSTEM_PROMPT}
                ] + messages,
                stream=True,
                max_tokens=512,
                temperature=0.1,
            )
        elif self._openrouter:
            stream = await self._openrouter.chat.completions.create(
                model=self._settings.openrouter_model,
                messages=[
                    {"role": "system", "content": _FORMAT_SYSTEM_PROMPT}
                ] + messages,
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