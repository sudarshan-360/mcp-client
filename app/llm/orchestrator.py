"""
app/llm/orchestrator.py
───────────────────────
Agentic orchestration loop.

Flow:
  1. Build message history from session
  2. Call LLM with MCP tools in context
  3. If LLM wants a tool → call MCP, append result, loop
  4. When LLM produces a final text answer → stream it to the client
  5. Persist updated history to session store

Yields SSEEvent objects that the route handler converts to SSE text.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from app.config import Settings
from app.llm.groq_client import LLMClient
from app.mcp.manager import call_tool, list_tools
from app.models.chat import (
    Message,
    Role,
    SSEEvent,
    SSEEventType,
    ToolCallInfo,
)

logger = logging.getLogger(__name__)

# Prevent token explosion
MAX_TOOL_CONTENT = 250


class Orchestrator:

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
    ) -> None:

        self._settings = settings
        self._llm = llm

    async def run(
        self,
        user_message: str,
        history: list[Message],
    ) -> AsyncIterator[SSEEvent]:

        # ─────────────────────────────────────────────
        # Build conversation
        # ─────────────────────────────────────────────

        working: list[dict[str, Any]] = (
            self._history_to_api_format(history)
        )

        working.append({
            "role": "user",
            "content": user_message,
        })

        # ─────────────────────────────────────────────
        # Load tools
        # ─────────────────────────────────────────────

        tools = await list_tools()

        tool_calls_made: list[ToolCallInfo] = []

        iteration = 0
        max_iter = self._settings.max_tool_iterations

        # ─────────────────────────────────────────────
        # Tool loop
        # ─────────────────────────────────────────────

        while iteration < max_iter:

            iteration += 1

            logger.debug(
                "Orchestrator iteration %d",
                iteration,
            )

            response_msg = await self._llm.chat_with_tools(
                working,
                tools,
            )

            # ─────────────────────────────────────────
            # Final answer (no tool calls)
            # ─────────────────────────────────────────

            if not response_msg.get("tool_calls"):

                working.append({
                    "role": "assistant",
                    "content": response_msg.get("content", ""),
                })

                break

            # ─────────────────────────────────────────
            # Convert tool arguments to JSON STRINGS
            # Required by Groq/OpenAI APIs
            # ─────────────────────────────────────────

            assistant_tool_message = {
                "role": "assistant",
                "content": response_msg.get("content", ""),
                "tool_calls": [],
            }

            for tc in response_msg["tool_calls"]:

                fn = tc["function"]

                arguments_raw = fn["arguments"]

                # MUST be JSON string
                if isinstance(arguments_raw, dict):

                    arguments_str = json.dumps(arguments_raw)

                else:

                    arguments_str = arguments_raw

                assistant_tool_message["tool_calls"].append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": fn["name"],
                        "arguments": arguments_str,
                    },
                })

            # Save assistant tool-call message
            working.append(assistant_tool_message)

            # ─────────────────────────────────────────
            # Execute tool calls
            # ─────────────────────────────────────────

            for tc in response_msg["tool_calls"]:

                fn = tc["function"]

                tool_name = fn["name"]

                arguments_raw = fn["arguments"]

                tc_id = tc["id"]

                # Parse arguments back to dict
                if isinstance(arguments_raw, str):

                    try:

                        arguments = json.loads(arguments_raw)

                    except Exception:

                        arguments = {}

                else:

                    arguments = arguments_raw

                # ─────────────────────────────────────
                # Emit tool start
                # ─────────────────────────────────────

                yield SSEEvent(
                    event=SSEEventType.TOOL_CALL_START,
                    data=json.dumps({
                        "tool": tool_name,
                        "args_preview": _summarise_args(
                            arguments
                        ),
                    }),
                )

                # ─────────────────────────────────────
                # Execute MCP tool
                # ─────────────────────────────────────

                try:

                    result_text = await call_tool(
                        tool_name,
                        arguments,
                    )

                    tool_info = ToolCallInfo(
                        tool_name=tool_name,
                        arguments=arguments,
                        result=result_text,
                    )

                except Exception as exc:

                    result_text = (
                        f"ERROR calling {tool_name}: {exc}"
                    )

                    tool_info = ToolCallInfo(
                        tool_name=tool_name,
                        arguments=arguments,
                        error=str(exc),
                    )

                tool_calls_made.append(tool_info)

                # ─────────────────────────────────────
                # Emit tool result preview
                # ─────────────────────────────────────

                yield SSEEvent(
                    event=SSEEventType.TOOL_CALL_RESULT,
                    data=json.dumps({
                        "tool": tool_name,
                        "result_preview": (
                            result_text[:200] + "…"
                            if len(result_text) > 200
                            else result_text
                        ),
                    }),
                )

                # ─────────────────────────────────────
                # COMPRESS TOOL OUTPUT
                # Prevent TPM/token overflow
                # ─────────────────────────────────────

                tool_content = compress_tool_output(
                    result_text,
                    MAX_TOOL_CONTENT,
                )

                # ─────────────────────────────────────
                # Append tool result
                # ─────────────────────────────────────

                working.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": tool_content,
                })

        # ─────────────────────────────────────────────
        # Max iteration safety
        # ─────────────────────────────────────────────

        else:

            logger.warning(
                "Orchestrator hit max iterations (%d)",
                max_iter,
            )

            working.append({
                "role": "assistant",
                "content": (
                    "I've gathered clinical information "
                    "from the decision-support tools. "
                    "Please ask me to summarise or clarify "
                    "any aspect of the recommendation."
                ),
            })

        # ─────────────────────────────────────────────
        # Stream final answer
        # ─────────────────────────────────────────────

        final_content_parts: list[str] = []

        async for token in self._llm.stream_answer(working):

            final_content_parts.append(token)

            yield SSEEvent(
                event=SSEEventType.TEXT_DELTA,
                data=token,
            )

            if self._settings.stream_chunk_delay > 0:

                await asyncio.sleep(
                    self._settings.stream_chunk_delay
                )

        final_content = "".join(final_content_parts)

        # ─────────────────────────────────────────────
        # Done event
        # ─────────────────────────────────────────────

        yield SSEEvent(
            event=SSEEventType.DONE,
            data=json.dumps({
                "tool_calls": [
                    tc.model_dump()
                    for tc in tool_calls_made
                ],
                "iteration_count": iteration,
                "model": self._llm.model_name,
                "final_message": final_content,
            }),
        )

    # ─────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────

    @staticmethod
    def _history_to_api_format(
        history: list[Message],
    ) -> list[dict[str, Any]]:

        result: list[dict[str, Any]] = []

        for msg in history:

            if msg.role == Role.USER:

                result.append({
                    "role": "user",
                    "content": msg.content,
                })

            elif msg.role == Role.ASSISTANT:

                result.append({
                    "role": "assistant",
                    "content": msg.content,
                })

            elif msg.role == Role.TOOL:

                result.append({
                    "role": "tool",
                    "tool_call_id": (
                        msg.tool_call_id or ""
                    ),
                    "content": msg.content,
                })

        return result


def compress_tool_output(
    text: str,
    limit: int,
) -> str:

    # Remove markdown/code fences
    text = text.replace("```", "")

    # Remove bullets and extra spacing
    lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Skip decorative separators
        if set(line) <= {"-", "="}:
            continue

        lines.append(line)

    # Keep VERY SMALL context
    compact = "\n".join(lines[:12])

    # Hard truncate
    compact = compact[:limit]

    return compact


def _summarise_args(args: Any) -> str:

    if isinstance(args, str):

        try:

            args = json.loads(args)

        except Exception:

            return args[:120]

    important = {
        k: v
        for k, v in args.items()
        if v is not None and k in (
            "age",
            "primary_site",
            "figo_stage",
            "ajcc_stage",
            "overall_stage",
            "t_stage",
            "n_stage",
            "m_stage",
            "histology",
            "er_status",
            "pr_status",
            "her2_status",
            "subtype",
            "who_grade",
            "tumour_type",
        )
    }

    if not important:

        important = {
            k: v
            for k, v in list(args.items())[:3]
            if v is not None
        }

    return ", ".join(
        f"{k}={v}"
        for k, v in important.items()
    )