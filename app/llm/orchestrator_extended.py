"""
app/llm/orchestrator_extended.py
─────────────────────────────────
Enhanced orchestrator that:
  1. Calls MCP tool (same as before)
  2. Formats with LLM (same as before)
  3. ALSO stores case context for multi-turn conversations

The case context is used for follow-up questions that should be
constrained to the specific case.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from app.config import Settings
from app.llm.groq_client import LLMClient
from app.mcp.manager import call_tool
from app.core.tool_router import get_tool_name, validate_cancer_type
from app.models.chat import (
    ClinicalParameters,
    SSEEvent,
    SSEEventType,
)
from app.models.session import CaseContextData

logger = logging.getLogger(__name__)


class OrchestratorExtended:
    """
    Enhanced orchestrator for multi-turn conversations.
    
    Differences from base Orchestrator:
      - Yields additional "case_context_ready" event after initial response
      - Returns case context data for storage
    
    Usage:
      async for event in orchestrator.run_with_case_context(...):
          yield event  # Includes case_context_ready event
          
      # Case context is now stored in session for follow-ups
    """

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
    ) -> None:
        self._settings = settings
        self._llm = llm

    async def run_with_case_context(
        self,
        cancer_type: str,
        patient_data: dict[str, Any] | ClinicalParameters,
        request_context: str | None = None,
    ) -> AsyncIterator[SSEEvent | dict[str, Any]]:
        """
        Run decision-support AND prepare case context for follow-ups.
        
        Args:
            cancer_type: Type of cancer (e.g., "breast", "cervical")
            patient_data: Structured clinical parameters
            request_context: Optional additional context for LLM
        
        Yields:
            SSEEvent objects (for streaming)
            AND a final dict containing case context (for storage)
            
        The final yielded dict has type "case_context" and contains:
            {
                "type": "case_context",
                "data": CaseContextData instance
            }
        """

        # ───────────────────────────────────────────────────────────────────────
        # Step 1: Validate cancer type and get tool name
        # ───────────────────────────────────────────────────────────────────────

        try:
            validated_cancer_type = validate_cancer_type(cancer_type)
            logger.info(f"Cancer type validated: {validated_cancer_type}")
        except ValueError as e:
            yield SSEEvent(
                event=SSEEventType.ERROR,
                data=json.dumps({"error": str(e)}),
            )
            return

        try:
            tool_name = get_tool_name(validated_cancer_type)
            logger.info(f"Determined tool: {tool_name}")
        except ValueError as e:
            yield SSEEvent(
                event=SSEEventType.ERROR,
                data=json.dumps({"error": str(e)}),
            )
            return

        # ───────────────────────────────────────────────────────────────────────
        # Step 2: Prepare arguments for MCP tool
        # ───────────────────────────────────────────────────────────────────────

        if isinstance(patient_data, ClinicalParameters):
            arguments = patient_data.model_dump(exclude_none=True)
        else:
            arguments = {k: v for k, v in patient_data.items() if v is not None}

        logger.debug(f"Tool arguments: {len(arguments)} fields")

        # ───────────────────────────────────────────────────────────────────────
        # Step 3: Call MCP tool
        # ───────────────────────────────────────────────────────────────────────

        yield SSEEvent(
            event=SSEEventType.TOOL_CALL_START,
            data=json.dumps({
                "tool": tool_name,
                "cancer_type": cancer_type,
                "param_count": len(arguments),
            }),
        )

        tool_output = ""
        tool_error = None

        try:
            logger.info(f"Calling MCP tool: {tool_name}")
            tool_output = await call_tool(tool_name, arguments)
            logger.debug(f"Tool returned {len(tool_output)} chars")

        except Exception as exc:
            tool_error = str(exc)
            logger.error(f"Tool execution failed: {exc}")
            yield SSEEvent(
                event=SSEEventType.ERROR,
                data=json.dumps({
                    "error": f"Tool execution failed: {exc}",
                    "tool": tool_name,
                }),
            )
            return

        # ───────────────────────────────────────────────────────────────────────
        # Step 4: Emit tool result
        # ───────────────────────────────────────────────────────────────────────

        yield SSEEvent(
            event=SSEEventType.TOOL_CALL_RESULT,
            data=json.dumps({
                "tool": tool_name,
                "result_preview": (
                    tool_output[:300] + "…"
                    if len(tool_output) > 300
                    else tool_output
                ),
            }),
        )

        # ───────────────────────────────────────────────────────────────────────
        # Step 5: Stream LLM-formatted explanation
        # ───────────────────────────────────────────────────────────────────────

        final_content_parts: list[str] = []
        buffer = ""

        try:
            async for token in self._llm.format_tool_output(
                tool_output=tool_output,
                cancer_type=cancer_type,
                patient_context=request_context or "",
            ):
                final_content_parts.append(token)
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
                    )
                    buffer = ""

                if self._settings.stream_chunk_delay > 0:
                    await asyncio.sleep(self._settings.stream_chunk_delay)

            if buffer:
                yield SSEEvent(
                    event=SSEEventType.TEXT_DELTA,
                    data=buffer,
                )
        except Exception as exc:
            logger.error(f"LLM formatting failed: {exc}")
            yield SSEEvent(
                event=SSEEventType.TEXT_DELTA,
                data=tool_output,
            )
            final_content_parts = [tool_output]

        # ───────────────────────────────────────────────────────────────────────
        # Step 6: Create and yield case context for storage
        # ───────────────────────────────────────────────────────────────────────

        final_content = "".join(final_content_parts)

        case_context = CaseContextData(
            cancer_type=cancer_type,
            patient_data=arguments,
            request_context=request_context,
            initial_tool_output=tool_output,
            initial_llm_response=final_content,
            tool_name=tool_name,
            parameters_sent=len(arguments),
        )

        # Yield special marker so caller knows case context is ready
        yield {
            "type": "case_context",
            "data": case_context,
        }

        # ───────────────────────────────────────────────────────────────────────
        # Step 7: Final done event with summary
        # ───────────────────────────────────────────────────────────────────────

        yield SSEEvent(
            event=SSEEventType.DONE,
            data=json.dumps({
                "tool_name": tool_name,
                "cancer_type": cancer_type,
                "parameters_sent": len(arguments),
                "tool_output_chars": len(tool_output),
                "formatted_output_chars": len(final_content),
                "error": tool_error,
                "model": self._llm.model_name,
                "case_context_ready": True,
            }),
        )