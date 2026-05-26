"""
app/mcp/client.py
─────────────────
Async MCP client. Spawns server.py as a subprocess, connects via stdio,
then exposes list_tools() and call_tool() for the orchestrator.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.config import Settings

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Wraps the MCP Python SDK into a simple async interface.

    Usage:
        async with MCPClient(settings) as client:
            tools = await client.list_tools()
            result = await client.call_tool("breast_cancer", {...})
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session: ClientSession | None = None
        self._cm = None            # context manager returned by stdio_client
        self._session_cm = None    # context manager returned by ClientSession
        self._tools_cache: list[dict[str, Any]] | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Start the MCP server process and open a session."""
        params = StdioServerParameters(
            command=self._settings.mcp_python_bin,
            args=[self._settings.mcp_server_path],
            env=None,
        )

        logger.info(
            "Starting MCP server: %s %s",
            self._settings.mcp_python_bin,
            self._settings.mcp_server_path,
        )

        self._cm = stdio_client(params)
        read, write = await self._cm.__aenter__()

        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()

        await self._session.initialize()
        logger.info("MCP session initialised ✓")

    async def disconnect(self) -> None:
        """Clean up session and subprocess."""
        if self._session_cm:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if self._cm:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:
                pass
        self._session = None
        logger.info("MCP session closed.")

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.disconnect()

    # ── Public API ────────────────────────────────────────────────────────────

    async def list_tools(self, use_cache: bool = True) -> list[dict[str, Any]]:
        """
        Return list of tool descriptors in OpenAI function-calling format
        so they can be forwarded directly to Groq / OpenRouter.

        Each dict:  { "type": "function", "function": { "name", "description",
                       "parameters": { "type", "properties", "required" } } }
        """
        if use_cache and self._tools_cache is not None:
            return self._tools_cache

        self._require_session()
        response = await self._session.list_tools()

        tools: list[dict[str, Any]] = []
        for t in response.tools:
            # MCP tools carry inputSchema as a JSON-Schema dict
            schema = t.inputSchema if hasattr(t, "inputSchema") else {}

            tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": schema or {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            })

        self._tools_cache = tools
        logger.info("Discovered %d MCP tools", len(tools))
        return tools

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        """
        Invoke a named MCP tool and return its text result.
        Raises ValueError on unknown tools, RuntimeError on execution failure.
        """
        self._require_session()

        logger.info("Calling MCP tool '%s' with %d args", tool_name, len(arguments))

        try:
            result = await self._session.call_tool(tool_name, arguments)
        except Exception as exc:
            raise RuntimeError(f"MCP tool '{tool_name}' failed: {exc}") from exc

        # MCP result is a list of content blocks; concat text blocks
        text_parts: list[str] = []
        for block in result.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])

        output = "\n".join(text_parts) if text_parts else "(no output)"
        logger.debug("Tool '%s' returned %d chars", tool_name, len(output))
        return output

    async def ping(self) -> bool:
        """Return True if the MCP session is alive."""
        try:
            self._require_session()
            await self._session.list_tools()
            return True
        except Exception:
            return False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _require_session(self) -> None:
        if self._session is None:
            raise RuntimeError(
                "MCP session is not open. Call connect() or use as context manager."
            )
