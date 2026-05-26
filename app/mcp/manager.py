"""
app/mcp/manager.py
──────────────────
Singleton that holds the live MCPClient across the FastAPI app lifetime.
Exposed as a dependency: `Depends(get_mcp_client)`.
"""
from __future__ import annotations

import logging
from typing import Any

from app.mcp.client import MCPClient
from app.config import Settings

logger = logging.getLogger(__name__)

_mcp_client: MCPClient | None = None


async def start_mcp(settings: Settings) -> None:
    """Called in FastAPI lifespan startup."""
    global _mcp_client
    _mcp_client = MCPClient(settings)
    await _mcp_client.connect()
    # Warm the tool cache immediately
    await _mcp_client.list_tools()
    logger.info("MCP client ready.")


async def stop_mcp() -> None:
    """Called in FastAPI lifespan shutdown."""
    global _mcp_client
    if _mcp_client:
        await _mcp_client.disconnect()
        _mcp_client = None


def get_mcp_client() -> MCPClient:
    """FastAPI dependency — injects the live client."""
    if _mcp_client is None:
        raise RuntimeError("MCPClient not initialised. Check app startup.")
    return _mcp_client


async def list_tools() -> list[dict[str, Any]]:
    """Convenience shortcut for the orchestrator."""
    return await get_mcp_client().list_tools()


async def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """Convenience shortcut for the orchestrator."""
    return await get_mcp_client().call_tool(name, arguments)
