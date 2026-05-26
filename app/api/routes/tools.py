"""
app/api/routes/tools.py — list all available MCP tools
"""
from fastapi import APIRouter, Depends
from app.mcp.manager import get_mcp_client
from app.models.chat import ToolInfo

router = APIRouter()


@router.get("/tools", response_model=list[ToolInfo])
async def list_tools(mcp=Depends(get_mcp_client)):
    """Return all MCP tool descriptors."""
    raw = await mcp.list_tools()
    return [
        ToolInfo(
            name=t["function"]["name"],
            description=t["function"]["description"],
            input_schema=t["function"].get("parameters", {}),
        )
        for t in raw
    ]
