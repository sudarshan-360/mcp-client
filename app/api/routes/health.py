"""
app/api/routes/health.py
"""
from fastapi import APIRouter, Depends
from app.mcp.manager import get_mcp_client

router = APIRouter()


@router.get("/health")
async def health(mcp=Depends(get_mcp_client)):
    mcp_ok = await mcp.ping()
    tools = await mcp.list_tools() if mcp_ok else []
    return {
        "status": "ok" if mcp_ok else "degraded",
        "mcp": {
            "connected": mcp_ok,
            "tool_count": len(tools),
        },
    }
