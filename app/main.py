"""
app/main.py — FastAPI application entry point
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.db.mongo import close_db, connect_db
from app.mcp.manager import start_mcp, stop_mcp
from app.api.routes import chat, health, tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start/stop background services around the app lifetime."""
    settings = get_settings()

    logger.info("▶ Starting Oncology Decision Support Backend")

    # Connect MongoDB (non-fatal if unavailable in dev)
    try:
        await connect_db(settings)
    except Exception as exc:
        logger.warning("MongoDB unavailable (%s) — sessions will be in-memory only", exc)

    # Connect MCP server
    await start_mcp(settings)

    yield   # ← app is live

    logger.info("■ Shutting down…")
    await stop_mcp()
    await close_db()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Oncology Decision Support API",
        description=(
            "AI-powered clinical decision support for 9 cancer sites. "
            "Decision-support only — all outputs require clinical judgment."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS (Next.js dev server + production domain)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(health.router, tags=["system"])
    app.include_router(tools.router, tags=["tools"])
    app.include_router(chat.router, tags=["chat"])

    return app


app = create_app()
