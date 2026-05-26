#!/usr/bin/env python3
"""
test_api.py
───────────
Integration tests against the running FastAPI backend.
Start the backend first:  uvicorn app.main:app --reload

Tests: /health, /tools, /chat/stream (SSE), /sessions
"""
from __future__ import annotations

import asyncio
import json
import sys
import httpx

BASE = "http://localhost:8000"


def ok(msg):    print(f"  \033[92m✓\033[0m  {msg}")
def fail(msg):  print(f"  \033[91m✗\033[0m  {msg}")
def info(msg):  print(f"  \033[94m→\033[0m  {msg}")
def header(msg):print(f"\n\033[1m{msg}\033[0m")


async def test_health():
    header("1. /health")
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{BASE}/health")
    data = r.json()
    if data.get("mcp", {}).get("connected"):
        ok(f"MCP connected — {data['mcp']['tool_count']} tools")
    else:
        fail(f"MCP not connected: {data}")


async def test_tools():
    header("2. /tools")
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{BASE}/tools")
    tools = r.json()
    ok(f"Found {len(tools)} tools")
    for t in tools:
        info(t["name"])


async def test_stream_chat():
    header("3. /chat/stream  (SSE)")
    payload = {
        "message": (
            "45-year-old postmenopausal female. "
            "Left breast invasive ductal carcinoma T2N1M0. "
            "ER+, PR+, HER2-. Grade 2. Ki-67 18%. "
            "Post-MRM, margins negative, 2/15 nodes positive. "
            "What is the recommended treatment plan?"
        ),
    }
    session_id = None
    text_parts = []
    tool_events = []

    info("Sending clinical query…")
    async with httpx.AsyncClient(timeout=120) as c:
        async with c.stream("POST", f"{BASE}/chat/stream", json=payload) as r:
            async for raw_line in r.aiter_lines():
                if not raw_line.strip():
                    continue
                if raw_line.startswith("event:"):
                    event_type = raw_line[6:].strip()
                elif raw_line.startswith("data:"):
                    data_str = raw_line[5:].strip()
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        data = {"raw": data_str}

                    if event_type == "session":
                        session_id = data.get("session_id")
                        ok(f"Session created: {session_id}")

                    elif event_type == "tool_call_start":
                        tool_events.append(data)
                        info(f"Tool called: {data.get('tool')} — {data.get('args_preview')}")

                    elif event_type == "text_delta":
                        text_parts.append(data.get("chunk", ""))

                    elif event_type == "done":
                        ok(f"Done — {data.get('iteration_count')} iterations, {len(data.get('tool_calls',[]))} tool calls")

                    elif event_type == "error":
                        fail(f"Server error: {data}")

    final = "".join(text_parts)
    print("\n" + "─"*60)
    print(final[:600] + ("…" if len(final) > 600 else ""))
    print("─"*60)

    if tool_events:
        ok(f"Tool calls made: {[t['tool'] for t in tool_events]}")
    else:
        fail("No tool calls made — check LLM is configured")

    return session_id


async def test_session_history(session_id: str | None):
    if not session_id:
        return
    header("4. /sessions/{id}")
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{BASE}/sessions/{session_id}")
    data = r.json()
    msgs = data.get("messages", [])
    ok(f"Session has {len(msgs)} messages")


async def main():
    print("\n\033[1mOncology Backend API Tests\033[0m")
    print(f"Target: {BASE}\n")
    try:
        await test_health()
        await test_tools()
        session_id = await test_stream_chat()
        await test_session_history(session_id)
        print("\n\033[92m✓ All tests passed\033[0m\n")
    except httpx.ConnectError:
        print(f"\n\033[91m✗ Could not connect to {BASE}\033[0m")
        print("  Start the backend first:  uvicorn app.main:app --reload\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
