#!/usr/bin/env python3
"""
test_mcp.py
───────────
Run BEFORE the full app to verify:
  1. MCP server process starts correctly
  2. Tools are discovered
  3. A tool can be called and returns valid output

Usage:
  python test_mcp.py --server /absolute/path/to/server.py

No LLM API key needed — pure MCP test.
"""
from __future__ import annotations

import asyncio
import argparse
import json
import sys

# Allow running from repo root
sys.path.insert(0, ".")

from app.mcp.client import MCPClient
from app.config import Settings


# ── Coloured output helpers ───────────────────────────────────────────────────

def ok(msg):    print(f"  \033[92m✓\033[0m  {msg}")
def fail(msg):  print(f"  \033[91m✗\033[0m  {msg}")
def info(msg):  print(f"  \033[94m→\033[0m  {msg}")
def header(msg):print(f"\n\033[1m{msg}\033[0m")


async def run_tests(server_path: str) -> bool:
    settings = Settings(
        mcp_server_path=server_path,
        mcp_python_bin=sys.executable,
    )

    all_passed = True

    header("1. MCP Server Connection")
    try:
        async with MCPClient(settings) as client:
            ok("MCP server started and session initialised")

            # ── Tool discovery ─────────────────────────────────────────────
            header("2. Tool Discovery")
            tools = await client.list_tools()

            if tools:
                ok(f"Discovered {len(tools)} tools:")
                for t in tools:
                    info(t["function"]["name"] + " — " + t["function"]["description"][:60])
            else:
                fail("No tools found")
                all_passed = False

            # ── Tool execution: breast_cancer ──────────────────────────────
            header("3. Tool Execution — breast_cancer")
            breast_args = {
                "age": 45,
                "sex": "female",
                "ecog": 0,
                "menopausal_status": "postmenopausal",
                "laterality": "left",
                "histology": "invasive ductal carcinoma",
                "tumor_size_cm": 2.0,
                "grade": 2,
                "lvi": False,
                "n_stage": "N1",
                "nodes_examined": 15,
                "nodes_positive": 2,
                "m_stage": "M0",
                "er_status": "positive",
                "pr_status": "positive",
                "her2_status": "negative",
                "t_stage": "T2",
                "overall_stage": "IIA",
                "surgery_done": True,
                "surgery_type": "MRM",
                "margin_status": "negative",
                "ki67_percent": 18,
            }
            info(f"Calling breast_cancer with {len(breast_args)} args…")
            result = await client.call_tool("breast_cancer", breast_args)

            if "CASE SUMMARY" in result or "PRIMARY RECOMMENDATION" in result:
                ok("breast_cancer returned a valid treatment recommendation")
                print("\n" + "─" * 60)
                print(result[:800] + ("…" if len(result) > 800 else ""))
                print("─" * 60)
            else:
                fail("Unexpected output format")
                print(result[:400])
                all_passed = False

            # ── Tool execution: cervix_cancer ──────────────────────────────
            header("4. Tool Execution — cervix_cancer")
            cervix_args = {
                "age": 52,
                "ecog": 0,
                "figo_stage": "IIB",
                "histology": "scc",
                "tumor_size_cm": 4.5,
                "pelvic_nodes_positive": True,
                "para_aortic_nodes_positive": False,
                "hydronephrosis": False,
                "creatinine_clearance": 75.0,
            }
            info(f"Calling cervix_cancer with {len(cervix_args)} args…")
            result2 = await client.call_tool("cervix_cancer", cervix_args)

            if "Locally advanced" in result2 or "chemoradiation" in result2.lower():
                ok("cervix_cancer returned a valid recommendation")
            else:
                fail("Unexpected output")
                print(result2[:300])
                all_passed = False

            # ── Ping ──────────────────────────────────────────────────────
            header("5. Ping")
            alive = await client.ping()
            ok("Ping OK") if alive else fail("Ping failed")

    except FileNotFoundError as exc:
        fail(f"Could not start MCP server: {exc}")
        fail(f"Check MCP_SERVER_PATH: {server_path}")
        all_passed = False
    except Exception as exc:
        fail(f"Unexpected error: {exc}")
        import traceback; traceback.print_exc()
        all_passed = False

    header("Result")
    if all_passed:
        ok("All tests passed — ready to start the full app")
    else:
        fail("Some tests failed — review output above")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MCP connection")
    parser.add_argument(
        "--server",
        default="",
        help="Absolute path to server.py (overrides .env)",
    )
    args = parser.parse_args()

    server_path = args.server
    if not server_path:
        # Try loading from .env
        try:
            from dotenv import load_dotenv
            import os
            load_dotenv()
            server_path = os.getenv("MCP_SERVER_PATH", "")
        except ImportError:
            pass

    if not server_path:
        print("Usage: python test_mcp.py --server /path/to/server.py")
        print("  or set MCP_SERVER_PATH in your .env file")
        sys.exit(1)

    success = asyncio.run(run_tests(server_path))
    sys.exit(0 if success else 1)
