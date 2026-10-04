"""
Standalone test client for the neuralcad MCP server (no LLM).

Starts server.py over stdio and drives a short session that proves the two
things that matter: (1) state persists across tool calls (step 2 fillets the
box built in step 1), and (2) render_views returns actual image content blocks.
"""

import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client, get_default_environment

# mcp passes only a minimal "safe" env to the server subprocess (no DISPLAY).
# Forward DISPLAY so the server can render to WSLg's X server (:0). On a truly
# headless host, install Xvfb instead and export_as_image auto-starts it.
env = get_default_environment()
if os.environ.get("DISPLAY"):
    env["DISPLAY"] = os.environ["DISPLAY"]

server_params = StdioServerParameters(
    command=sys.executable,  # same interpreter as the client (conda env)
    args=["src/harnesses/cadquery_mcp/server.py"],
    env=env,
)


def show(label, result):
    """Print a tool result, summarizing image blocks instead of dumping base64."""
    print(f"\n=== {label} ===")
    for block in result.content:
        if getattr(block, "type", None) == "image":
            print(f"  [image] {block.mimeType}, {len(block.data)} base64 chars")
        else:
            print(" ", getattr(block, "text", block))


async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Tools:", [t.name for t in tools.tools])

            # 1. build a box
            show("run_step: build box", await session.call_tool(
                "run_step",
                arguments={"code": "result = cq.Workplane('XY').box(30, 20, 10)"},
            ))

            # 2. fillet it -> proves `result` from step 1 is still alive
            show("run_step: fillet (uses previous result)", await session.call_tool(
                "run_step",
                arguments={"code": "result = result.edges('|Z').fillet(2)"},
            ))

            # 3. inspect current dimensions
            show("inspect", await session.call_tool("inspect", arguments={}))

            # 4. render from several sides -> image content blocks
            show("render_views [front, top, toprightiso]", await session.call_tool(
                "render_views",
                arguments={"views": ["front", "top", "toprightiso"]},
            ))

            # 5. undo the fillet -> step count drops
            show("undo", await session.call_tool("undo", arguments={}))


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
