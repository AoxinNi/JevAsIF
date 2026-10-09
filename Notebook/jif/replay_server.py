"""MCP server (stdio) that replays FinMCP-Bench tool responses.

Run: python -m jif.replay_server [--splits multi_tool,multi_turn]
Agent-facing tools are the 56 FinMCP tools. Two control tools are for the harness only and are hidden by
the client: _replay_set_task (prefer responses from one task) and _replay_log (return and clear call log).
"""
import argparse
import json

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from .replay import ReplayStore


def make_server(store: ReplayStore) -> Server:
    state = {"task_id": None, "log": []}
    specs = store.tool_specs()
    tools = [types.Tool(name=s["name"], description=s["description"], inputSchema=s["input_schema"]) for s in specs]
    tools += [
        types.Tool(name="_replay_set_task", description="harness only", inputSchema={
            "type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}),
        types.Tool(name="_replay_log", description="harness only", inputSchema={"type": "object", "properties": {}}),
    ]
    known = {s["name"] for s in specs}

    def text(payload, error=False):
        body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return types.CallToolResult(content=[types.TextContent(type="text", text=body)], isError=error)

    async def list_tools(ctx, params):
        return types.ListToolsResult(tools=tools)

    async def call_tool(ctx, params):
        name, args = params.name, dict(params.arguments or {})
        if name == "_replay_set_task":
            state["task_id"], state["log"] = args.get("task_id"), []
            return text({"ok": True, "task_id": state["task_id"]})
        if name == "_replay_log":
            log, state["log"] = state["log"], []
            return text(log)
        if name not in known:
            return text({"error": "UNKNOWN_TOOL", "message": f"没有名为 {name} 的工具"}, error=True)
        status, body, source, score = store.lookup(name, args, state["task_id"])
        state["log"].append({"tool": name, "args": args, "status": status, "source": source, "score": score})
        if status == "miss":
            return text({"error": "REPLAY_MISS", "message": "这组参数没有记录到返回结果"}, error=True)
        return text(body)

    return Server("finmcp-replay", on_list_tools=list_tools, on_call_tool=call_tool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="multi_tool,multi_turn")
    splits = tuple(s for s in ap.parse_args().splits.split(",") if s)
    server = make_server(ReplayStore(splits))

    async def run():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(run)


if __name__ == "__main__":
    main()
