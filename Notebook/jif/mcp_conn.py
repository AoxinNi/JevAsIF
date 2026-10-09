"""Synchronous MCP client that runs the async session in a background thread.

Works the same in scripts and in Jupyter (whose own event loop is already running).
"""
import asyncio
import json
import sys
import threading
import time

from mcp import ClientSession, StdioServerParameters, stdio_client

from .config import CACHE, ROOT


def _get(obj, *names):
    return next((getattr(obj, n) for n in names if hasattr(obj, n)), None)


class McpConnection:
    def __init__(self, command=None, args=None, env=None, cwd=None, log_name="mcp_server.log", timeout=60):
        self.params = StdioServerParameters(command=command or sys.executable,
                                            args=args if args is not None else ["-m", "jif.replay_server"],
                                            env=env, cwd=str(cwd or ROOT))
        self.timeout = timeout
        self._errlog = open(CACHE / log_name, "a", encoding="utf-8")
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._ready, self._session, self._stop, self._error = threading.Event(), None, None, None
        self._runner = asyncio.run_coroutine_threadsafe(self._run(), self._loop)
        if not self._ready.wait(timeout):
            raise TimeoutError("MCP 服务启动超时，日志见 " + str(CACHE / log_name))
        if self._error:
            raise RuntimeError(f"MCP 服务启动失败：{self._error}")

    async def _run(self):
        self._stop = asyncio.Event()
        try:
            async with stdio_client(self.params, errlog=self._errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._stop.wait()
        except Exception as exc:  # surfaced to the constructor
            self._error = repr(exc)
            self._ready.set()

    def _call(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(self.timeout)

    def list_tools(self, include_hidden=False) -> list:
        res = self._call(self._session.list_tools())
        tools = [{"name": t.name, "description": t.description or "",
                  "input_schema": _get(t, "input_schema", "inputSchema") or {"type": "object"}} for t in res.tools]
        return [t for t in tools if include_hidden or not t["name"].startswith("_")]

    def call(self, name, args=None) -> dict:
        """Returns {ok, text, json, latency}."""
        t0 = time.perf_counter()
        res = self._call(self._session.call_tool(name, args or {}))
        text = "".join(getattr(c, "text", "") for c in res.content)
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        return {"ok": not _get(res, "is_error", "isError"), "text": text, "json": parsed,
                "latency": time.perf_counter() - t0}

    # harness helpers for the replay server
    def set_task(self, task_id):
        return self.call("_replay_set_task", {"task_id": task_id})

    def replay_log(self) -> list:
        return self.call("_replay_log")["json"] or []

    def close(self):
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
            try:
                self._runner.result(10)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)
        self._errlog.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
