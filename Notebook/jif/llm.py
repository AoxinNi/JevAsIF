"""OpenAI-compatible chat client with a disk cache and a shared cost ledger."""
import hashlib
import json
import time

from .config import CACHE


class Ledger:
    """Accounting for LLM, Jev and tool calls. Use ``mark()``/``since(mark)`` to slice one task."""

    def __init__(self):
        self.records = []

    def add(self, kind, **fields):
        self.records.append({"kind": kind, **fields})

    def mark(self):
        return len(self.records)

    def since(self, mark=0):
        return self.records[mark:]

    @staticmethod
    def summarize(records):
        out = {}
        for r in records:
            s = out.setdefault(r["kind"], {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency": 0.0, "cached": 0})
            s["calls"] += 1
            s["input_tokens"] += r.get("input_tokens", 0) or 0
            s["output_tokens"] += r.get("output_tokens", 0) or 0
            s["latency"] += r.get("latency", 0.0) or 0.0
            s["cached"] += bool(r.get("cached"))
        return out


class LLM:
    def __init__(self, base_url, api_key, model, ledger=None, temperature=0.0, max_tokens=4096, timeout=180,
                 use_cache=True):
        from openai import OpenAI
        if not (api_key and model):
            raise ValueError("请在 keys.py 里填写 LLM_API_KEY 和 LLM_MODEL")
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.ledger = ledger or Ledger()
        self.cache_dir = CACHE / "llm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache

    @classmethod
    def from_keys(cls, ledger=None, **kwargs):
        """Client configured from keys.py (or environment variables): ``LLM.from_keys()``."""
        from .config import load_keys
        k = load_keys()
        return cls(k["LLM_BASE_URL"], k["LLM_API_KEY"], k["LLM_MODEL"], ledger=ledger, **kwargs)

    def list_models(self):
        return [m.id for m in self.client.models.list().data]

    def chat(self, messages, tools=None, tag="", json_mode=False) -> dict:
        """Returns {message: {role, content, tool_calls}, usage: {...}, latency, cached}."""
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature,
                   "max_tokens": self.max_tokens}
        if tools:
            payload["tools"] = tools
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        key = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:32]
        path = self.cache_dir / f"{key}.json"
        if self.use_cache and path.exists():
            out = json.loads(path.read_text(encoding="utf-8"))
            out["cached"] = True
        else:
            t0 = time.perf_counter()
            resp = self.client.chat.completions.create(**payload)
            latency = time.perf_counter() - t0
            msg = resp.choices[0].message
            calls = [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                     for c in (msg.tool_calls or [])]
            usage = resp.usage
            out = {"message": {"role": "assistant", "content": msg.content or "", "tool_calls": calls},
                   "usage": {"input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                             "output_tokens": getattr(usage, "completion_tokens", 0) or 0},
                   "latency": latency, "cached": False}
            if self.use_cache:
                path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        self.ledger.add("llm", tag=tag, latency=out["latency"], cached=out["cached"], **out["usage"])
        return out
