"""Replay store: answers tool calls with responses logged in FinMCP-Bench.

Lookup order: exact match in the current task → exact match in any task → closest call of the same tool
(Jaccard over argument values, paging ignored) → miss. Every lookup reports which of hit / approx / miss it was.
"""
import json
from collections import Counter, defaultdict

from . import finmcp

JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "list": "array", "dict": "object"}


def _norm(v):
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in sorted(v.items())}
    if isinstance(v, list):
        return sorted((_norm(x) for x in v), key=lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        return v.strip()
    return v


def canon(args: dict) -> str:
    core = {k: v for k, v in (args or {}).items() if k not in finmcp.PAGING_ARGS and v is not None}
    return json.dumps(_norm(core), ensure_ascii=False, sort_keys=True)


def flat(args: dict) -> set:
    out = set()
    for k, v in (args or {}).items():
        if k in finmcp.PAGING_ARGS or v is None:
            continue
        for x in (v if isinstance(v, list) else [v]):
            out.add((k, json.dumps(_norm(x), ensure_ascii=False, sort_keys=True)))
    return out


class ReplayStore:
    def __init__(self, splits=("multi_tool", "multi_turn")):
        self.exact = defaultdict(list)           # (tool, canon) -> [(task_id, text)]
        self.by_tool = defaultdict(list)         # tool -> [(task_id, flat_args, text)]
        self.calls = Counter()
        self.arg_stats = defaultdict(lambda: defaultdict(lambda: {"n": 0, "types": Counter(), "values": set()}))
        self.top_keys = {}
        for tid, task in finmcp.iter_tasks(splits):
            for call in finmcp.tool_calls(task):
                tool, args, text = call["name"], call["args"], call["text"]
                self.calls[tool] += 1
                self.exact[(tool, canon(args))].append((tid, text))
                self.by_tool[tool].append((tid, flat(args), text))
                for k, v in args.items():
                    st = self.arg_stats[tool][k]
                    st["n"] += 1
                    st["types"][type(v).__name__] += 1
                    if isinstance(v, str) and len(st["values"]) < 20:
                        st["values"].add(v)
                if tool not in self.top_keys and isinstance(call["json"], dict):
                    self.top_keys[tool] = list(call["json"])[:8]

    def lookup(self, tool, args, task_id=None, min_score=0.5):
        """Returns (status, text, source_task_id, score)."""
        hits = self.exact.get((tool, canon(args)))
        if hits:
            same = [h for h in hits if h[0] == task_id]
            tid, text = (same or hits)[0]
            return "hit", text, tid, 1.0
        want = flat(args)
        best = None
        for tid, have, text in self.by_tool.get(tool, ()):
            union = want | have
            jaccard = len(want & have) / len(union) if union else 1.0
            covered = len(want & have) / len(want) if want else 1.0   # our request is a subset of the logged one
            score = max(jaccard, 0.9 * covered)
            key = (score + (0.25 if tid == task_id else 0), score)   # prefer the current task
            if best is None or key > best[0]:
                best = (key, tid, text)
        if best and best[0][1] >= min_score:
            return "approx", best[2], best[1], round(best[0][1], 3)
        return "miss", None, None, round(best[0][1], 3) if best else 0.0

    def tool_specs(self) -> list:
        """MCP tool definitions inferred from logged calls (FinMCP-Bench ships no schemas)."""
        specs = []
        for tool in sorted(self.calls):
            n, props, required = self.calls[tool], {}, []
            for arg, st in self.arg_stats[tool].items():
                jt = JSON_TYPES.get(st["types"].most_common(1)[0][0], "string")
                prop = {"type": jt}
                if jt == "array":
                    prop["items"] = {}
                desc = f"在日志中出现 {st['n']}/{n} 次"
                if jt == "string" and 0 < len(st["values"]) <= 8:
                    desc += "；取值示例：" + "、".join(sorted(st["values"])[:8])
                prop["description"] = desc
                props[arg] = prop
                if st["n"] / n >= 0.95:
                    required.append(arg)
            keys = self.top_keys.get(tool)
            desc = f"{tool}：返回 JSON" + (f"，顶层字段 {', '.join(keys)}" if keys else "")
            specs.append({"name": tool, "description": desc,
                          "input_schema": {"type": "object", "properties": props, "required": required}})
        return specs
