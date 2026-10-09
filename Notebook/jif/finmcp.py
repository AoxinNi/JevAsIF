"""FinMCP-Bench access: tasks, trajectories, screening and observed data flow."""
import json
import re
from functools import lru_cache

from .config import DATA_DIR

SPLITS = {"single_tool": "benchmark_single_tool.json", "multi_tool": "benchmark_multi_tool.json",
          "multi_turn": "benchmark_multi_turn.json"}
PAGING_ARGS = {"page", "pageNum", "pageSize", "size", "limit", "offset"}
USER_ID = re.compile(r"当前用户id是\s*([0-9a-zA-Z_-]+)")


@lru_cache(maxsize=None)
def load_split(split: str) -> list:
    return json.loads((DATA_DIR / SPLITS[split]).read_text(encoding="utf-8"))


def iter_tasks(splits=("multi_tool",)):
    for split in splits:
        for i, task in enumerate(load_split(split)):
            yield f"{split}:{i}", task


def get_task(task_id: str) -> dict:
    split, i = task_id.split(":")
    return load_split(split)[int(i)]


def parse_json(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def user_text(task) -> str:
    return "\n".join(m.get("content") or "" for m in task["messages"] if m["role"] == "user")


def first_question(task) -> str:
    return next(m.get("content") or "" for m in task["messages"] if m["role"] == "user")


def user_id(task):
    m = USER_ID.search(user_text(task))
    return m.group(1) if m else None


def reference_answer(task) -> str:
    finals = [m.get("content") for m in task["messages"] if m["role"] == "assistant" and m.get("content")]
    return finals[-1] if finals else ""


def tool_calls(task) -> list:
    """Calls in order, each paired with its response: [{step, id, name, args, text, json}]."""
    responses = {m.get("tool_call_id"): m.get("content") or "" for m in task["messages"] if m["role"] == "tool"}
    calls, step = [], 0
    for m in task["messages"]:
        if m["role"] == "assistant" and m.get("tool_calls"):
            step += 1
            for tc in m["tool_calls"]:
                text = responses.get(tc.get("id"), "")
                calls.append({"step": step, "id": tc.get("id"), "name": tc["function"]["name"],
                              "args": parse_json(tc["function"]["arguments"]) or {}, "text": text, "json": parse_json(text)})
    return calls


def leaves(obj, path="", out=None):
    """All scalar leaves as (path, key, value); list positions are written as [*]."""
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            leaves(v, f"{path}.{k}" if path else k, out)
    elif isinstance(obj, list):
        for v in obj:
            leaves(v, f"{path}[*]", out)
    elif obj is not None:
        out.append((path, path.rsplit(".", 1)[-1].replace("[*]", ""), obj))
    return out


def norm(v) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip()


CODE_LIKE = re.compile(r"^[A-Za-z0-9_.\-]{4,40}$")
DATE_LIKE = re.compile(r"^\d{4}-?\d{2}-?\d{2}")
NUMBER = re.compile(r"^-?\d+(\.\d+)?$")


def identifier_like(value) -> bool:
    """Values that can name an entity (codes, ids); excludes dates, decimals, short numbers and free text."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return False
    s = norm(value)
    if not CODE_LIKE.match(s) or DATE_LIKE.match(s):
        return False
    return not (NUMBER.match(s) and ("." in s or len(s) < 5))


def _trivial(name: str, value) -> bool:
    return name in PAGING_ARGS or not identifier_like(value)


def observed_flows(task, calls=None, context=None) -> list:
    """Upstream-to-downstream value flows in a reference trajectory.

    Each flow: {src_tool, src_path, src_key, dst_tool, dst_arg, value, n_candidates}. A value that also
    appears in ``context`` (by default the user's words) is treated as known, not as coming from a tool.
    ``calls`` restricts the analysis to some calls (e.g. one turn of a dialogue).
    """
    asked = user_text(task) if context is None else context
    flows, seen = [], []           # seen: [(tool, leaves)] of earlier responses
    for call in (tool_calls(task) if calls is None else calls):
        for arg, raw in call["args"].items():
            for v in (raw if isinstance(raw, list) else [raw]):
                if isinstance(v, (dict, list)) or v is None or _trivial(arg, v) or norm(v) in asked:
                    continue
                for tool, lv in reversed(seen):
                    hit = next(((p, k) for p, k, x in lv if norm(x) == norm(v)), None)
                    if hit:
                        siblings = {norm(x) for p, k, x in lv if p == hit[0]}
                        flows.append({"src_tool": tool, "src_path": hit[0], "src_key": hit[1], "dst_tool": call["name"],
                                      "dst_arg": arg, "value": norm(v), "n_candidates": len(siblings)})
                        break
        if call["json"] is not None:
            seen.append((call["name"], leaves(call["json"])))
    return flows


def screen(task, calls=None, context=None) -> dict:
    """Classify each upstream-derived argument: pick one / pick a subset / pick all / reference."""
    groups = {}
    for f in observed_flows(task, calls, context):
        groups.setdefault((f["dst_tool"], f["dst_arg"], f["src_tool"], f["src_path"]), []).append(f)
    kinds = {"pick_one": 0, "pick_subset": 0, "pick_all": 0, "reference": 0}
    decisions = []
    for (dst_tool, dst_arg, src_tool, src_path), fs in groups.items():
        n, k = fs[0]["n_candidates"], len({f["value"] for f in fs})
        kind = "reference" if n <= 1 else "pick_one" if k == 1 else "pick_all" if k >= n else "pick_subset"
        kinds[kind] += 1
        if kind in ("pick_one", "pick_subset"):
            decisions.append({"dst_tool": dst_tool, "dst_arg": dst_arg, "src_tool": src_tool, "src_path": src_path,
                              "chosen": sorted({f["value"] for f in fs}), "n_candidates": n, "kind": kind})
    return {**kinds, "decision": bool(decisions), "decisions": decisions}


def split_turns(task) -> list:
    """Dialogue turns: [{turn, user, history, messages, calls, context, answer}].

    ``history`` is every message before this user message; ``context`` is its text (including earlier
    tool outputs), which the agent already knows when it plans this turn.
    """
    msgs, starts = task["messages"], [i for i, m in enumerate(task["messages"]) if m["role"] == "user"]
    all_calls = tool_calls(task)
    turns = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(msgs)
        part = msgs[start:end]
        ids = {c.get("id") for m in part if m["role"] == "assistant" for c in (m.get("tool_calls") or [])}
        history = msgs[:start]
        finals = [m.get("content") for m in part if m["role"] == "assistant" and m.get("content")]
        turns.append({"turn": k, "user": part[0].get("content") or "", "history": history, "messages": part,
                      "calls": [c for c in all_calls if c["id"] in ids],
                      "context": "\n".join((m.get("content") or "") for m in history + [part[0]]),
                      "answer": finals[-1] if finals else ""})
    return turns
