"""Decision items from FinMCP trajectories, for gate calibration and evaluator checks.

A "pick one" decision in a reference trajectory becomes: the user request, the candidate records found at
the source path of the upstream response, and the value the reference trajectory chose.
"""
import random

from . import finmcp


def _records_at(value, path):
    """Records that hold the leaf at ``path`` (e.g. rows[*].code -> every element of rows)."""
    tokens = path.split(".")
    nodes = [value]
    for token in tokens[:-1]:
        name, stars = token, 0
        while name.endswith("[*]"):
            name, stars = name[:-3], stars + 1
        nxt = []
        for node in nodes:
            v = node.get(name) if isinstance(node, dict) and name else node
            for _ in range(stars):
                v = [x for item in (v or []) for x in (item if isinstance(item, list) else [item])] if isinstance(v, list) else []
            nxt.extend(v if isinstance(v, list) else [v])
        nodes = nxt
    last = tokens[-1]
    if last.endswith("[*]"):                     # list of scalar codes
        name = last[:-3]
        return [x for node in nodes for x in ((node.get(name) if isinstance(node, dict) else node) or [])], None
    return [n for n in nodes if isinstance(n, dict) and last in n], last


def choice_items(splits=("multi_tool",), limit=None, seed=0):
    """[{task_id, request, items, id_key, gold, n}] for pick-one decisions with 2–255 candidates."""
    out = []
    for tid, task in finmcp.iter_tasks(splits):
        sc = finmcp.screen(task)
        if not sc["decision"]:
            continue
        calls = finmcp.tool_calls(task)
        for d in sc["decisions"]:
            if d["kind"] != "pick_one":
                continue
            src = next((c for c in calls if c["name"] == d["src_tool"] and c["json"] is not None
                        and any(finmcp.norm(v) == d["chosen"][0] for _, _, v in finmcp.leaves(c["json"]))), None)
            if src is None:
                continue
            items, id_key = _records_at(src["json"], d["src_path"])
            keys = [finmcp.norm(i[id_key]) if id_key else finmcp.norm(i) for i in items]
            if not 2 <= len(set(keys)) <= 255 or d["chosen"][0] not in keys:
                continue
            out.append({"task_id": tid, "request": finmcp.user_text(task), "items": items, "id_key": id_key,
                        "gold": d["chosen"][0], "n": len(set(keys)), "dst": f"{d['dst_tool']}.{d['dst_arg']}"})
    random.Random(seed).shuffle(out)
    return out[:limit] if limit else out


def decision_items(task, calls=None, context=None, task_id=""):
    """Every judgment decision in a trajectory, with its candidate records and gold choice(s).

    Returns [{task_id, kind, src_tool, src_path, dst_tool, dst_arg, id_key, candidates, gold, n}];
    ``kind`` is pick_one or pick_subset, ``gold`` is a list of chosen identifier values.
    """
    calls = finmcp.tool_calls(task) if calls is None else calls
    out = []
    for d in finmcp.screen(task, calls, context)["decisions"]:
        src = next((c for c in calls if c["name"] == d["src_tool"] and c["json"] is not None
                    and any(finmcp.norm(v) in d["chosen"] for _, _, v in finmcp.leaves(c["json"]))), None)
        if src is None:
            continue
        items, id_key = _records_at(src["json"], d["src_path"])
        keys = [finmcp.norm(i[id_key]) if id_key else finmcp.norm(i) for i in items]
        gold = [g for g in d["chosen"] if g in keys]
        if len(set(keys)) < 2 or not gold:
            continue
        out.append({"task_id": task_id, "kind": d["kind"], "src_tool": d["src_tool"], "src_path": d["src_path"],
                    "dst_tool": d["dst_tool"], "dst_arg": d["dst_arg"], "id_key": id_key, "candidates": items,
                    "gold": gold, "n": len(set(keys))})
    return out
