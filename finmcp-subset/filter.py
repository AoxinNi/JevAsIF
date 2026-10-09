"""Screen FinMCP-Bench for tasks suited to graph-guided tool compilation (Jev-as-IF).

Run:  python finmcp-subset/filter.py
Writes into this folder: tasks.jsonl, index.csv, decisions.jsonl, excluded.csv, tools.json, review.csv, stats.json

Units
- multi_tool: one unit per task.
- multi_turn: one unit per dialogue turn; earlier messages (including earlier tool outputs) are known context.
- single_tool: excluded (one call, no chain); its tools are still listed in tools.json.

Groups
- D  at least one judgment decision: an argument picked (one or a subset) among several upstream candidates
- M  dependencies exist but all are mechanical (direct reference or taking every candidate)
- I  two or more calls, all independent of each other
Excluded: fewer than two calls, or a call that needs generated content (rendering HTML/charts).
multi_turn turns in group I are excluded too: calls within such a turn are retries after errors or reuse values
already known from earlier turns, so there is no chain to compile.
"""
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Notebook"))
sys.path.insert(0, str(HERE))
from jif import calib, finmcp  # noqa: E402

import labels  # noqa: E402

GEN_TOOLS = {"RenderHtmlToPdf", "RenderEchart"}
DEV_SHARE, SEED = 0.2, 2026
GROUPS = {"D": "含语义判断", "M": "只有机械依赖", "I": "无依赖"}


def classify(sc, calls):
    names = {c["name"] for c in calls}
    if len(calls) < 2:
        return None, "工具调用少于 2 次，没有调用链"
    if names & GEN_TOOLS:
        return None, "含需要生成内容的调用（" + "、".join(sorted(names & GEN_TOOLS)) + "）"
    if sc["decision"]:
        return "D", ""
    if sc["pick_all"] + sc["reference"]:
        return "M", ""
    return "I", ""


def identity(task, calls):
    uid = finmcp.user_id(task)
    return uid or next((str(c["args"]["userId"]) for c in calls if c["args"].get("userId")), None)


def flags_for(calls, user_id):
    flags = []
    if any(c["json"] is None or '"error"' in (c["text"] or "")[:300] for c in calls):
        flags.append("参考轨迹含报错或非JSON返回")
    if not user_id:
        flags.append("不涉及用户身份")
    return flags


def steps(messages):
    return sum(1 for m in messages if m["role"] == "assistant" and m.get("tool_calls"))


def parallel(messages):
    return any(len(m.get("tool_calls") or []) > 1 for m in messages if m["role"] == "assistant")


def summarize_decisions(sc):
    return [{"kind": d["kind"], "src": f"{d['src_tool']}.{d['src_path']}", "dst": f"{d['dst_tool']}.{d['dst_arg']}",
             "n_candidates": d["n_candidates"], "chosen": d["chosen"]} for d in sc["decisions"]]


def unit(uid, source, task, group, calls, sc, *, question, messages, history, answer, user_id, dialogue=None, turn=None):
    return {"id": uid, "source": source, "unit": "turn" if turn is not None else "task", "dialogue_id": dialogue,
            "turn": turn, "group": group, "group_name": GROUPS[group], "split": None,
            "question": question, "user_id": user_id, "history": history, "messages": messages,
            "reference_answer": answer, "scenery": task["scenery"], "difficulty": task["diff_level"].get("level"),
            "tools": [c["name"] for c in calls], "n_calls": len(calls), "n_steps": steps(messages),
            "has_parallel": parallel(messages), "flags": flags_for(calls, user_id),
            "screen": {k: sc[k] for k in ("pick_one", "pick_subset", "pick_all", "reference")},
            "decisions": summarize_decisions(sc)}


def collect():
    kept, excluded, decisions = [], [], []
    turn_counts = Counter()
    for tid, task in finmcp.iter_tasks(("single_tool",)):
        excluded.append({"id": tid, "source": "single_tool", "unit": "task", "reason": "单工具子集：只有一次调用",
                         "n_calls": len(finmcp.tool_calls(task)), "question": finmcp.first_question(task)[:80]})
    for tid, task in finmcp.iter_tasks(("multi_tool",)):
        calls = finmcp.tool_calls(task)
        sc = finmcp.screen(task)
        group, reason = classify(sc, calls)
        if not group:
            excluded.append({"id": tid, "source": "multi_tool", "unit": "task", "reason": reason, "n_calls": len(calls),
                             "question": finmcp.first_question(task)[:80]})
            continue
        kept.append(unit(tid, "multi_tool", task, group, calls, sc, question=finmcp.first_question(task),
                         messages=task["messages"], history=[], answer=finmcp.reference_answer(task),
                         user_id=identity(task, calls)))
        decisions += calib.decision_items(task, calls, None, tid)
    for tid, task in finmcp.iter_tasks(("multi_turn",)):
        uid_dialogue = identity(task, finmcp.tool_calls(task))
        for turn in finmcp.split_turns(task):
            uid, calls = f"{tid}#t{turn['turn']}", turn["calls"]
            turn_counts["all"] += 1
            if len(calls) < 2:
                turn_counts["fewer_than_2_calls"] += 1
                continue
            sc = finmcp.screen(task, calls, turn["context"])
            group, reason = classify(sc, calls)
            if group == "I":
                group, reason = None, "轮内调用彼此独立（多为报错后重试，或复用前几轮已知的值）"
            if not group:
                excluded.append({"id": uid, "source": "multi_turn", "unit": "turn", "reason": reason, "n_calls": len(calls),
                                 "question": turn["user"][:80]})
                continue
            kept.append(unit(uid, "multi_turn", task, group, calls, sc, question=turn["user"], messages=turn["messages"],
                             history=turn["history"], answer=turn["answer"], user_id=uid_dialogue, dialogue=tid,
                             turn=turn["turn"]))
            decisions += calib.decision_items(task, calls, turn["context"], uid)
    return kept, excluded, decisions, turn_counts


def assign_splits(kept):
    """Stratified 20/80 dev/test. Turns of one dialogue always share a split (no leakage)."""
    rng = random.Random(SEED)
    blocks = defaultdict(list)
    for r in kept:
        key = r["dialogue_id"] or r["id"]
        blocks[(r["source"], r["group"] if r["unit"] == "task" else "dialogue")].append(key)
    dev = set()
    for (_, _), keys in sorted(blocks.items()):
        keys = sorted(set(keys))
        rng.shuffle(keys)
        dev.update(keys[: math.ceil(len(keys) * DEV_SHARE)])
    for r in kept:
        r["split"] = "dev" if (r["dialogue_id"] or r["id"]) in dev else "test"


def tool_table(kept):
    stats = {}
    for r in kept:
        calls = finmcp.tool_calls({"messages": r["history"] + r["messages"]})
        ids = {c.get("id") for m in r["messages"] if m["role"] == "assistant" for c in (m.get("tool_calls") or [])}
        for c in calls:
            if c["id"] not in ids:
                continue
            t = stats.setdefault(c["name"], {"name": c["name"], "calls": 0, "groups": Counter(), "sources": Counter(),
                                             "args": defaultdict(lambda: {"count": 0, "types": Counter(), "examples": []})})
            t["calls"] += 1
            t["groups"][r["group"]] += 1
            t["sources"][r["source"]] += 1
            for k, v in c["args"].items():
                a = t["args"][k]
                a["count"] += 1
                a["types"][type(v).__name__] += 1
                ex = json.dumps(v, ensure_ascii=False)[:60]
                if ex not in a["examples"] and len(a["examples"]) < 5:
                    a["examples"].append(ex)
    used = sorted(stats.values(), key=lambda t: -t["calls"])
    for t in used:
        t["groups"], t["sources"] = dict(t["groups"]), dict(t["sources"])
        t["args"] = {k: {"count": v["count"], "types": dict(v["types"]), "examples": v["examples"]} for k, v in t["args"].items()}
    every = {c["name"] for s in finmcp.SPLITS for _, task in finmcp.iter_tasks((s,)) for c in finmcp.tool_calls(task)}
    return {"tools": used, "not_in_subset": sorted(every - set(stats)),
            "note": "tools 是筛选后单元里实际调用的工具（建图的输入）；not_in_subset 是只出现在被排除数据里的工具，可作为干扰工具。"}


def write(kept, excluded, decisions, turn_counts):
    split_of = {r["id"]: r["split"] for r in kept}
    group_of = {r["id"]: r["group"] for r in kept}
    with (HERE / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (HERE / "decisions.jsonl").open("w", encoding="utf-8") as f:
        for i, d in enumerate(decisions):
            f.write(json.dumps({"decision_id": f"d{i:04d}", "unit_id": d["task_id"], "split": split_of[d["task_id"]],
                                "group": group_of[d["task_id"]], **{k: v for k, v in d.items() if k != "task_id"}},
                               ensure_ascii=False) + "\n")
    cols = ["id", "source", "unit", "group", "group_name", "split", "difficulty", "scenery", "n_calls", "n_steps",
            "n_decisions", "pick_one", "pick_subset", "pick_all", "reference", "has_parallel", "flags", "tools", "question"]
    with (HERE / "index.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in kept:
            w.writerow({**{k: r[k] for k in cols if k in r}, **r["screen"], "n_decisions": len(r["decisions"]),
                        "flags": "；".join(r["flags"]), "tools": " > ".join(r["tools"]),
                        "question": " ".join(r["question"].split())[:120]})
    with (HERE / "excluded.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "source", "unit", "reason", "n_calls", "question"])
        w.writeheader()
        for e in excluded:
            w.writerow({**e, "question": " ".join(e["question"].split())})
    with (HERE / "review.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "question", "decisions", "真需要语义判断(Y/N)", "标准答案合理且唯一(Y/N/随意)", "备注"])
        for r in kept:
            if r["group"] != "D":
                continue
            ds = "；".join(f"{d['src']} → {d['dst']}：{d['n_candidates']} 选 {len(d['chosen'])}（{', '.join(d['chosen'][:5])}）"
                          for d in r["decisions"])
            w.writerow([r["id"], r["split"], " ".join(r["question"].split())[:200], ds, "", "", ""])
    (HERE / "tools.json").write_text(json.dumps(tool_table(kept), ensure_ascii=False, indent=1), encoding="utf-8")
    labels.sync_csv(labels.load())                 # keep judgments already made in the review tool

    n = Counter((r["source"], r["group"], r["split"]) for r in kept)
    cand = [d["n"] for d in decisions]
    stats = {
        "kept_units": len(kept),
        "by_source_group_split": {f"{s}/{g}/{sp}": c for (s, g, sp), c in sorted(n.items())},
        "by_source_group": {f"{s}/{g}": sum(c for (s2, g2, _), c in n.items() if (s2, g2) == (s, g))
                            for s, g in sorted({(s, g) for s, g, _ in n})},
        "excluded": dict(Counter(f"{e['source']}：{e['reason']}" for e in excluded)),
        "multi_turn_turns": dict(turn_counts),
        "decisions": {"total": len(decisions), "pick_one": sum(d["kind"] == "pick_one" for d in decisions),
                      "pick_subset": sum(d["kind"] == "pick_subset" for d in decisions),
                      "candidates_median": sorted(cand)[len(cand) // 2] if cand else None,
                      "candidates_p90": sorted(cand)[int(0.9 * (len(cand) - 1))] if cand else None,
                      "candidates_max": max(cand) if cand else None,
                      "over_255": sum(c > 255 for c in cand)},
        "flags": dict(Counter(f for r in kept for f in r["flags"])),
        "seed": SEED, "dev_share": DEV_SHARE,
    }
    (HERE / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


if __name__ == "__main__":
    kept, excluded, decisions, turn_counts = collect()
    assign_splits(kept)
    print(json.dumps(write(kept, excluded, decisions, turn_counts), ensure_ascii=False, indent=1))
