"""The unified v0.4 ontology (../finmcp-graph) as the agent-facing graph for compiled tool programs.

- ``Onto``: objects, actions, the tool mapping; the interface text an agent reads; graph-elaborated Jev
  questions and candidates; per-unit expressibility of reference trajectories.
- ``JevJudge``: one System-1 request per decision (choice for pick-one, a batch of nouls for pick-subset),
  cached on disk.
- ``Runtime``: calls actions by ontology parameter names against the replay store and pulls records out of
  responses through the mapping's field paths.
"""
import hashlib
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from . import finmcp
from .calib import _records_at
from .config import CACHE, ROOT

GRAPH = ROOT.parent / "finmcp-graph"
SUBSET = ROOT.parent / "finmcp-subset"
NONE = "NONE"
PRICE_PER_MTOK = 0.042          # Jev 1.13 input price, USD per million tokens


def obj_of(ref):
    return ref.split(".", 1)[0]


def prop_of(ref):
    return ref.split(".", 1)[1]


def path_regex(pattern):
    """Mapping path -> regex over concrete leaf paths ([*] list item, <x> map key, ** any nesting)."""
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            out += r"(?:\.[^\s]*?)?"
            i += 2
        elif pattern[i] == "<":
            j = pattern.index(">", i)
            out += r"[^.\[]+"
            i = j + 1
        elif pattern.startswith("[*]", i):
            out += r"\[\*\]"
            i += 3
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile("^" + out + "$")


def load_units():
    return [json.loads(l) for l in (SUBSET / "tasks.jsonl").open(encoding="utf-8")]


def load_decisions():
    return [json.loads(l) for l in (SUBSET / "decisions.jsonl").open(encoding="utf-8")]


def request_text(unit):
    """What the customer has said so far: earlier user turns (multi-turn units) plus this request."""
    earlier = [m.get("content") or "" for m in unit.get("history") or [] if m["role"] == "user"]
    q = unit["question"]
    return ("（此前客户说过）" + " / ".join(earlier[-3:]) + "\n（本轮）" + q) if earlier else q


def short(meaning):
    """'基金代码；六位数字…' -> '基金代码'."""
    return re.split(r"[；;，,（(：:]", meaning, 1)[0].strip() or meaning


class Onto:
    def __init__(self):
        self.ontology = json.loads((GRAPH / "finmcp-object-v04.json").read_text(encoding="utf-8"))
        self.mapping = json.loads((GRAPH / "finmcp-tool-mapping.json").read_text(encoding="utf-8"))["tools"]
        self.objects = {n["id"]: n for n in self.ontology["objectOntology"]["nodes"]}
        self.relations = self.ontology["objectOntology"]["edges"]
        self.actions = {n["id"]: n for n in self.ontology["operationOntology"]["nodes"] if n["mode"] == "action"}
        self.field_rx = {t: [(ref, path_regex(p)) for ref, ps in s["fields"].items() for p in ps]
                         for t, s in self.mapping.items()}
        self.param_by_api = {(t, p["apiField"]): p for t, s in self.mapping.items() for p in s["params"]}
        self.param_by_name = {(t, p["name"]): p for t, s in self.mapping.items() for p in s["params"]}
        self.producers, self.consumers = defaultdict(set), defaultdict(set)
        for a in self.actions.values():
            for r in a["output"]:
                if r not in a["input"]:                       # echoes are not sources
                    self.producers[r].add(a["id"])
            for r in a["input"]:
                self.consumers[r].add(a["id"])

    # ---------- what the agent reads ----------
    def meaning(self, ref):
        return self.objects[obj_of(ref)]["properties"].get(prop_of(ref), "")

    def interface(self, actions=None, with_objects=True):
        """Compact text of the unified graph: objects (properties), relations, actions (inputs/outputs as Obj.prop)."""
        acts = [self.actions[a] for a in (actions or self.actions)]
        used = {obj_of(r) for a in acts for r in a["input"] + a["output"]}
        lines = []
        if with_objects:
            lines.append("# 对象（对象.属性 是动作输入输出的唯一名字）")
            for o in self.objects.values():
                if o["id"] not in used:
                    continue
                props = "；".join(f"{k}={short(v)}" for k, v in o["properties"].items())
                lines.append(f"{o['id']}（{o['label']}）：{props}")
                for r in o["rules"]:
                    lines.append(f"  规则：{r}")
            lines.append("# 对象关系")
            lines.append("；".join(f"{e['from']} -{e['label']}-> {e['to']}" for e in self.relations
                                  if e["from"] in used and e["to"] in used))
        lines.append("# 动作（工具）")
        for a in acts:
            m = self.mapping[a["id"]]
            ins = ", ".join(f"{p['name']}: {p['property']}" + ("（会话注入）" if p["fill"] == "identity" else "")
                            for p in m["params"])
            outs = defaultdict(list)
            for r in a["output"]:
                outs[obj_of(r)].append(prop_of(r))
            out = "; ".join(f"{o}.{{{','.join(ps)}}}" for o, ps in outs.items())
            lines.append(f"{a['id']}（{a['label']}）({ins}) -> {out}")
            for r in a["rules"]:
                lines.append(f"  规则：{r}")
        return "\n".join(lines)

    # ---------- graph-elaborated decisions ----------
    def refs_at(self, tool, path):
        return [ref for ref, rx in self.field_rx.get(tool, []) if rx.match(path)]

    def target(self, dst_tool, dst_arg):
        p = self.param_by_api.get((dst_tool, dst_arg))
        return p["property"] if p else None

    def question(self, dst_tool, dst_arg, kind):
        """Question text built from the destination action and the ontology property it needs."""
        a, ref = self.actions.get(dst_tool), self.target(dst_tool, dst_arg)
        if not a or not ref:
            return None, []
        obj = self.objects[obj_of(ref)]
        need = f"下一步要「{a['label']}」，需要{obj['label']}的{short(self.meaning(ref))}（{ref}）。"
        ask = ("根据客户的请求，应该选哪一个？" if kind == "pick_one"
               else "根据客户的请求，这个候选是否应该被选中、传给下一步？")
        return need + ask, obj["rules"] + a["rules"]

    def describe(self, src_tool, src_path, record, budget=600):
        """Candidate record pruned to mapped fields and renamed by property meaning (graph elaboration)."""
        prefix = src_path.rsplit(".", 1)[0] if "." in src_path else ""
        rx = self.field_rx.get(src_tool, [])

        def walk(v, path):
            if isinstance(v, dict):
                out = {}
                for k, x in v.items():
                    sub = walk(x, f"{path}.{k}" if path else k)
                    if sub not in (None, {}, []):
                        refs = [r for r, p in rx if p.match(f"{prefix}.{path}.{k}".strip(".").replace("..", "."))
                                ] if not isinstance(x, (dict, list)) else []
                        name = short(self.meaning(refs[0])) if refs else k
                        out[name] = sub
                return out
            if isinstance(v, list):
                items = [walk(x, path + "[*]") for x in v]
                return [x for x in items if x not in (None, {}, [])][:6]
            full = f"{prefix}.{path}".strip(".")
            return v if any(p.match(full) for _, p in rx) else None

        pruned = walk(record, "")
        text = json.dumps(pruned or record, ensure_ascii=False, separators=(",", ":"))
        return text[:budget]

    # ---------- expressibility of reference trajectories ----------
    def unit_profile(self, unit):
        """How a reference trajectory maps onto the graph.

        Returns counts of argument sources (identity / literal / reference / selection), uncovered flows,
        LLM rounds in the reference (ReAct) trajectory, and characters an LLM would read in each style.
        """
        msgs = unit["history"] + unit["messages"]
        task = {"messages": msgs}
        ids = {c.get("id") for m in unit["messages"] if m["role"] == "assistant" for c in (m.get("tool_calls") or [])}
        calls = [c for c in finmcp.tool_calls(task) if c["id"] in ids]
        context = request_text(unit) if unit["unit"] == "turn" else unit["question"]
        flows = finmcp.observed_flows(task, calls, context)
        covered = uncovered = 0
        sel_groups, ref_groups = set(), set()
        for f in flows:
            dst = self.target(f["dst_tool"], f["dst_arg"])
            ok = dst is not None and dst in self.refs_at(f["src_tool"], f["src_path"])
            covered += ok
            uncovered += not ok
            g = (f["dst_tool"], f["dst_arg"], f["src_tool"], f["src_path"])
            (sel_groups if f["n_candidates"] > 1 else ref_groups).add(g)
        args = defaultdict(int)
        flowed = {(f["dst_tool"], f["dst_arg"]) for f in flows}
        for c in calls:
            for k in c["args"]:
                p = self.param_by_api.get((c["name"], k))
                kind = ("identity" if p and p["fill"] == "identity" else "upstream" if (c["name"], k) in flowed
                        else "literal")
                args[kind] += 1
        # characters read by the LLM: ReAct re-reads the growing transcript at every round
        react_chars, prefix, rounds = 0, len(context), 0
        for m in unit["messages"]:
            if m["role"] == "assistant":
                rounds += 1
                react_chars += prefix
            body = (m.get("content") or "") + json.dumps(m.get("tool_calls") or "", ensure_ascii=False)
            prefix += len(body) if m["role"] != "user" else 0
        results_chars = sum(len(c["text"] or "") for c in calls)
        return {"id": unit["id"], "group": unit["group"], "split": unit["split"], "n_calls": len(calls),
                "rounds": rounds, "flows": len(flows), "covered": covered, "uncovered": uncovered,
                "selection_points": len(sel_groups), "reference_points": len(ref_groups),
                "args_identity": args["identity"], "args_literal": args["literal"], "args_upstream": args["upstream"],
                "react_chars": react_chars, "results_chars": results_chars, "context_chars": len(context)}


class JevJudge:
    """Decision items -> Jev requests (cached). ``variant``: 'plain' or 'graph'."""

    def __init__(self, onto, model="jev-1.13.0", cache=True):
        self.onto, self.model, self.cache = onto, model, cache
        self.dir = CACHE / "jev_onto"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from typesafe_sdk import TypeSafeClient
            self._client = TypeSafeClient(model=self.model)
        return self._client

    def _ask(self, state, questions):
        """questions: {key: ("choice", instructions, criteria) | ("noul", instructions)} -> cached answers."""
        key = hashlib.sha256(json.dumps([self.model, state, questions], ensure_ascii=False, sort_keys=True)
                             .encode()).hexdigest()[:32]
        path = self.dir / f"{key}.json"
        if self.cache and path.exists():
            return {**json.loads(path.read_text(encoding="utf-8")), "cached": True}
        from typesafe_sdk import Choice, Noul
        typed = {k: (Choice(instructions=q[1], criteria=q[2]) if q[0] == "choice" else Noul(instructions=q[1]))
                 for k, q in questions.items()}
        for attempt in range(3):
            try:
                t0 = time.perf_counter()
                r = self.client.system_one(state, typed)
                latency = time.perf_counter() - t0
                break
            except Exception:                              # transient network / rate errors
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        out = {"choices": {k: dict(a.probabilities) for k, a in r.choices.items()},
               "nouls": {k: a.noul for k, a in r.nouls.items()},
               "tokens": r.usage.input_tokens, "latency": latency, "model": r.model}
        if self.cache:
            path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return {**out, "cached": False}

    VARIANTS = {                     # what the graph contributes to a decision
        "plain": dict(fields=False, question=False, rules=False),
        "fields": dict(fields=True, question=False, rules=False),
        "graph": dict(fields=True, question=True, rules=False),
        "graph+rules": dict(fields=True, question=True, rules=True),
    }

    def build(self, d, request, variant):
        """(state, questions, keys) for one decision item."""
        v = self.VARIANTS[variant]
        keys, recs = [], {}
        for c in d["candidates"]:
            k = finmcp.norm(c[d["id_key"]]) if d["id_key"] and isinstance(c, dict) else finmcp.norm(c)
            if k not in recs:
                keys.append(k)
                recs[k] = c
        if v["fields"]:
            text = {k: (self.onto.describe(d["src_tool"], d["src_path"], recs[k]) if isinstance(recs[k], dict) else k)
                    for k in keys}
        else:
            text = {k: json.dumps(recs[k], ensure_ascii=False, separators=(",", ":"))[:600] for k in keys}
        q, rules = self.onto.question(d["dst_tool"], d["dst_arg"], d["kind"]) if v["question"] else (None, [])
        if q is None:
            q = ("根据客户的请求，应该选哪一个？" if d["kind"] == "pick_one" else "根据客户的请求，这个候选是否应该被选中？")
        return self._questions(request, q, rules if v["rules"] else [], keys, text, d["kind"]) + (keys,)

    @staticmethod
    def _questions(request, q, rules, keys, text, kind):
        instr = q + ("\n规则：" + "；".join(rules) if rules else "")
        state = {"customer_request": request}
        if kind == "pick_one":
            crit = {k: text[k] for k in keys}
            crit[NONE] = "以上都不是，或者无法确定"
            return state, {"pick": ("choice", instr, crit)}
        return state, {f"c{i}": ("noul", {"question": instr, "candidate": text[k]}) for i, k in enumerate(keys)}

    def select(self, request, question, items, texts, kind="pick_subset", threshold=0.5):
        """Generic primitive for programs: items[i] is described by texts[i]; returns (picked items, scores, call)."""
        keys = [str(i) for i in range(len(items))]
        state, qs = self._questions(request, question, [], keys, dict(zip(keys, texts)), kind)
        out = self._ask(state, qs)
        if kind == "pick_one":
            probs = out["choices"]["pick"]
            best = max(keys, key=lambda k: probs.get(k, 0))
            return [items[int(best)]], {k: probs.get(k, 0) for k in keys}, out
        score = {k: out["nouls"][f"c{i}"] for i, k in enumerate(keys)}
        return [items[int(k)] for k in keys if score[k] >= threshold], score, out

    def judge(self, d, request, variant):
        state, questions, keys = self.build(d, request, variant)
        out = self._ask(state, questions)
        if d["kind"] == "pick_one":
            probs = out["choices"]["pick"]
            ranked = sorted(keys, key=lambda k: -probs.get(k, 0))
            pred, p_top = [ranked[0]], probs.get(ranked[0], 0)
            p_none = probs.get(NONE, 0)
            score = {k: probs.get(k, 0) for k in keys}
        else:
            score = {k: out["nouls"][f"c{i}"] for i, k in enumerate(keys)}
            pred = [k for k in keys if score[k] >= 0.5]
            p_top, p_none = None, None
        return {"decision_id": d["decision_id"], "variant": variant, "pred": pred, "score": score, "p_top": p_top,
                "p_none": p_none, "tokens": out["tokens"], "latency": out["latency"], "cached": out["cached"]}

    def run(self, decisions, requests, variant, workers=8):
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(lambda d: self.judge(d, requests[d["unit_id"]], variant), decisions))


class Runtime:
    """Calls actions with ontology parameter names against the replay store (for hand-written programs)."""

    def __init__(self, onto, store, task_id, identity=None):
        self.onto, self.store, self.task_id = onto, store, task_id
        self.identity, self.trace, self.last_path = identity or {}, [], {}

    def call(self, action, **kwargs):
        m = self.onto.mapping[action]
        args = {}
        for p in m["params"]:
            if p["fill"] == "identity" and p["property"] in self.identity:
                args[p["apiField"]] = self.identity[p["property"]]
            elif p["name"] in kwargs:
                args[p["apiField"]] = kwargs.pop(p["name"])
            elif p["apiField"] in kwargs:
                args[p["apiField"]] = kwargs.pop(p["apiField"])
        if kwargs:
            raise TypeError(f"{action} 没有这些参数：{sorted(kwargs)}")
        t0 = time.perf_counter()
        status, text, _, score = self.store.lookup(action, args, task_id=self.task_id)
        value = finmcp.parse_json(text) if text else None
        self.trace.append({"action": action, "args": args, "replay": status, "score": score,
                           "ms": round((time.perf_counter() - t0) * 1000, 1)})
        return value

    def records(self, value, action, ref):
        """Records in ``value`` that carry ``ref`` (an Obj.prop) according to the mapping's paths."""
        rx = [p for r, p in self.onto.field_rx.get(action, []) if r == ref]
        paths = sorted({p for p, _, _ in finmcp.leaves(value) if any(x.match(p) for x in rx)})
        out = []
        for p in paths:
            recs, key = _records_at(value, p)
            out += [r for r in recs if r not in out]
            self.last_path[(action, ref)] = p
        return out

    def describe(self, records, action, ref):
        """Graph-elaborated text for records returned by ``records(...)``."""
        path = self.last_path.get((action, ref), "")
        return [self.onto.describe(action, path, r) for r in records]

    def values(self, value, action, ref):
        rx = [p for r, p in self.onto.field_rx.get(action, []) if r == ref]
        out = []
        for p, _, v in finmcp.leaves(value):
            if any(x.match(p) for x in rx) and v not in out:
                out.append(v)
        return out


def llm_select(llm, d, request, variant_texts):
    """LLM at the decision point (the 'LLM-as-IF' baseline): same candidates, answer is a JSON list of keys."""
    keys = list(variant_texts)
    listing = "\n".join(f"[{k}] {variant_texts[k]}" for k in keys)
    ask = "只选一个" if d["kind"] == "pick_one" else "可以选一个或多个，也可以都不选"
    msg = [{"role": "user", "content": f"客户的请求：{request}\n\n下一步要调用 {d['dst_tool']}，参数 {d['dst_arg']}。"
            f"根据客户的请求，从下面的候选中选择（{ask}）。只输出 JSON：{{\"keys\": [...]}}\n\n{listing}"}]
    out = llm.chat(msg, tag="select", json_mode=True)
    try:
        got = json.loads(out["message"]["content"]).get("keys", [])
    except Exception:
        got = []
    return {"pred": [str(k) for k in got if str(k) in variant_texts], "latency": out["latency"],
            "tokens": out["usage"]["input_tokens"] + out["usage"]["output_tokens"], "cached": out["cached"]}
