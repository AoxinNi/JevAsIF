"""Build a SOG automatically from tool-call logs (steps 1–5 of the construction procedure).

1. operations from observed calls and responses
2. property alignment: union-find over identifier fields (tool, path) and parameters (tool, arg),
   using observed value flows, specific shared names, and value containment
3. entity induction: one entity per identifier cluster; label property by name pattern
4. relation induction: nested records, co-occurring identifiers, user scope and lookup operations
5. derivation and validation: derived data flow, reachability, coverage of observed flows
Step 6 (curation by an expert) is left to the notebook.
"""
import re
from collections import Counter, defaultdict

from . import finmcp
from .sog import (SOG, Entity, InputParam, ObjectGraph, Operation, OperationGraph, OutputBinding, Property,
                  Relation, collapse_maps, derived_dataflow, merge_shapes, normalize_path, shape_of, validate)

ID_KEY = re.compile(r"^(code|codes|id|ids)$|[a-z](Code|Codes|Id|Ids|ID)$|代码$|编号$|ID$")
ID_ARG = re.compile(r"^(code|codes|id|ids)$|[a-z](Code|Codes|Id|Ids|IdList|CodeList)$|^fundList$")
GENERIC = {"code", "id", "ids", "identifier", "value", "key", "type"}
LABEL_KEY = re.compile(r"(name|名称|简称|标题|title)$", re.I)
ENTITY_NAMES = [  # (pattern over member names, entity id, label); best match wins, ties by order
    (r"manager|经理", "Manager", "基金经理"),
    (r"announcement|公告", "Announcement", "公告"),
    (r"assetplan", "AssetPlan", "资产配置方案"),
    (r"serviceaccount|account|assetid", "Account", "账户"),
    (r"category|分类", "Category", "资产类别"),
    (r"index", "Index", "指数"),
    (r"fund|基金", "Fund", "基金"),
    (r"strategy|策略|^po|组合", "Strategy", "投顾策略/组合"),
    (r"assetcode|prod", "Product", "持有产品"),
    (r"user", "User", "用户"),
]
JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "list": "array",
              "dict": "object", "NoneType": "null"}


class UnionFind:
    """Union-find with cannot-link labels: two clusters with different labels are never merged."""

    def __init__(self):
        self.parent, self.label, self.blocked = {}, {}, 0

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def add(self, x, label=None):
        r = self.find(x)
        if label and not self.label.get(r):
            self.label[r] = label

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        la, lb = self.label.get(ra), self.label.get(rb)
        if la and lb and la != lb:
            self.blocked += 1
            return
        self.parent[ra] = rb
        self.label[rb] = lb or la

    def groups(self):
        g = defaultdict(set)
        for x in list(self.parent):
            g[self.find(x)].add(x)
        return list(g.values())


def collect(tasks):
    """Step 1 inputs: per-tool argument stats, merged return shapes, leaf values and observed flows."""
    st = {"calls": Counter(), "args": defaultdict(lambda: defaultdict(lambda: {"n": 0, "types": Counter(), "values": set()})),
          "shapes": {}, "leaf": defaultdict(lambda: {"key": None, "values": set(), "n": 0}), "flows": [], "tasks": 0}
    for _, task in tasks:
        st["tasks"] += 1
        for call in finmcp.tool_calls(task):
            tool = call["name"]
            st["calls"][tool] += 1
            for arg, v in call["args"].items():
                a = st["args"][tool][arg]
                a["n"] += 1
                a["types"][type(v).__name__] += 1
                for x in (v if isinstance(v, list) else [v]):
                    if not isinstance(x, (dict, list)) and len(a["values"]) < 300:
                        a["values"].add(finmcp.norm(x))
            if call["json"] is not None:
                st["shapes"][tool] = merge_shapes(st["shapes"].get(tool), shape_of(call["json"]))
                for path, key, value in finmcp.leaves(call["json"]):
                    lf = st["leaf"][(tool, path)]
                    lf["key"], lf["n"] = key, lf["n"] + 1
                    if len(lf["values"]) < 300:
                        lf["values"].add(finmcp.norm(value))
            else:
                st["shapes"].setdefault(tool, {"t": "any"})
        st["flows"].extend(finmcp.observed_flows(task))
    # identifier-keyed objects become maps; rewrite every observed path to match the collapsed shapes
    st["shapes"] = {t: collapse_maps(s) for t, s in st["shapes"].items()}
    leaf = defaultdict(lambda: {"key": None, "values": set(), "n": 0})
    for (tool, path), lf in st["leaf"].items():
        tgt = leaf[(tool, normalize_path(st["shapes"].get(tool), path))]
        tgt["key"], tgt["n"] = lf["key"], tgt["n"] + lf["n"]
        tgt["values"] |= lf["values"]
    st["leaf"] = leaf
    for f in st["flows"]:
        f["src_path"] = normalize_path(st["shapes"].get(f["src_tool"]), f["src_path"])
    return st


def _code_like(values):
    vals = [v for v in values if v]
    return bool(vals) and sum(finmcp.identifier_like(v) for v in vals) / len(vals) >= 0.7


def align(st, min_overlap=3, containment=0.5):
    """Step 2: cluster identifier fields ("p", tool, path) and parameters ("a", tool, arg)."""
    uf = UnionFind()
    id_args = {(t, a): v["values"] for t, args in st["args"].items() for a, v in args.items()
               if a not in finmcp.PAGING_ARGS and ID_ARG.search(a)}
    flow_src = {(f["src_tool"], f["src_path"]) for f in st["flows"] if (f["dst_tool"], f["dst_arg"]) in id_args}
    id_paths = {(t, p): lf["values"] for (t, p), lf in st["leaf"].items()
                if _code_like(lf["values"]) and ((t, p) in flow_src or ID_KEY.search(lf["key"]))}
    key_of = {(t, p): st["leaf"][(t, p)]["key"] for (t, p) in id_paths}
    for n in id_paths:
        uf.add(("p", *n), seed_label(key_of[n]))
    for n in id_args:
        uf.add(("a", *n), seed_label(n[1]))
    for f in st["flows"]:                                   # a. observed value flows
        src, dst = (f["src_tool"], f["src_path"]), (f["dst_tool"], f["dst_arg"])
        if src in id_paths and dst in id_args:
            uf.union(("p", *src), ("a", *dst))
    by_name = defaultdict(list)                             # b. specific names shared across tools
    for n, k in key_of.items():
        if k.lower() not in GENERIC:
            by_name[k].append(("p", *n))
    for (t, a) in id_args:
        if a.lower() not in GENERIC:
            by_name[a].append(("a", t, a))
    for nodes in by_name.values():
        for x in nodes[1:]:
            uf.union(nodes[0], x)
    paths = sorted(id_paths)                                # c. value containment between fields
    for i, n1 in enumerate(paths):
        for n2 in paths[i + 1:]:
            v1, v2 = id_paths[n1], id_paths[n2]
            inter = len(v1 & v2)
            if inter >= min_overlap and inter / max(1, min(len(v1), len(v2))) >= containment:
                uf.union(("p", *n1), ("p", *n2))
    for na, va in id_args.items():                          # d. parameters whose values match a field
        for n, vp in id_paths.items():
            inter = len(va & vp)
            if inter >= min_overlap and inter / max(1, len(va)) >= containment:
                uf.union(("a", *na), ("p", *n))
    return uf.groups(), key_of, uf.blocked


SEED_LABELS = [(r"fund|基金", "Fund"), (r"strategy|策略|^po[A-Z]|^po$|组合", "Strategy"), (r"user", "User"),
               (r"account", "Account"), (r"announcement|公告", "Announcement"), (r"manager|经理", "Manager")]


def seed_label(name):
    """Type label from a specific field name; generic or mixed names (code, assetCode) get none."""
    if name.lower() in GENERIC:
        return None
    if re.search(r"manager|经理", name, re.I):
        return "Manager"
    return next((lab for pat, lab in SEED_LABELS if re.search(pat, name, re.I)), None)


def _name_entity(names, taken):
    low = [n.lower() for n in names]
    best, best_score = None, 0
    for pattern, eid, label in ENTITY_NAMES:
        score = sum(bool(re.search(pattern, n)) for n in low)
        if score > best_score:
            best, best_score = (eid, label), score
    eid, label = best or ("E_" + sorted(names)[0], "")
    base, i = eid, 2
    while eid in taken:
        eid, i = f"{base}{i}", i + 1
    return eid, label


def build_sog(tasks, sog_id="finmcp-auto", label="FinMCP-Bench（自动建图）"):
    """Run steps 1–5; returns (sog, report)."""
    st = collect(list(tasks))
    groups, key_of, blocked = align(st)
    name_count = Counter()
    for (tool, path), lf in st["leaf"].items():
        name_count[lf["key"]] += lf["n"]

    # ---- Step 3: entities from identifier clusters ----
    ent_of_path, ent_of_arg, canon, entities, taken = {}, {}, {}, {}, {"User"}
    for members in sorted(groups, key=lambda g: sorted(map(str, g))):
        paths = [(m[1], m[2]) for m in members if m[0] == "p"]
        args = [(m[1], m[2]) for m in members if m[0] == "a"]
        if any(a == "userId" for _, a in args):
            for n in args:
                ent_of_arg[n] = "User"
            continue
        names = {key_of[n] for n in paths} | {a for _, a in args}
        if not paths or (not args and all(n.lower() in GENERIC for n in names) and len(paths) < 2):
            continue
        eid, elabel = _name_entity(names, taken)
        taken.add(eid)
        specific = [key_of[n] for n in paths if key_of[n].lower() not in GENERIC] or [key_of[n] for n in paths]
        canon[eid] = Counter(specific).most_common(1)[0][0]
        entities[eid] = {"label": elabel, "props": {}, "aliases": sorted(names - {canon[eid]}), "nodes": 0}
        for n in paths:
            ent_of_path[n] = eid
        for n in args:
            ent_of_arg[n] = eid
    entities["User"] = {"label": "用户（运行身份）", "props": {}, "aliases": [], "nodes": 0}
    canon["User"] = "userId"

    def ident(eid):
        return f"{eid}.{canon[eid]}"

    # ---- tag shapes; collect properties, output bindings and nesting relations ----
    relations, bindings, shapes = set(), defaultdict(list), {}

    def tag(shape, path, parent_entity, tool):
        t = shape.get("t")
        if t == "array":
            return {"t": "array", "i": tag(shape["i"], path + "[*]", parent_entity, tool)}
        if t == "object" and shape.get("map"):
            return {"t": "object", "map": True, "v": tag(shape["v"], f"{path}.<key>" if path else "<key>", parent_entity, tool)}
        if t != "object":
            return dict(shape)
        fields = shape["f"]

        def child(k):
            return f"{path}.{k}" if path else k
        scalar_ids = [ent_of_path[(tool, child(k))] for k, v in fields.items()
                      if v["t"] not in ("object", "array") and (tool, child(k)) in ent_of_path]
        primary = scalar_ids[0] if scalar_ids else None
        out = {"t": "object", "f": {}}
        if primary:
            out["entity"] = primary
            entities[primary]["nodes"] += 1
            if parent_entity and parent_entity != primary:
                relations.add((parent_entity, primary, "包含", None))
        for k, v in fields.items():
            p = child(k)
            if v["t"] in ("object", "array"):
                out["f"][k] = tag(v, p, primary or parent_entity, tool)
                if v["t"] == "array" and (tool, p + "[*]") in ent_of_path:
                    bindings[tool].append(OutputBinding(path=p + "[*]", property=ident(ent_of_path[(tool, p + "[*]")])))
                continue
            out["f"][k] = dict(v)
            target = ent_of_path.get((tool, p))
            if target:
                bindings[tool].append(OutputBinding(path=p, property=ident(target)))
                if primary and target != primary:
                    relations.add((primary, target, "引用", ident(target)))
            elif primary:
                entities[primary]["props"].setdefault(k, v["t"])
                bindings[tool].append(OutputBinding(path=p, property=f"{primary}.{k}"))
        return out

    for tool, shape in st["shapes"].items():
        shapes[tool] = tag(shape, "", None, tool)

    # ---- Step 1: operations ----
    operations = []
    for tool in sorted(st["calls"]):
        n_calls = st["calls"][tool]
        inputs = []
        for arg, a in st["args"][tool].items():
            jt = JSON_TYPES.get(a["types"].most_common(1)[0][0], "string")
            eid = ent_of_arg.get((tool, arg))
            prop = ident(eid) if eid else None
            fill = "identity" if arg == "userId" else "agent"
            desc = f"调用中出现 {a['n']}/{n_calls} 次"
            if not prop and jt == "string" and 0 < len(a["values"]) <= 8:
                desc += "；出现过的取值：" + "、".join(sorted(a["values"])[:8])
            inputs.append(InputParam(name=arg, type=jt, required=a["n"] / n_calls >= 0.95, description=desc,
                                     property=prop, fill=fill, check=["traced"] if prop and fill == "agent" else []))
        produced = {b.property.split(".", 1)[0] for b in bindings[tool]}
        if any(p.name == "userId" for p in inputs):
            relations.update(("User", e, "拥有", None) for e in produced)
        elif not any(p.check for p in inputs):
            relations.update(("User", e, "可检索", None) for e in produced)
        top = shapes[tool].get("f", {}) if shapes[tool].get("t") == "object" else {}
        desc = f"{tool}：返回字段 {', '.join(list(top)[:8])}" if top else tool
        uniq = {(b.path, b.property): b for b in bindings[tool]}
        operations.append(Operation(id=tool, tool=tool, label=tool, description=desc, inputs=inputs,
                                    outputs=list(uniq.values()), returns=shapes[tool]))

    # ---- Steps 3–4: assemble entities and relations ----
    ent_models = []
    for eid, e in entities.items():
        props = [Property(id=ident(eid), name=canon[eid], type="string", aliases=e["aliases"])]
        props += [Property(id=f"{eid}.{k}", name=k, type=t) for k, t in e["props"].items() if k != canon[eid]]
        label_candidates = [p for p in props if LABEL_KEY.search(p.name)]
        label_prop = max(label_candidates, key=lambda p: name_count[p.name]).id if label_candidates else None
        desc = "运行时绑定的客户身份" if eid == "User" else f"自动归纳：在 {e['nodes']} 处返回结构中出现"
        ent_models.append(Entity(id=eid, label=e["label"], identifier=ident(eid), label_property=label_prop,
                                 description=desc, properties=props))
    rels = []
    for s, t, lab, via in sorted(relations, key=lambda r: (r[0], r[1], r[2])):
        if s != t and not any(r.source == s and r.target == t for r in rels):
            rels.append(Relation(id=f"r{len(rels)}", source=s, target=t, label=lab, via=via))
    sog = SOG(id=sog_id, label=label, objects=ObjectGraph(entry="User", entities=ent_models, relations=rels),
              operations=OperationGraph(operations=operations),
              meta={"built_from_tasks": st["tasks"], "procedure": "auto steps 1-5"})

    # ---- Step 5: derivation, validation, coverage ----
    clusters = defaultdict(set)
    for (tool, path), eid in ent_of_path.items():
        clusters[eid].add(key_of[(tool, path)])
    for (tool, arg), eid in ent_of_arg.items():
        clusters[eid].add(arg)
    report = {"flows": st["flows"], "tasks": st["tasks"], "operations": len(operations), "entities": len(ent_models), "relations": len(rels),
              "properties": sum(len(e.properties) for e in ent_models), "derived_edges": len(derived_dataflow(sog)),
              "problems": validate(sog), "coverage": coverage(sog, st["flows"]), "blocked_merges": blocked,
              "clusters": {k: sorted(v) for k, v in clusters.items()}}
    return sog, report


def coverage(sog, flows):
    """Share of observed value flows that the derived data-flow edges explain."""
    out_prop = {(o.id, b.path): b.property for o in sog.operations.operations for b in o.outputs}
    in_prop = {(o.id, p.name): p.property for o in sog.operations.operations for p in o.inputs}
    res = {"all": [0, 0], "selection": [0, 0], "reference": [0, 0]}
    missed = Counter()
    for f in flows:
        src = out_prop.get((f["src_tool"], f["src_path"]))
        dst = in_prop.get((f["dst_tool"], f["dst_arg"]))
        ok = bool(src) and src == dst
        for k in ("all", "selection" if f["n_candidates"] > 1 else "reference"):
            res[k][0] += ok
            res[k][1] += 1
        if not ok:
            missed[(f["src_tool"], f["src_key"], f["dst_tool"], f["dst_arg"])] += 1
    summary = {k: {"covered": c, "total": n, "recall": round(c / n, 3) if n else None} for k, (c, n) in res.items()}
    summary["top_missed"] = [{"flow": " → ".join(k), "count": v} for k, v in missed.most_common(10)]
    return summary
