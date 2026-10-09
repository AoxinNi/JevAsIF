"""Split object--operation graph (SOG): schema, validation, derived data flow, disclosure and stubs.

A SOG has two parts that meet only through property ids (``Entity.prop``):
- the object graph: entities, their properties, relations and rules;
- the operation graph: tool-backed operations with inputs/outputs bound to properties.
Data-flow edges are derived (``out(a) ∩ in(a')``), never stored as authoritative data.
"""
import json
from collections import deque
from typing import Any, Literal

from pydantic import BaseModel, Field

SOG_VERSION = "0.1"
Fill = Literal["agent", "identity", "host", "const"]
Check = Literal["scope", "traced"]


class Property(BaseModel):
    id: str = Field(description="Entity.name，全图唯一")
    name: str = Field(description="数据中出现的字段名")
    description: str = Field("", description="含义、口径、单位")
    type: str = Field("string", description="string / integer / number / boolean / array / object")
    aliases: list[str] = Field(default_factory=list, description="表示同一个值的其他字段名或参数名")


class Entity(BaseModel):
    id: str = Field(description="实体 id，如 Fund")
    label: str = Field("", description="中文名称")
    description: str = Field("", description="业务含义；会写进 Jev 的判断依据")
    identifier: str | None = Field(None, description="标识属性的 id")
    label_property: str | None = Field(None, description="给标识配可读名称的属性 id")
    properties: list[Property] = Field(default_factory=list, description="属性列表")
    rules: list[str] = Field(default_factory=list, description="业务规则；会写进接口桩和 Jev 的判断依据")


class Relation(BaseModel):
    id: str = Field(description="关系 id")
    source: str = Field(description="起点实体；披露沿 source → target 方向展开")
    target: str = Field(description="终点实体")
    label: str = Field("", description="关系含义，如 可咨询、包含、引用")
    via: str | None = Field(None, description="承载这条关系的属性 id")


class InputParam(BaseModel):
    name: str = Field(description="工具参数名")
    type: str = Field("string", description="参数类型")
    required: bool = Field(False, description="是否必填")
    description: str = Field("", description="参数说明")
    property: str | None = Field(None, description="这个参数消费的属性 id；派生数据流据此推导")
    fill: Fill = Field("agent", description="谁来填：agent 模型 / identity 运行身份注入 / host 可信宿主 / const 固定值")
    check: list[Check] = Field(default_factory=list, description="执行时复核：traced 必须取自已观测的值，scope 必须在客户范围内")


class OutputBinding(BaseModel):
    path: str = Field(description="返回值中的路径，如 rows[*].fundCode；map 内部的键写作 <key>")
    property: str = Field(description="这个位置的值对应的属性 id")


class Operation(BaseModel):
    id: str = Field(description="操作 id，也是 agent 代码里调用的函数名")
    tool: str = Field(description="绑定的工具名（MCP 工具）")
    label: str = Field("", description="中文名称")
    description: str = Field("", description="功能说明，写进接口桩")
    inputs: list[InputParam] = Field(default_factory=list, description="输入参数")
    outputs: list[OutputBinding] = Field(default_factory=list, description="输出绑定：返回值的哪些位置产出哪些属性")
    returns: dict[str, Any] = Field(default_factory=dict, description="返回值形状树：{t: object|array|string|…, f: 字段, i: 元素, entity: 实体, map: 以代码为键}")
    rules: list[str] = Field(default_factory=list, description="操作级规则")


class ObjectGraph(BaseModel):
    entry: str = Field(description="入口实体；按对象范围披露从这里出发")
    entities: list[Entity] = Field(description="实体")
    relations: list[Relation] = Field(default_factory=list, description="实体之间的关系")


class OperationGraph(BaseModel):
    operations: list[Operation] = Field(description="操作；数据流边不存储，由属性的产出与消费推导")
    control_edges: list[dict[str, Any]] = Field(default_factory=list, description="可选的业务流程边")


class SOG(BaseModel):
    """Split object--operation graph."""
    sog_version: Literal["0.1"] = SOG_VERSION
    id: str = Field(description="图 id")
    label: str = Field("", description="名称")
    description: str = Field("", description="说明")
    objects: ObjectGraph = Field(description="对象图")
    operations: OperationGraph = Field(description="操作图")
    meta: dict[str, Any] = Field(default_factory=dict, description="来源、构建方式等元信息")

    # ---------- lookups ----------
    def entity(self, eid):
        return next((e for e in self.objects.entities if e.id == eid), None)

    def op(self, oid):
        return next((o for o in self.operations.operations if o.id == oid), None)

    def ops(self) -> dict:
        return {o.id: o for o in self.operations.operations}

    def property(self, pid):
        e = self.entity(pid.split(".", 1)[0])
        return next((p for p in e.properties if p.id == pid), None) if e else None

    def key_index(self) -> dict:
        """field/param name -> property id (names and aliases)."""
        idx = {}
        for e in self.objects.entities:
            for p in e.properties:
                for n in [p.name, *p.aliases]:
                    idx.setdefault(n, p.id)
        return idx

    def identifier_keys(self) -> dict:
        """field/param name -> entity id, for identifier properties only."""
        idx = {}
        for e in self.objects.entities:
            p = self.property(e.identifier) if e.identifier else None
            if p:
                for n in [p.name, *p.aliases]:
                    idx.setdefault(n, e.id)
        return idx

    def save(self, path):
        path.write_text(self.model_dump_json(indent=1, exclude_none=True), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def export_json_schema(path):
    path.write_text(json.dumps(SOG.model_json_schema(), ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- derived structure ----------
def owners(op: Operation) -> set:
    return {b.property.split(".", 1)[0] for b in op.outputs}


def derived_dataflow(sog: SOG) -> list:
    """(source op, target op, property) for every property produced by one op and consumed by another."""
    produced = {}
    for o in sog.operations.operations:
        for b in o.outputs:
            produced.setdefault(b.property, set()).add(o.id)
    edges = []
    for o in sog.operations.operations:
        for p in o.inputs:
            for src in sorted(produced.get(p.property, ())) if p.property else ():
                if src != o.id:
                    edges.append((src, o.id, p.property))
    return edges


def validate(sog: SOG) -> list:
    """Structural checks; returns human-readable problems (empty list = valid)."""
    problems = []
    ents = {e.id for e in sog.objects.entities}
    props = {p.id for e in sog.objects.entities for p in e.properties}
    if len(ents) != len(sog.objects.entities):
        problems.append("实体 id 重复")
    if sog.objects.entry not in ents:
        problems.append(f"入口实体 {sog.objects.entry} 不存在")
    for e in sog.objects.entities:
        for p in e.properties:
            if not p.id.startswith(e.id + "."):
                problems.append(f"属性 {p.id} 不属于实体 {e.id}")
        for ref in (e.identifier, e.label_property):
            if ref and ref not in props:
                problems.append(f"实体 {e.id} 引用了不存在的属性 {ref}")
    for r in sog.objects.relations:
        if r.source not in ents or r.target not in ents:
            problems.append(f"关系 {r.id} 的端点不存在")
    for o in sog.operations.operations:
        for p in o.inputs:
            if p.property and p.property not in props:
                problems.append(f"操作 {o.id} 的输入 {p.name} 引用了不存在的属性 {p.property}")
        for b in o.outputs:
            if b.property not in props:
                problems.append(f"操作 {o.id} 的输出 {b.path} 引用了不存在的属性 {b.property}")
    unreachable = ents - reachable(sog, sog.objects.entry, hops=None)
    if unreachable:
        problems.append(f"从入口不可达的实体：{sorted(unreachable)}")
    produced = {b.property for o in sog.operations.operations for b in o.outputs}
    for o in sog.operations.operations:
        for p in o.inputs:
            if "traced" in p.check and p.property not in produced:
                problems.append(f"操作 {o.id} 的输入 {p.name} 要求溯源，但没有操作产出 {p.property}")
    return problems


def reachable(sog: SOG, start: str, hops=None) -> set:
    adj = {}
    for r in sog.objects.relations:
        adj.setdefault(r.source, set()).add(r.target)
    seen, queue = {start}, deque([(start, 0)])
    while queue:
        node, d = queue.popleft()
        if hops is not None and d >= hops:
            continue
        for nxt in adj.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, d + 1))
    return seen


def disclose(sog: SOG, hops=1, entry=None) -> dict:
    """Object-scoped disclosure: an operation is visible iff every entity it produces is in scope."""
    scope = reachable(sog, entry or sog.objects.entry, hops)
    visible = [o.id for o in sog.operations.operations if owners(o) <= scope]
    frontier = sorted({r.target for r in sog.objects.relations if r.source in scope} - scope)
    return {"entities": sorted(scope), "operations": visible, "frontier": frontier}


# ---------- shapes ----------
def shape_of(value, max_items=20) -> dict:
    if isinstance(value, dict):
        return {"t": "object", "f": {k: shape_of(v, max_items) for k, v in value.items()}}
    if isinstance(value, list):
        s = None
        for v in value[:max_items]:
            s = merge_shapes(s, shape_of(v, max_items))
        return {"t": "array", "i": s or {"t": "any"}}
    if isinstance(value, bool):
        return {"t": "boolean"}
    if isinstance(value, int):
        return {"t": "integer"}
    if isinstance(value, float):
        return {"t": "number"}
    if isinstance(value, str):
        return {"t": "string"}
    return {"t": "null"}


def merge_shapes(a, b):
    if a is None:
        return b
    if b is None or a == b:
        return a
    ta, tb = a["t"], b["t"]
    if ta == "null":
        return b
    if tb == "null":
        return a
    if {ta, tb} <= {"integer", "number"}:
        return {"t": "number"}
    if ta == tb == "object":
        f = dict(a["f"])
        for k, v in b["f"].items():
            f[k] = merge_shapes(f.get(k), v)
        out = {"t": "object", "f": f}
        if a.get("entity") or b.get("entity"):
            out["entity"] = a.get("entity") or b.get("entity")
        return out
    if ta == tb == "array":
        return {"t": "array", "i": merge_shapes(a["i"], b["i"])}
    for structured in ("object", "array"):      # keep structure when an error or empty value disagrees
        if ta == structured and tb not in ("object", "array"):
            return a
        if tb == structured and ta not in ("object", "array"):
            return b
    return {"t": "any"}


def collapse_maps(shape, min_fields=20, share=0.5):
    """Objects keyed by identifiers ({"ZH001": {...}, ...}) become maps: {"t": "object", "map": True, "v": ...}."""
    import re as _re
    t = shape.get("t") if shape else None
    if t == "array":
        return {"t": "array", "i": collapse_maps(shape.get("i"), min_fields, share)}
    if t != "object":
        return shape
    if shape.get("map"):
        return {**shape, "v": collapse_maps(shape.get("v"), min_fields, share)}
    fields = shape.get("f", {})
    keyish = sum(bool(_re.match(r"^[A-Za-z0-9_.\-]{4,40}$", k)) and any(ch.isdigit() for ch in k) for k in fields)
    if len(fields) >= min_fields and keyish / max(1, len(fields)) >= share:
        v = None
        for x in fields.values():
            v = merge_shapes(v, x)
        out = {"t": "object", "map": True, "v": collapse_maps(v or {"t": "any"}, min_fields, share)}
        if shape.get("entity"):
            out["entity"] = shape["entity"]
        return out
    return {**shape, "f": {k: collapse_maps(v, min_fields, share) for k, v in fields.items()}}


def normalize_path(shape, path):
    """Rewrite a concrete leaf path so that keys inside collapsed maps read <key>."""
    if not shape or not path:
        return path
    out, cur = [], shape
    for token in path.split("."):
        name, stars = token, 0
        while name.endswith("[*]"):
            name, stars = name[:-3], stars + 1
        if name:
            if cur and cur.get("t") == "object" and cur.get("map"):
                name, cur = "<key>", cur.get("v")
            elif cur and cur.get("t") == "object":
                cur = cur.get("f", {}).get(name)
            else:
                cur = None
        for _ in range(stars):
            cur = cur.get("i") if cur and cur.get("t") == "array" else None
        out.append(name + "[*]" * stars)
    return ".".join(out)


PY_TYPES = {"string": "str", "integer": "int", "number": "float", "boolean": "bool", "null": "None", "any": "Any",
            "array": "list", "object": "dict"}


def _ann(s):
    t = s.get("t", "any")
    if t == "array":
        inner = s.get("i", {"t": "any"})
        return f"list[{_ann(inner)}]"
    return PY_TYPES.get(t, "Any")


def render_stubs(sog: SOG, op_ids=None, mode="sog", max_fields=12, max_depth=2) -> str:
    """Render operations as PyIF stubs.

    mode="sog": typed returns, entity tags, labels and rules (graph-guided).
    mode="flat": parameters only, untyped returns (what a raw MCP tool list gives).
    """
    ops = [sog.op(i) for i in (op_ids or [o.id for o in sog.operations.operations])]
    lines = []
    if mode == "sog":
        lines.append("# ---- 实体（对象图） ----")
        used = set().union(*(owners(o) for o in ops)) if ops else set()
        for e in sog.objects.entities:
            if e.id not in used:
                continue
            idp = sog.property(e.identifier).name if e.identifier and sog.property(e.identifier) else "-"
            lab = sog.property(e.label_property).name if e.label_property and sog.property(e.label_property) else "-"
            lines.append(f"# {e.id}（{e.label or e.id}）：标识字段 {idp}，名称字段 {lab}。{e.description}".rstrip())
            for r in e.rules:
                lines.append(f"#   规则：{r}")
        lines.append("")
        lines.append("# ---- 操作（操作图）：只能用关键字参数调用 ----")
    for o in ops:
        params = []
        for p in o.inputs:
            if p.fill == "identity" and mode == "sog":
                continue
            ann = {"array": "list", "object": "dict"}.get(p.type, PY_TYPES.get(p.type, "Any")) if mode == "sog" else "Any"
            params.append(f"{p.name}: {ann}" + ("" if p.required else " = None"))
        ret = f"{o.id}_R" if mode == "sog" and o.returns else "Any"
        lines.append(f"def {o.id}({', '.join(params)}) -> {ret}:")
        doc = o.description or o.label or o.id
        if mode == "sog":
            injected = [p.name for p in o.inputs if p.fill == "identity"]
            if injected:
                doc += f"（{', '.join(injected)} 由运行时注入，不要传）"
            doc += "".join(f" 规则：{r}" for r in o.rules)
            traced = [p.name for p in o.inputs if "traced" in p.check]
            if traced:
                doc += f" 参数 {', '.join(traced)} 必须取自前面工具的返回值或客户原话。"
        lines.append(f'    """{doc}"""')
        if mode == "sog" and o.returns:
            lines.extend(_render_classes(f"{o.id}_R", o.returns, max_fields, max_depth))
        lines.append("")
    return "\n".join(lines)


def _render_classes(name, shape, max_fields, max_depth):
    out, todo, seen = [], [(name, shape, 0)], {}
    while todo:
        cname, s, depth = todo.pop(0)
        if s.get("t") == "array":
            s = s.get("i", {})
        if s.get("t") != "object":
            continue
        tag = f"  # {s['entity']} 记录" if s.get("entity") else ""
        if s.get("map"):
            out.append(f"    # {cname} = dict[str, {cname}_V]：键是代码{tag}")
            todo.append((f"{cname}_V", s.get("v", {}), depth))
            continue
        out.append(f"    class {cname}:{tag}")
        fields = list(s.get("f", {}).items())
        for k, v in fields[:max_fields]:
            inner = v.get("i", {}) if v.get("t") == "array" else v
            if inner.get("t") == "object" and depth < max_depth:
                sig = json.dumps(inner, sort_keys=True, ensure_ascii=False)
                child = seen.get(sig)
                if child is None:                      # identical sibling shapes share one class
                    child = seen[sig] = f"{cname}_{_safe(k)}"
                    todo.append((child, inner, depth + 1))
                ann = f"list[{child}]" if v.get("t") == "array" else child
            elif inner.get("t") == "object":
                ann = "list[dict]" if v.get("t") == "array" else "dict"
            else:
                ann = _ann(v)
            out.append(f"        {k}: {ann}")
        if len(fields) > max_fields:
            out.append(f"        # 另有 {len(fields) - max_fields} 个字段：{', '.join(k for k, _ in fields[max_fields:max_fields + 8])} …")
    return out


def _safe(k):
    return "".join(ch if ch.isalnum() else "_" for ch in k)
