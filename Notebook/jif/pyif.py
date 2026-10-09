"""PyIF: the restricted Python subset agents write, with System-1 primitives.

``check(code, ...)`` statically checks a program against the SOG (types, fields, access, provenance).
``Interpreter.run(code)`` executes it: tools through a callback, primitives through a JevRuntime.
"""
import ast
import re
import time
from dataclasses import dataclass, field

ALLOWED = (ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.For, ast.If, ast.Pass, ast.Break, ast.Continue,
           ast.Name, ast.Load, ast.Store, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Attribute,
           ast.Subscript, ast.Slice, ast.Call, ast.keyword, ast.Compare, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp,
           ast.Not, ast.USub, ast.UAdd, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
           ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
           ast.ListComp, ast.comprehension, ast.IfExp, ast.JoinedStr, ast.FormattedValue)
BUILTINS = {"len": len, "sorted": sorted, "min": min, "max": max, "sum": sum, "list": list, "set": set, "dict": dict,
            "str": str, "int": int, "float": float, "round": round, "any": any, "all": all, "abs": abs,
            "enumerate": enumerate, "zip": zip, "range": range}
METHODS = {list: {"append", "extend", "index", "count"},
           dict: {"get", "keys", "values", "items"},
           str: {"lower", "upper", "strip", "split", "startswith", "endswith", "replace", "join", "format"}}
PRIMITIVES = {"choice", "noul", "filter"}
CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    m = CODE_BLOCK.findall(text or "")
    return (m[-1] if m else text or "").strip()


# ---------------------------------------------------------------- static checking
@dataclass
class Issue:
    line: int
    message: str

    def __str__(self):
        return f"第 {self.line} 行：{self.message}"


@dataclass
class Info:
    shape: dict | None = None      # shape tree from the SOG, or None when unknown
    prov: str = "unknown"          # observed | request | literal | unknown


def _combine(*provs):
    provs = [p for p in provs if p]
    if "literal" in provs:
        return "literal"
    if "observed" in provs:
        return "observed"
    if "request" in provs:
        return "request"
    return "unknown"


def _items(shape):
    if shape and shape.get("t") == "array":
        return shape.get("i")
    return None


def _merge(a, b):
    if a is None or b is None:
        return a or b
    if a == b:
        return a
    if a.get("t") == b.get("t") == "array":
        return {"t": "array", "i": _merge(a.get("i"), b.get("i"))}
    return None


class Checker:
    def __init__(self, ops: dict, request: str = "", visible=None, max_listed=12):
        self.ops = {k: v for k, v in ops.items() if visible is None or k in visible}
        self.request, self.max_listed = request or "", max_listed
        self.issues, self.answers = [], 0

    def err(self, node, msg):
        self.issues.append(Issue(getattr(node, "lineno", 0), msg))

    def check(self, code: str) -> list:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [Issue(e.lineno or 0, f"语法错误：{e.msg}")]
        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED) and not isinstance(node, ast.alias):
                self.err(node, f"PyIF 不允许 {type(node).__name__}（只能用赋值、for、if、推导式、工具调用和 jev 原语）")
        if self.issues:
            return self.issues
        env = {}
        self.block(tree.body, env)
        if self.answers == 0:
            self.issues.append(Issue(0, "程序最后必须调用 answer(...)，列出回答需要的事实"))
        return self.issues

    # ---- statements ----
    def block(self, stmts, env):
        for s in stmts:
            self.stmt(s, env)

    def stmt(self, s, env):
        if isinstance(s, ast.Assign):
            info = self.expr(s.value, env)
            for t in s.targets:
                self.bind(t, info, env)
        elif isinstance(s, ast.AugAssign):
            info = self.expr(s.value, env)
            if isinstance(s.target, ast.Name):
                old = env.get(s.target.id)
                if old is None:
                    self.err(s, f"变量 {s.target.id} 在使用前没有赋值")
                else:
                    env[s.target.id] = Info(_merge(old.shape, info.shape), _combine(old.prov, info.prov))
        elif isinstance(s, ast.For):
            if s.orelse:
                self.err(s, "for 不能带 else")
            it = self.expr(s.iter, env)
            self.bind(s.target, Info(_items(it.shape), it.prov), env)
            self.block(s.body, env)
        elif isinstance(s, ast.If):
            self.expr(s.test, env)
            self.block(s.body, env)
            self.block(s.orelse, env)
        elif isinstance(s, ast.Expr):
            if not isinstance(s.value, ast.Call):
                self.err(s, "单独的表达式没有作用，请赋值或删除")
            self.expr(s.value, env)

    def bind(self, target, info, env):
        if isinstance(target, ast.Name):
            env[target.id] = info
        elif isinstance(target, ast.Tuple):
            for el in target.elts:
                self.bind(el, Info(None, info.prov), env)
        elif isinstance(target, ast.Subscript):
            self.expr(target.value, env)
        else:
            self.err(target, "只能给变量赋值")

    # ---- expressions ----
    def expr(self, n, env) -> Info:
        if isinstance(n, ast.Constant):
            t = {str: "string", int: "integer", float: "number", bool: "boolean"}.get(type(n.value), "any")
            prov = "request" if isinstance(n.value, str) and n.value and n.value in self.request else "literal"
            return Info({"t": t}, prov)
        if isinstance(n, ast.Name):
            if n.id in env:
                return env[n.id]
            if n.id in BUILTINS or n.id in self.ops or n.id in ("jev", "answer", "True", "False", "None"):
                return Info(None, "unknown")
            self.err(n, f"变量 {n.id} 在使用前没有赋值")
            return Info()
        if isinstance(n, (ast.List, ast.Tuple, ast.Set)):
            infos = [self.expr(e, env) for e in n.elts]
            shape = None
            for i in infos:
                shape = _merge(shape, i.shape) if shape else i.shape
            return Info({"t": "array", "i": shape or {"t": "any"}}, _combine(*(i.prov for i in infos)) if infos else "literal")
        if isinstance(n, ast.Dict):
            infos = [self.expr(v, env) for v in n.values]
            fields = {k.value: i.shape or {"t": "any"} for k, i in zip(n.keys, infos) if isinstance(k, ast.Constant)}
            return Info({"t": "object", "f": fields}, _combine(*(i.prov for i in infos)))
        if isinstance(n, ast.Attribute):
            return self.field(n, self.expr(n.value, env), n.attr)
        if isinstance(n, ast.Subscript):
            base = self.expr(n.value, env)
            if isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
                return self.field(n, base, n.slice.value)
            if isinstance(n.slice, ast.Slice):
                return base
            self.expr(n.slice, env)
            return Info(_items(base.shape), base.prov)
        if isinstance(n, ast.Call):
            return self.call(n, env)
        if isinstance(n, ast.BinOp):
            a, b = self.expr(n.left, env), self.expr(n.right, env)
            shape = _merge(a.shape, b.shape) if isinstance(n.op, ast.Add) and a.shape and a.shape.get("t") == "array" else None
            return Info(shape, _combine(a.prov, b.prov))
        if isinstance(n, (ast.Compare, ast.BoolOp, ast.UnaryOp)):
            parts = [n.left, *n.comparators] if isinstance(n, ast.Compare) else n.values if isinstance(n, ast.BoolOp) else [n.operand]
            infos = [self.expr(p, env) for p in parts]
            return Info({"t": "boolean"}, _combine(*(i.prov for i in infos)))
        if isinstance(n, ast.IfExp):
            self.expr(n.test, env)
            a, b = self.expr(n.body, env), self.expr(n.orelse, env)
            return Info(_merge(a.shape, b.shape), _combine(a.prov, b.prov))
        if isinstance(n, ast.ListComp):
            local = dict(env)
            prov = []
            for g in n.generators:
                it = self.expr(g.iter, local)
                prov.append(it.prov)
                self.bind(g.target, Info(_items(it.shape), it.prov), local)
                for cond in g.ifs:
                    self.expr(cond, local)
            elt = self.expr(n.elt, local)
            return Info({"t": "array", "i": elt.shape or {"t": "any"}}, _combine(elt.prov, *prov))
        if isinstance(n, ast.JoinedStr):
            infos = [self.expr(v.value, env) for v in n.values if isinstance(v, ast.FormattedValue)]
            return Info({"t": "string"}, _combine("literal", *(i.prov for i in infos)))
        return Info()

    def field(self, node, base: Info, key: str) -> Info:
        s = base.shape
        if s and s.get("t") == "array":
            self.err(node, f"这是一个列表，没有字段 {key}；请先遍历或取下标")
            return Info(None, base.prov)
        if s and s.get("t") == "object" and s.get("map"):
            return Info(s.get("v"), base.prov)
        if s and s.get("t") == "object" and s.get("f"):
            if key not in s["f"]:
                avail = list(s["f"])
                more = f" 等 {len(avail)} 个" if len(avail) > self.max_listed else ""
                self.err(node, f"返回值里没有字段 {key}；可用字段：{', '.join(avail[:self.max_listed])}{more}")
                return Info(None, base.prov)
            return Info(s["f"][key], base.prov)
        return Info(None, base.prov)

    def call(self, n, env) -> Info:
        f = n.func
        if isinstance(f, ast.Name):
            if f.id in self.ops:
                return self.op_call(n, self.ops[f.id], env)
            if f.id == "answer":
                self.answers += 1
                for a in [*n.args, *(k.value for k in n.keywords)]:
                    self.expr(a, env)
                return Info()
            if f.id in BUILTINS:
                infos = [self.expr(a, env) for a in n.args]
                for k in n.keywords:
                    self.expr(k.value, env)
                prov = _combine(*(i.prov for i in infos)) if infos else "literal"
                if f.id in ("sorted", "list") and infos:
                    return Info(infos[0].shape, prov)
                return Info(None, prov)
            self.err(n, f"没有可调用的 {f.id}（只能调用披露的工具、jev 原语、answer 和少量内置函数）")
            return Info()
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "jev":
            return self.primitive(n, f.attr, env)
        if isinstance(f, ast.Attribute):
            base = self.expr(f.value, env)
            allowed = set().union(*METHODS.values())
            if f.attr not in allowed:
                self.err(n, f"不允许调用方法 .{f.attr}()")
            for a in [*n.args, *(k.value for k in n.keywords)]:
                self.expr(a, env)
            if f.attr == "get" and base.shape and base.shape.get("t") == "object" and n.args \
                    and isinstance(n.args[0], ast.Constant):
                return self.field(n, base, n.args[0].value)
            return Info(None, base.prov)
        self.err(n, "不支持这种调用")
        return Info()

    def op_call(self, n, op, env) -> Info:
        if n.args:
            self.err(n, f"请用关键字参数调用 {op.id}，例如 {op.id}({', '.join(p.name + '=...' for p in op.inputs[:2])})")
        params = {p.name: p for p in op.inputs}
        given = set()
        for k in n.keywords:
            if k.arg is None:
                self.err(n, "不能使用 **kwargs")
                continue
            given.add(k.arg)
            info = self.expr(k.value, env)
            p = params.get(k.arg)
            if p is None:
                self.err(n, f"{op.id} 没有参数 {k.arg}；可用参数：{', '.join(params) or '无'}")
            elif p.fill == "identity":
                self.err(n, f"参数 {k.arg} 由运行时注入，不要传")
            elif "traced" in p.check and info.prov == "literal":
                self.err(n, f"参数 {k.arg} 必须取自前面工具的返回值或客户原话，不能写死成常量")
        for p in op.inputs:
            if p.required and p.fill == "agent" and p.name not in given:
                self.err(n, f"{op.id} 缺少必填参数 {p.name}")
        return Info(op.returns or None, "observed")

    def primitive(self, n, name, env) -> Info:
        if name not in PRIMITIVES:
            self.err(n, f"jev 只有 choice / noul / filter 三个原语，没有 {name}")
            return Info()
        args = list(n.args)
        kws = {k.arg: k.value for k in n.keywords}
        q = args[0] if args else kws.get("question") or kws.get("q")
        if q is None or not (isinstance(q, ast.JoinedStr) or (isinstance(q, ast.Constant) and isinstance(q.value, str))):
            self.err(n, f"jev.{name} 的第一个参数必须是问题字符串")
        elif isinstance(q, ast.JoinedStr):
            self.expr(q, env)
        if name == "noul":
            if len(args) > 1:
                self.expr(args[1], env)
            return Info({"t": "boolean"}, "observed")
        xs = args[1] if len(args) > 1 else kws.get("items") or kws.get("xs")
        if xs is None:
            self.err(n, f"jev.{name} 需要候选列表作为第二个参数")
            return Info()
        info = self.expr(xs, env)
        if info.prov in ("literal", "request"):
            self.err(n, f"jev.{name} 的候选必须来自工具返回，不能是写死的列表")
        if info.shape and info.shape.get("t") not in ("array", "any"):
            self.err(n, f"jev.{name} 的候选应该是列表")
        k = kws.get("k")
        if k is not None and not (isinstance(k, ast.Constant) and k.value in (1, "*")):
            self.err(n, "k 只能是 1 或 \"*\"")
        multi = name == "filter" or (k is not None and isinstance(k, ast.Constant) and k.value == "*")
        return Info(info.shape if multi else _items(info.shape), "observed")


def check(code, ops, request="", visible=None) -> list:
    return Checker(ops, request, visible).check(code)


# ---------------------------------------------------------------- interpreter
class PyIFError(Exception):
    def __init__(self, line, message):
        super().__init__(f"第 {line} 行：{message}")
        self.line, self.message = line, message


@dataclass
class RunResult:
    facts: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    error: str | None = None
    tool_calls: int = 0


class Interpreter:
    def __init__(self, ops: dict, call_tool, jev=None, request="", identity=None, max_steps=20000, max_tool_calls=40):
        self.ops, self.call_tool, self.jev = ops, call_tool, jev
        self.request, self.identity = request, identity or {}
        self.max_steps, self.max_tool_calls = max_steps, max_tool_calls

    def run(self, code: str) -> RunResult:
        self.res, self.steps = RunResult(), 0
        if self.jev:
            self.jev.start(self.request)
        try:
            tree = ast.parse(code)
            env = {}
            self.block(tree.body, env)
        except PyIFError as e:
            self.res.error = str(e)
        except _Stop:
            pass
        if self.jev:
            for d in self.jev.decisions:
                self.res.trace.append({"type": "jev", "kind": d.kind, "question": d.question, "result": d.result,
                                       "gated": d.gated, "n_options": len(d.options), "latency": d.latency})
        return self.res

    def tick(self, node):
        self.steps += 1
        if self.steps > self.max_steps:
            raise PyIFError(getattr(node, "lineno", 0), "执行步数超限")

    # ---- statements ----
    def block(self, stmts, env):
        for s in stmts:
            self.stmt(s, env)

    def stmt(self, s, env):
        self.tick(s)
        if isinstance(s, ast.Assign):
            v = self.eval(s.value, env)
            for t in s.targets:
                self.assign(t, v, env)
        elif isinstance(s, ast.AugAssign):
            cur = self.eval(s.target, env)
            self.assign(s.target, self.binop(s.op, cur, self.eval(s.value, env), s), env)
        elif isinstance(s, ast.For):
            for item in list(self.eval(s.iter, env) or []):
                self.assign(s.target, item, env)
                try:
                    self.block(s.body, env)
                except _Continue:
                    continue
                except _Break:
                    break
        elif isinstance(s, ast.If):
            self.block(s.body if self.eval(s.test, env) else s.orelse, env)
        elif isinstance(s, ast.Expr):
            self.eval(s.value, env)
        elif isinstance(s, ast.Break):
            raise _Break()
        elif isinstance(s, ast.Continue):
            raise _Continue()

    def assign(self, t, v, env):
        if isinstance(t, ast.Name):
            env[t.id] = v
        elif isinstance(t, ast.Tuple):
            for el, x in zip(t.elts, v):
                self.assign(el, x, env)
        elif isinstance(t, ast.Subscript):
            self.eval(t.value, env)[self.eval(t.slice, env)] = v

    # ---- expressions ----
    def eval(self, n, env):
        self.tick(n)
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in env:
                return env[n.id]
            if n.id in BUILTINS:
                return BUILTINS[n.id]
            raise PyIFError(n.lineno, f"变量 {n.id} 未定义")
        if isinstance(n, ast.List):
            return [self.eval(e, env) for e in n.elts]
        if isinstance(n, ast.Tuple):
            return tuple(self.eval(e, env) for e in n.elts)
        if isinstance(n, ast.Set):
            return {self.eval(e, env) for e in n.elts}
        if isinstance(n, ast.Dict):
            return {self.eval(k, env): self.eval(v, env) for k, v in zip(n.keys, n.values)}
        if isinstance(n, ast.Attribute):
            return self.get(self.eval(n.value, env), n.attr, n)
        if isinstance(n, ast.Subscript):
            base = self.eval(n.value, env)
            if isinstance(n.slice, ast.Slice):
                lo = self.eval(n.slice.lower, env) if n.slice.lower else None
                hi = self.eval(n.slice.upper, env) if n.slice.upper else None
                return base[lo:hi]
            key = self.eval(n.slice, env)
            try:
                return base[key]
            except (KeyError, IndexError, TypeError) as e:
                raise PyIFError(n.lineno, f"取值失败：{type(e).__name__} {key!r}")
        if isinstance(n, ast.Call):
            return self.call(n, env)
        if isinstance(n, ast.BinOp):
            return self.binop(n.op, self.eval(n.left, env), self.eval(n.right, env), n)
        if isinstance(n, ast.BoolOp):
            if isinstance(n.op, ast.And):
                v = True
                for x in n.values:
                    v = self.eval(x, env)
                    if not v:
                        return v
                return v
            for x in n.values:
                v = self.eval(x, env)
                if v:
                    return v
            return v
        if isinstance(n, ast.UnaryOp):
            v = self.eval(n.operand, env)
            return {ast.Not: lambda: not v, ast.USub: lambda: -v, ast.UAdd: lambda: +v}[type(n.op)]()
        if isinstance(n, ast.Compare):
            left = self.eval(n.left, env)
            for op, right_n in zip(n.ops, n.comparators):
                right = self.eval(right_n, env)
                if not _COMPARE[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(n, ast.IfExp):
            return self.eval(n.body if self.eval(n.test, env) else n.orelse, env)
        if isinstance(n, ast.ListComp):
            return self.listcomp(n, env)
        if isinstance(n, ast.JoinedStr):
            return "".join(str(self.eval(v.value, env)) if isinstance(v, ast.FormattedValue) else str(v.value)
                           for v in n.values)
        raise PyIFError(getattr(n, "lineno", 0), f"不支持 {type(n).__name__}")

    def listcomp(self, n, env):
        out = []

        def gen(i, local):
            if i == len(n.generators):
                out.append(self.eval(n.elt, local))
                return
            g = n.generators[i]
            for item in list(self.eval(g.iter, local) or []):
                inner = dict(local)
                self.assign(g.target, item, inner)
                if all(self.eval(c, inner) for c in g.ifs):
                    gen(i + 1, inner)
        gen(0, dict(env))
        return out

    def binop(self, op, a, b, node):
        try:
            return _BINOPS[type(op)](a, b)
        except Exception as e:
            raise PyIFError(getattr(node, "lineno", 0), f"运算失败：{e}")

    def get(self, obj, key, node):
        if isinstance(obj, dict):
            if key not in obj:
                raise PyIFError(node.lineno, f"返回值里没有字段 {key}")
            return obj[key]
        raise PyIFError(node.lineno, f"{type(obj).__name__} 没有字段 {key}")

    def call(self, n, env):
        f = n.func
        if isinstance(f, ast.Name) and f.id in self.ops:
            return self.tool(n, self.ops[f.id], env)
        if isinstance(f, ast.Name) and f.id == "answer":
            for k in n.keywords:
                self.res.facts[k.arg] = self.eval(k.value, env)
            for i, a in enumerate(n.args):
                self.res.facts[f"fact{i}"] = self.eval(a, env)
            raise _Stop()
        if isinstance(f, ast.Name) and f.id in BUILTINS:
            args = [self.eval(a, env) for a in n.args]
            kwargs = {k.arg: self.eval(k.value, env) for k in n.keywords}
            try:
                return BUILTINS[f.id](*args, **kwargs)
            except Exception as e:
                raise PyIFError(n.lineno, f"{f.id}() 失败：{e}")
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "jev":
            return self.primitive(n, f.attr, env)
        if isinstance(f, ast.Attribute):
            obj = self.eval(f.value, env)
            allowed = next((m for t, m in METHODS.items() if isinstance(obj, t)), set())
            if f.attr not in allowed:
                raise PyIFError(n.lineno, f"不允许调用 {type(obj).__name__}.{f.attr}()")
            args = [self.eval(a, env) for a in n.args]
            return getattr(obj, f.attr)(*args, **{k.arg: self.eval(k.value, env) for k in n.keywords})
        raise PyIFError(n.lineno, "不支持这种调用")

    def tool(self, n, op, env):
        if self.res.tool_calls >= self.max_tool_calls:
            raise PyIFError(n.lineno, f"工具调用次数超过 {self.max_tool_calls}")
        args = {k.arg: self.eval(k.value, env) for k in n.keywords}
        coerced = []
        for p in op.inputs:
            if p.fill == "identity" and p.name in self.identity:
                args[p.name] = self.identity[p.name]
            if p.name in args and args[p.name] is not None:
                v = args[p.name]
                if p.type == "array" and not isinstance(v, list):
                    args[p.name] = [v]
                    coerced.append(p.name)
                elif p.type != "array" and isinstance(v, list) and len(v) == 1:
                    args[p.name] = v[0]
                    coerced.append(p.name)
        args = {k: v for k, v in args.items() if v is not None}
        t0 = time.perf_counter()
        out = self.call_tool(op.tool, args)
        self.res.tool_calls += 1
        self.res.trace.append({"type": "tool", "line": n.lineno, "name": op.tool, "args": args, "ok": out["ok"],
                               "chars": len(out.get("text") or ""), "latency": time.perf_counter() - t0,
                               **({"coerced": coerced} if coerced else {})})
        if not out["ok"]:
            raise PyIFError(n.lineno, f"工具 {op.tool} 返回错误：{(out.get('text') or '')[:200]}")
        value = out["json"] if out.get("json") is not None else out.get("text")
        if self.jev:
            self.jev.observe(value)
        return value

    def primitive(self, n, name, env):
        if self.jev is None:
            raise PyIFError(n.lineno, "没有配置 Jev 运行时")
        args = [self.eval(a, env) for a in n.args]
        kws = {k.arg: self.eval(k.value, env) for k in n.keywords}
        q = args[0] if args else kws.pop("question", kws.pop("q", ""))
        if name == "noul":
            item = args[1] if len(args) > 1 else kws.get("item")
            return self.jev.noul(q, item)
        xs = args[1] if len(args) > 1 else kws.get("items", kws.get("xs"))
        if name == "filter":
            return self.jev.filter(q, xs)
        return self.jev.choice(q, xs, k=kws.get("k", 1))


class _Stop(Exception):
    pass


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


_COMPARE = {ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b, ast.Lt: lambda a, b: a < b,
            ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
            ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b, ast.Is: lambda a, b: a is b,
            ast.IsNot: lambda a, b: a is not b}
_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
           ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b}
