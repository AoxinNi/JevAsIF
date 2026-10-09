"""Agents: a ReAct baseline over MCP tools, the Jev-as-IF compiler agent, and an LLM judge."""
import json
import time

from . import pyif
from .llm import Ledger
from .sog import disclose, render_stubs

# ---------------------------------------------------------------- prompts
REACT_SYSTEM = ("你是证券公司的投顾助手，可以调用工具查询数据。请按需调用工具，拿到事实后用中文回答客户。"
                "不要编造工具没有返回的数据。")

PYIF_SYSTEM = """你负责把客户的问题写成一段 PyIF 程序，一次性完成所有工具调用。

PyIF 是 Python 的受限子集：
- 只能用：赋值、for 循环、if/else、列表推导式、算术和比较、下面列出的工具函数、jev 原语、answer()，以及 len/sorted/min/max/sum/list/set/round/any/all/abs/enumerate/zip/range 和列表、字典、字符串的常用方法。
- 不能用：import、def、class、while、lambda、try、with、print、eval/exec。
- 工具函数只能用关键字参数调用，例如 SearchSomething(keyword="债券")。
- 工具返回嵌套的 dict/list，用点号或方括号取字段：r.rows[0].代码 或 r["rows"][0]["代码"]。中文字段名可以直接写。
- 字段名必须和接口说明里的一致，不要猜。
- userId 等身份参数由运行时注入，不要传。
- 标注"必须取自前面工具的返回值或客户原话"的参数，不能写死成别的常量。

需要语义判断时（客户说的是哪一个、哪些条目和问题相关、某条记录是否满足一个定性条件），不要写字符串匹配，改用 jev 原语：
- jev.choice(问题, 候选列表)          → 从候选里选出一个（候选是工具返回的记录或代码）
- jev.choice(问题, 候选列表, k="*")   → 选出所有符合的候选，返回列表
- jev.filter(问题, 候选列表)          → 保留问题成立的候选，返回列表
- jev.noul(问题, 某条记录)            → 返回 True/False，可写成 if jev.noul(...):
问题要写成一句完整、明确的中文问题。数字比较、日期计算直接写代码，不要交给 jev。

程序最后必须调用一次 answer(名称=值, ...)，列出回答客户需要的事实（可以直接放工具返回的记录）。answer 之后的代码不会执行。

示例（工具名和字段名只是演示，实际以下面的接口为准）：
```python
found = SearchProducts(keyword="稳健")
target = jev.choice("客户说的'那个稳健的组合'是下面哪一个？", found.rows)
fees = GetFeeRules(productCodes=[target.code])
answer(product=target, fees=fees)
```

只输出一个 ```python 代码块，不要解释。

可用接口：
{stubs}"""

ANSWER_SYSTEM = ("你是证券公司的投顾助手。请只依据下面通过工具查到的事实，用中文回答客户。"
                 "不要编造事实里没有的数字或产品；事实不足时如实说明。")

FALLBACK_SYSTEM = "你负责做一个判断。只输出 JSON，不要解释。"

JUDGE_SYSTEM = ("你是严格的评审。判断待评回答是否正确回答了客户问题，并且与参考回答的关键事实一致。"
                "不要求措辞相同，也不要求覆盖参考回答的全部细节；但如果有关键事实错误，或没有回答客户问的核心内容，判 0。"
                '只输出 JSON：{"score": 1 或 0, "reason": "一句话理由"}')


def _parse_json(text, default):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    try:
        return json.loads(text[text.find("{"): text.rfind("}") + 1])
    except ValueError:
        return default


def truncate_json(value, cap=20000):
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= cap else text[:cap] + f"…（已截断，原长 {len(text)} 字符）"


def llm_fallback(llm):
    """Fallback for uncertain System-1 decisions: one LLM call per decision."""
    def decide(kind, question, options, request):
        lines = "\n".join(f"- {k}: {v}" for k, v in options.items())
        fmt = {"choice": '输出 {"choice": "候选的键；都不符合时输出 NONE"}',
               "filter": '输出 {"keep": ["符合的候选键", ...]}',
               "noul": '输出 {"answer": true 或 false}'}[kind]
        out = llm.chat([{"role": "system", "content": FALLBACK_SYSTEM},
                        {"role": "user", "content": f"客户问题：{request}\n判断问题：{question}\n候选：\n{lines}\n{fmt}"}],
                       tag="fallback")
        data = _parse_json(out["message"]["content"], {})
        if kind == "choice":
            return data.get("choice", "NONE")
        if kind == "filter":
            return [k for k in data.get("keep", []) if k in options]
        return bool(data.get("answer", False))
    return decide


def _counts(ledger, mark):
    s = Ledger.summarize(ledger.since(mark))
    llm, jev = s.get("llm", {}), s.get("jev", {})
    return {"llm_calls": llm.get("calls", 0), "llm_in": llm.get("input_tokens", 0), "llm_out": llm.get("output_tokens", 0),
            "llm_latency": round(llm.get("latency", 0.0), 2), "jev_calls": jev.get("calls", 0),
            "jev_tokens": jev.get("input_tokens", 0), "jev_latency": round(jev.get("latency", 0.0), 3)}


# ---------------------------------------------------------------- ReAct baseline
class ReActAgent:
    name = "ReAct"

    def __init__(self, llm, conn, max_rounds=12, obs_cap=30000):
        self.llm, self.conn, self.max_rounds, self.obs_cap = llm, conn, max_rounds, obs_cap
        self.tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                        "parameters": t["input_schema"]}} for t in conn.list_tools()]

    def run(self, question, task_id=None, identity=None):
        mark, t0 = self.llm.ledger.mark(), time.perf_counter()
        if task_id:
            self.conn.set_task(task_id)
        messages = [{"role": "system", "content": REACT_SYSTEM}, {"role": "user", "content": question}]
        trace, answer, rounds = [], "", 0
        for rounds in range(1, self.max_rounds + 1):
            out = self.llm.chat(messages, tools=self.tools, tag="react")
            msg = out["message"]
            if not msg["tool_calls"]:
                answer = msg["content"]
                break
            messages.append({"role": "assistant", "content": msg["content"] or None, "tool_calls": msg["tool_calls"]})
            for c in msg["tool_calls"]:
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except ValueError:
                    args = {}
                r = self.conn.call(c["function"]["name"], args)
                text = r["text"] if len(r["text"]) <= self.obs_cap else r["text"][: self.obs_cap] + "…（已截断）"
                trace.append({"type": "tool", "name": c["function"]["name"], "args": args, "ok": r["ok"], "chars": len(r["text"])})
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": text})
        else:
            answer = "（达到最大轮数，未给出回答）"
        return {"method": self.name, "task_id": task_id, "answer": answer, "rounds": rounds, "trace": trace,
                "tool_calls": len(trace), "replay": self.conn.replay_log() if task_id else [],
                "wall": round(time.perf_counter() - t0, 2), **_counts(self.llm.ledger, mark)}


# ---------------------------------------------------------------- Jev-as-IF compiler agent
class JevAsIfAgent:
    name = "Jev-as-IF"

    def __init__(self, llm, conn, sog, jev, hops=None, stub_mode="sog", static_check=True, max_repairs=2,
                 runtime_repairs=1, facts_cap=20000):
        self.llm, self.conn, self.sog, self.jev = llm, conn, sog, jev
        self.hops, self.stub_mode, self.static_check = hops, stub_mode, static_check
        self.max_repairs, self.runtime_repairs, self.facts_cap = max_repairs, runtime_repairs, facts_cap
        self.ops = sog.ops()

    def visible_ops(self):
        if self.hops is None:
            return list(self.ops)
        return disclose(self.sog, self.hops)["operations"]

    def write_program(self, question, visible):
        stubs = render_stubs(self.sog, visible, mode=self.stub_mode)
        messages = [{"role": "system", "content": PYIF_SYSTEM.replace("{stubs}", stubs)},
                    {"role": "user", "content": f"客户问题：{question}\n（客户身份已绑定，userId 由运行时注入）"}]
        out = self.llm.chat(messages, tag="plan")
        code = pyif.extract_code(out["message"]["content"])
        checks = []
        for _ in range(self.max_repairs + 1):
            issues = pyif.check(code, self.ops, question, visible) if self.static_check else []
            checks.append([str(i) for i in issues])
            if not issues:
                break
            messages += [{"role": "assistant", "content": f"```python\n{code}\n```"},
                         {"role": "user", "content": "静态检查没有通过：\n" + "\n".join(f"- {i}" for i in issues)
                          + "\n请修改后输出完整程序。"}]
            code = pyif.extract_code(self.llm.chat(messages, tag="repair")["message"]["content"])
        return code, checks, messages, stubs

    def run(self, question, task_id=None, identity=None):
        mark, t0 = self.llm.ledger.mark(), time.perf_counter()
        if task_id:
            self.conn.set_task(task_id)
        visible = self.visible_ops()
        code, checks, messages, stubs = self.write_program(question, visible)
        result = {"method": self.name, "task_id": task_id, "code": code, "checks": checks,
                  "stub_chars": len(stubs), "visible_ops": len(visible)}
        if checks[-1]:
            result.update(answer="（程序没有通过静态检查）", error="static: " + "; ".join(checks[-1]))
        else:
            interp = pyif.Interpreter(self.ops, self.conn.call, self.jev, request=question, identity=identity or {})
            run = interp.run(code)
            runtime_repairs = 0
            while run.error and runtime_repairs < self.runtime_repairs:
                runtime_repairs += 1
                done = [f"{t['name']}({json.dumps(t['args'], ensure_ascii=False)[:200]}) ok={t['ok']}"
                        for t in run.trace if t["type"] == "tool"]
                messages += [{"role": "assistant", "content": f"```python\n{code}\n```"},
                             {"role": "user", "content": f"程序运行出错：{run.error}\n已执行的工具调用：\n" + "\n".join(done)
                              + "\n请修改程序，输出完整程序。"}]
                code = pyif.extract_code(self.llm.chat(messages, tag="runtime_repair")["message"]["content"])
                issues = pyif.check(code, self.ops, question, visible) if self.static_check else []
                if issues:
                    run.error = "static after runtime repair: " + "; ".join(map(str, issues))
                    break
                run = interp.run(code)
            result.update(code=code, runtime_repairs=runtime_repairs, error=run.error, trace=run.trace,
                          tool_calls=run.tool_calls, decisions=[d.__dict__ for d in self.jev.decisions],
                          facts_chars=len(json.dumps(run.facts, ensure_ascii=False, default=str)))
            if run.error and not run.facts:
                result["answer"] = "（程序执行失败）"
            else:
                out = self.llm.chat([{"role": "system", "content": ANSWER_SYSTEM},
                                     {"role": "user", "content": f"客户问题：{question}\n\n查到的事实（JSON）：\n"
                                      + truncate_json(run.facts, self.facts_cap)}], tag="answer")
                result["answer"] = out["message"]["content"]
        result["replay"] = self.conn.replay_log() if task_id else []
        result["wall"] = round(time.perf_counter() - t0, 2)
        result.update(_counts(self.llm.ledger, mark))
        return result


# ---------------------------------------------------------------- judge
def judge(llm, question, reference, answer):
    out = llm.chat([{"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": f"客户问题：{question}\n\n参考回答：\n{reference[:6000]}\n\n待评回答：\n{answer[:6000]}"}],
                   tag="judge")
    data = _parse_json(out["message"]["content"], {"score": 0, "reason": "无法解析评审输出"})
    return {"score": int(data.get("score", 0) or 0), "reason": data.get("reason", "")}
