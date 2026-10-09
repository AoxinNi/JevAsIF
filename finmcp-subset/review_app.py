"""Local review tool for the D-group units (judgment decisions).

Run:  python finmcp-subset/review_app.py        (then the browser opens http://127.0.0.1:8777)
Every judgment is saved at once to review_labels.json and mirrored into review.csv.
Standard library only; nothing leaves this machine.
"""
import argparse
import json
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Notebook"))
sys.path.insert(0, str(HERE))
from jif import finmcp  # noqa: E402

import labels as store  # noqa: E402

PAGE = HERE / "review" / "index.html"
RESULT_CAP = 30000
LABEL_KEY = re.compile(r"(名称|简称|name|Name|title|标题)$")
PAGING = finmcp.PAGING_ARGS

# Chinese one-line descriptions, inferred from tool names, parameters and returned fields (FinMCP ships none).
TOOL_DESC = {
    "AnalyzeAssetLiability": "资产负债分析：按资产、负债和收支计算净资产与负债率",
    "AnalyzeCashFlow": "现金流规划：按收入、支出和家庭成员推演未来现金流与可投资金额",
    "AnalyzeFamilyMembers": "家庭成员分析：判断家庭所处的生命周期阶段",
    "AnalyzeFinancialIndicators": "财务指标分析：计算资产负债率、流动性比率、储蓄率等",
    "AnalyzeFundRisk": "基金风险分析：给出一组基金的风险评分",
    "AnalyzeInvestmentPerformance": "投资方案测算：按投资金额和预期收益评估方案可行性",
    "AnalyzePortfolioRisk": "持仓组合风险分析：给出一组持仓的风险评分",
    "BatchGetFundNavHistory": "批量查询基金净值走势",
    "BatchGetFundTradeLimit": "批量查询基金交易限制：能否申购、赎回、转换及最低金额",
    "BatchGetFundTradeRules": "批量查询基金交易规则：交易时间流程、费率规则",
    "BatchGetFundsDetail": "批量查询基金详情：运作信息、基金经理、资产配置、行业配置",
    "BatchGetFundsDividendRecord": "批量查询基金分红记录",
    "BatchGetFundsFeeRule": "批量查询基金费率：申购、赎回、运作费率",
    "BatchGetFundsHolderInfo": "批量查询基金持有人结构",
    "BatchGetFundsHolding": "批量查询基金持仓：股票、债券、基金投资明细",
    "BatchGetPoTradeComposition": "批量查询投顾组合的交易成分（按基金类别分组）",
    "BatchGetStrategiesComposition": "批量查询投顾策略的持仓成分（按基金类别分组）",
    "BatchGetStrategyRiskInfo": "批量查询投顾策略的风险等级与建议持有期",
    "DiagnoseFundPortfolio": "基金组合诊断：资产配置、相关性、回测",
    "GetAccountDailyProfits": "查询用户账户每日收益",
    "GetAccountProfitDetail": "查询用户某个账户的收益明细",
    "GetAnnouncementContent": "查询某条基金公告的正文",
    "GetAssetAllocation": "计算一组基金的大类资产配置",
    "GetAssetAllocationPlan": "按投资期限、目标收益和回撤生成资产配置方案",
    "GetBatchFundPerformance": "批量查询基金业绩：阶段收益与风险指标",
    "GetCompositeModel": "按配置方案编号查询具体的基金组合",
    "GetCurrentTime": "获取当前时间",
    "GetFundAnnouncements": "查询基金公告列表",
    "GetFundAssetClassAnalysis": "分析一组持仓基金的资产类别分布",
    "GetFundDiagnosis": "按名称或代码诊断单只基金",
    "GetFundLatestProfit": "查询用户持有基金的最新收益",
    "GetFundProfit": "查询用户持有基金在某段时间的收益",
    "GetFundsBackTest": "一组基金的历史回测",
    "GetFundsCorrelation": "一组基金之间的相关性",
    "GetLatestProfit": "查询用户持有投顾组合的最新收益",
    "GetPopularFund": "查询热门基金",
    "GetPortfolioNavHistory": "查询投顾组合的净值走势",
    "GetProfitProd": "查询用户持有产品在某段时间的收益",
    "GetStrategyAssetClassAnalysis": "查询投顾策略的资产类别分布",
    "GetStrategyDetails": "批量查询投顾策略详情：简介、管理人、收益、回撤等",
    "GetStrategyRiskInfo": "查询单个投顾策略的风险等级与建议持有期",
    "GetTxnDayRange": "计算前后若干个交易日的日期",
    "GetUserAssetBrief": "查询用户资产概况：总资产、收益",
    "GetUserAssetHoldingDistribution": "查询用户在指定基金或组合上的持仓分布",
    "GetUserAssetView": "查询用户资产视图：账户结构与持有产品",
    "GetUserInfo": "查询用户基本信息",
    "GetUserInvestDays": "查询用户投资天数",
    "GuessFundCode": "按基金名称或代码识别基金",
    "MonteCarloSimulate": "按资产权重做蒙特卡洛模拟，估计收益与回撤分布",
    "RenderEchart": "把图表配置渲染成图片",
    "RenderHtmlToPdf": "把 HTML 渲染成 PDF",
    "SearchFinancialNews": "按关键词搜索财经资讯",
    "SearchFunds": "按关键词和条件搜索基金",
    "SearchManagerViewpoint": "按关键词搜索基金经理观点",
    "StrategySearchByKeyword": "按关键词搜索投顾策略",
    "searchInvestAdvisorContent": "按关键词搜索投顾内容（文章、观点）",
}


def _short(v, n=60):
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[: n - 1] + "…"


def _calls(rec):
    """Calls of this unit in order, with their step number (one assistant message = one step)."""
    step_of, step = {}, 0
    for m in rec["messages"]:
        if m["role"] == "assistant" and m.get("tool_calls"):
            step += 1
            for c in m["tool_calls"]:
                step_of[c.get("id")] = step
    calls = [c for c in finmcp.tool_calls({"messages": rec["history"] + rec["messages"]}) if c["id"] in step_of]
    return calls, step_of


def _context(rec):
    if rec["unit"] == "turn":
        return "\n".join((m.get("content") or "") for m in rec["history"]) + "\n" + rec["question"]
    return finmcp.user_text(finmcp.get_task(rec["id"]))


def _flows(calls, context):
    """Per-call argument sources and the edges between calls (one edge per source call, target call, argument)."""
    seen, raw, sources = [], {}, {}
    for j, call in enumerate(calls):
        for arg, value in call["args"].items():
            tags = set()
            for v in (value if isinstance(value, list) else [value]):
                if isinstance(v, (dict, list)) or v is None:
                    tags.add("复合值")
                    continue
                nv = finmcp.norm(v)
                if arg == "userId":
                    tags.add("身份（运行时注入）")
                elif arg in PAGING:
                    tags.add("分页参数")
                elif nv and nv in context:
                    tags.add("客户原话")
                elif not finmcp.identifier_like(v):
                    tags.add("常量")
                else:
                    hit = None
                    for i, lv in reversed(seen):
                        path = next((p for p, _, x in lv if finmcp.norm(x) == nv), None)
                        if path:
                            hit = (i, path, len({finmcp.norm(x) for p, _, x in lv if p == path}))
                            break
                    if hit:
                        e = raw.setdefault((hit[0], j, arg, hit[1]), {"values": set(), "n": hit[2]})
                        e["values"].add(nv)
                        tags.add(f"上游 #{hit[0] + 1}")
                    else:
                        tags.add("常量")
            sources[(j, arg)] = sorted(tags)
        if call["json"] is not None:
            seen.append((j, finmcp.leaves(call["json"])))
    edges = []
    for (i, j, arg, path), e in raw.items():
        k, n = len(e["values"]), e["n"]
        kind = "reference" if n <= 1 else "pick_one" if k == 1 else "pick_all" if k >= n else "pick_subset"
        edges.append({"from": i, "to": j, "arg": arg, "path": path, "n": n, "k": k, "kind": kind,
                      "values": sorted(e["values"])[:12]})
    for (j, arg), tags in sources.items():
        if "客户原话" in tags:
            edges.append({"from": -1, "to": j, "arg": arg, "kind": "context"})
    return sources, edges


def _candidates(d):
    rows, seen = [], set()
    for item in d["candidates"]:
        key = finmcp.norm(item.get(d["id_key"])) if isinstance(item, dict) and d["id_key"] else finmcp.norm(item)
        if key in seen:
            continue
        seen.add(key)
        label, extra = "", []
        if isinstance(item, dict):
            label = next((str(v) for k, v in item.items() if LABEL_KEY.search(k) and isinstance(v, str) and v), "")
            extra = [f"{k}: {_short(v, 40)}" for k, v in item.items()
                     if k != d["id_key"] and isinstance(v, (str, int, float)) and not isinstance(v, bool)
                     and str(v).strip() and str(v) != label and len(str(v)) <= 60][:3]
        rows.append({"key": key, "label": label, "extra": extra, "chosen": key in d["gold"]})
    return rows


def build_payload():
    units = [json.loads(line) for line in (HERE / "tasks.jsonl").open(encoding="utf-8")]
    units = [u for u in units if u["group"] == "D"]
    decisions = {}
    for line in (HERE / "decisions.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        decisions.setdefault(d["unit_id"], []).append(d)

    def order(u):
        return (u["split"] != "dev", u["source"], [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", u["id"])])

    out = []
    for u in sorted(units, key=order):
        calls, step_of = _calls(u)
        sources, edges = _flows(calls, _context(u))
        call_rows = []
        for j, c in enumerate(calls):
            text = json.dumps(c["json"], ensure_ascii=False, indent=1) if c["json"] is not None else (c["text"] or "")
            call_rows.append({"index": j, "step": step_of[c["id"]], "name": c["name"],
                              "desc": TOOL_DESC.get(c["name"], ""),
                              "args": [{"name": k, "value": _short(v, 300), "sources": sources.get((j, k), [])}
                                       for k, v in c["args"].items()],
                              "chars": len(c["text"] or ""), "error": c["json"] is None or '"error"' in (c["text"] or "")[:300],
                              "result": text[:RESULT_CAP], "truncated": len(text) > RESULT_CAP})
        decs = [{"kind": d["kind"], "src": f"{d['src_tool']}.{d['src_path']}", "dst": f"{d['dst_tool']}.{d['dst_arg']}",
                 "n": d["n"], "gold": d["gold"], "id_key": d["id_key"], "rows": _candidates(d)}
                for d in decisions.get(u["id"], [])]
        if not decs:   # extraction found no candidate records; fall back to the screening summary
            decs = [{"kind": d["kind"], "src": d["src"], "dst": d["dst"], "n": d["n_candidates"], "gold": d["chosen"],
                     "id_key": None, "rows": []} for d in u["decisions"]]
        history = [{"role": m["role"], "text": _short(m.get("content") or "", 600) if m["role"] != "assistant" or not m.get("tool_calls")
                    else "调用 " + "、".join(c["function"]["name"] for c in m["tool_calls"])} for m in u["history"]]
        out.append({"id": u["id"], "split": u["split"], "source": u["source"], "unit": u["unit"],
                    "question": u["question"], "user_id": u["user_id"], "scenery": u["scenery"],
                    "difficulty": u["difficulty"], "n_calls": u["n_calls"], "n_steps": u["n_steps"],
                    "has_parallel": u["has_parallel"], "flags": u["flags"], "decisions": decs, "calls": call_rows,
                    "edges": edges, "answer": u["reference_answer"], "history": history})
    return {"units": out, "semantic_options": store.SEMANTIC, "reference_options": store.REFERENCE}


class Handler(BaseHTTPRequestHandler):
    payload = b""
    labels = {}
    lock = threading.Lock()

    def log_message(self, *args):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/data":
            self.send(200, Handler.payload)
        elif self.path == "/api/labels":
            self.send(200, Handler.labels)
        else:
            self.send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/labels":
            return self.send(404, {"error": "not found"})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        unit_id = body.get("id")
        if not unit_id:
            return self.send(400, {"error": "missing id"})
        with Handler.lock:
            store.update(Handler.labels, unit_id, body.get("label"))
            saved = Handler.labels.get(unit_id)
        self.send(200, {"ok": True, "label": saved})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    print("准备数据……")
    Handler.payload = json.dumps(build_payload(), ensure_ascii=False).encode()
    Handler.labels = store.load()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"审核页面：{url}\n判断保存在：{store.LABELS}（同步到 {store.REVIEW_CSV.name}）\n按 Ctrl+C 退出")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()
