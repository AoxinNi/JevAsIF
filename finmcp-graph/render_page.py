"""Render graph.html (object graph + action graph viewer) from graph-data.json.

Run after build_graph.py:  python finmcp-graph/render_page.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Presentation-only grouping (colour) of objects and actions; not part of the ontology.
DOMAINS = {
    "cust": "客户与资产", "strat": "投顾策略", "fund": "基金", "plan": "配置与分析", "info": "资讯观点", "aux": "时间与条件",
}
OBJECT_DOMAIN = {
    **dict.fromkeys(["Customer", "Account", "AssetOverview", "Holding", "ProfitRecord"], "cust"),
    **dict.fromkeys(["Strategy", "Component"], "strat"),
    **dict.fromkeys(["Fund", "FundManager", "NavPoint", "Performance", "RiskRating", "FeeRule", "TradeRule",
                     "Dividend", "FundHolding", "HolderStructure", "Announcement"], "fund"),
    **dict.fromkeys(["AllocationPlan", "AssetClass", "PortfolioAnalysis", "FinancialProfile"], "plan"),
    "Article": "info",
    **dict.fromkeys(["MarketCalendar", "QueryCondition"], "aux"),
}
ACTION_DOMAIN = {
    **dict.fromkeys(["GetUserInfo", "GetUserInvestDays", "GetUserAssetBrief", "GetUserAssetView",
                     "GetUserAssetHoldingDistribution", "GetLatestProfit", "GetFundLatestProfit", "GetProfitProd",
                     "GetAccountDailyProfits", "GetAccountProfitDetail", "GetFundProfit"], "cust"),
    **dict.fromkeys(["StrategySearchByKeyword", "GetStrategyDetails", "GetStrategyRiskInfo", "BatchGetStrategyRiskInfo",
                     "GetStrategyAssetClassAnalysis", "BatchGetStrategiesComposition", "BatchGetPoTradeComposition",
                     "GetPortfolioNavHistory"], "strat"),
    **dict.fromkeys(["SearchFunds", "GuessFundCode", "GetPopularFund", "BatchGetFundsDetail", "GetFundDiagnosis",
                     "GetBatchFundPerformance", "BatchGetFundNavHistory", "AnalyzeFundRisk", "BatchGetFundsFeeRule",
                     "BatchGetFundTradeRules", "BatchGetFundTradeLimit", "BatchGetFundsDividendRecord",
                     "BatchGetFundsHolding", "BatchGetFundsHolderInfo", "GetFundAnnouncements",
                     "GetAnnouncementContent"], "fund"),
    **dict.fromkeys(["SearchFinancialNews", "SearchManagerViewpoint", "searchInvestAdvisorContent"], "info"),
    **dict.fromkeys(["GetAssetAllocationPlan", "GetCompositeModel", "GetFundAssetClassAnalysis", "GetAssetAllocation",
                     "DiagnoseFundPortfolio", "GetFundsBackTest", "GetFundsCorrelation", "AnalyzePortfolioRisk",
                     "MonteCarloSimulate", "AnalyzeAssetLiability", "AnalyzeCashFlow",
                     "AnalyzeInvestmentPerformance"], "plan"),
    **dict.fromkeys(["GetCurrentTime", "GetTxnDayRange", "close_consultation"], "aux"),
}


def page_data():
    d = json.loads((HERE / "graph-data.json").read_text(encoding="utf-8"))
    o = d["ontology"]
    objects, actions = o["objectOntology"]["nodes"], o["operationOntology"]["nodes"]
    assert set(OBJECT_DOMAIN) == {n["id"] for n in objects}
    assert set(ACTION_DOMAIN) == {n["id"] for n in actions}
    return {
        "meta": {k: o[k] for k in ("id", "label", "version", "executionMode", "semanticVersion", "disclosure", "globalRules")}
        | {"objectEntry": o["objectOntology"]["entry"], "actionEntry": o["operationOntology"]["entry"]},
        "objects": objects, "relations": o["objectOntology"]["edges"], "actions": actions,
        "mapping": d["mapping"]["tools"], "edges": d["edges"], "usage": d["usage"], "report": d["report"],
        "domains": DOMAINS, "objectDomain": OBJECT_DOMAIN, "actionDomain": ACTION_DOMAIN,
    }


def main():
    body = (HERE / "graph_template.html").read_text(encoding="utf-8")
    data = json.dumps(page_data(), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    body = body.replace("__GRAPH_DATA__", data)
    # graph_template.html is a page fragment (the artifact host adds the document skeleton);
    # the local copy gets its own so it opens directly in a browser.
    (HERE / "graph.html").write_text(
        '<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head>\n<body>\n'
        + body + "\n</body></html>\n", encoding="utf-8")
    (HERE / "graph.fragment.html").write_text(body, encoding="utf-8")
    print("wrote", HERE / "graph.html")


if __name__ == "__main__":
    main()
