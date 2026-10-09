# Jev-as-IF：基于图的 Agent 工具编译器

agent 依据对象/操作分体图（SOG）一次写出调用工具的程序，程序里的语义判断由 TypeSafe 的 System-1 模型 Jev 执行，从而减少大模型轮次、降低延迟和成本。目标会议 WWW '27。

| 目录 | 内容 |
| --- | --- |
| `research/` | 方向调研与差异化定位 |
| `Texpaper/` | 论文 LaTeX 稿（ACM 格式） |
| `Experiments/` | 实验设计 |
| `finmcp-analysis/` | FinMCP-Bench 原始数据与分析脚本 |
| `finmcp-subset/` | 筛选出的任务子集、判断题、人工审核工具 |
| `finmcp-graph/` | 按 ZSZQ 组合详解 v0.4 schema 建的本体（对象图 + 动作图）与可视化页面 |
| `Notebook/` | 预实验 notebook 与 `jif` 代码包 |
| `jev-playground/` | Jev 用法示例 |

## 配置

API key 填在 `Notebook/keys.py`（已加入 `.gitignore`，不会提交），格式见 `Notebook/README.md`；填完后运行 `python Notebook/check_keys.py` 检查连通。

## 数据许可

FinMCP-Bench 数据及本仓库中由其衍生的数据（`finmcp-analysis/data`、`finmcp-subset`、`finmcp-graph` 中的统计与映射）沿用 CC BY-NC-SA 4.0，仅限非商业用途。
