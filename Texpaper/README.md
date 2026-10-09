# 论文草稿：Jev-as-IF（目标 WWW '27）

题目：*Jev-as-IF: Graph-Guided Compilation of Agent Tool Chains for the Web*

按 ACM Web Conference 2027 研究论文格式（`acmart` sigconf，双盲评审模式）写的完整版草稿。实验部分只有结构，结果全部留空。实验方案见 [../Experiments/实验设计.md](../Experiments/实验设计.md)。

## 主线

Jev-as-IF 是一个**基于图的工具编译器**：agent 根据图提供的信息，一次写出受限 Python 子集（PyIF）的代码，语义判断写成 `jev.choice` / `jev.noul` / `jev.filter` 调用；编译器对照图做检查、优化并降为可执行流程；运行时由执行器调用工具，由 Jev 判断代码里的条件。两个关键设计：

1. **图引导（SOG，对象和操作分开建图）**：编译器的输入和类型系统。操作图提供类型化接口桩和数据流检查，对象图为 Jev 的判断提供语义。
2. **Jev-as-IF（在代码里调用 System-1 原语）**：代码里语义条件的运行时求值方式，配合共形门控和一次回退。

选择型绑定的占比只作为数据集筛选，放在第 6.1 节。

## 投稿信息

| 项目 | 内容 |
| --- | --- |
| 会议 | The ACM Web Conference 2027，都柏林，2027-05-10 至 05-14 |
| 截止 | 摘要 2026-10-18，全文 2026-10-25（AoE，不延期） |
| 建议 track | 9. Web Infrastructure and Agentic Systems（备选：7. Semantics and Knowledge，适合以 SOG 为主打时） |
| 篇幅 | 正文 8 页 + 参考文献 + 附录，总计不超过 12 页 |
| 硬性要求 | 第一页必须写明与 Web 和所投 track 的关系（引言第一段已写） |

## 文件

| 文件 | 内容 |
| --- | --- |
| `main.tex` | 主文件；`\sog`、`\jif`、`\pyif` 等命名宏在这里统一修改 |
| `sections/00-abstract.tex` | 摘要 |
| `sections/01-introduction.tex` | 引言：写代码的 agent 还剩两个障碍（语义判断要回大模型；对陌生工具写代码只能靠猜）；编译器与两个关键设计；贡献 |
| `sections/02-motivation.tex` | 背景：一个步骤的输入有哪几种来源（表 1）；System-1 模型 |
| `sections/03-sog.tex` | **图**：SOG 的定义、为什么要分开（表 2）、按对象范围披露并渲染成接口桩、建图流程 |
| `sections/04-compiler.tex` | **编译器**：PyIF 语言与 `jev.*` 原语、基于图的静态检查、用对象图构造问题、优化与降为流程、共形门控、成本 |
| `sections/05-implementation.tex` | 实现：基于本体运行时（匿名） |
| `sections/06-experiments.tex` | 实验：任务筛选、RQ1–RQ4、错误类型表、以 PTC 为基线的析因表（**留白**） |
| `sections/07-related.tex` | 相关工作：代码执行式方法放在第一段，并说明"用图选工具、做规划"和"用图编译"的区别 |
| `sections/08-discussion.tex` | 讨论与局限 |
| `sections/09-conclusion.tex` | 结论 |
| `sections/appendix.tex` | 附录：程序示例、筛选方法、建图细节、提示词 |
| `references.bib` | 参考文献 |
| `acmart.cls`、`ACM-Reference-Format.bst` | ACM 模板 |
| `main.pdf` | 当前编译结果（8 页） |

## 编译

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

本机 BasicTeX 缺的宏包已用 `tlmgr --usermode` 装在 `~/Library/texmf`。也可以上传到 Overleaf 编译。

## 投稿前待办

1. **实验**：第 6 节全部内容，以及摘要和结论里的核心结果句。
2. **筛选统计**：人工核实后替换表 3（目前是启发式结果）。
3. **建图流程**：§3.4 的建图流程要和实际实现一致，补上派生边召回率的指标（附录 C）。
4. **参考文献**：标了 `TODO verify` 的条目要核对，`navi2025`、`gap2025` 还没有作者。可以用 `/ars-citation-check`。
5. **匿名化**：第 5 节不能出现公司、产品或部署名称。
6. **概览图**：图 1 现在是文字占位框。
7. **CCS 概念编号**和**生成式 AI 使用声明**。
8. 删除 `\todo` 和 `\tbd` 宏，确认最终页数。
