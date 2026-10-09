# FinMCP 顾问本体（对象图 + 动作图）

按 ZSZQ 运行包里"组合详解"v0.4 本体（`business-strategy-detail-object-v04.json`）的 schema，对 `../finmcp-subset/` 筛选出的 324 个单元建图：

- **对象图**：问题和工具返回里的名词性概念；
- **动作图**：子集实际用到的 52 个工具。

```bash
python finmcp-graph/build_graph.py    # 生成本体、映射、报告；需要 ../Notebook/jif 和 ZSZQ 运行包
python finmcp-graph/render_page.py    # 生成 graph.html，浏览器直接打开
```

## 结果

| 项目 | 数量 |
| --- | --- |
| 对象 | 25 个，260 个属性 |
| 对象关系 | 39 条，全部对象可从入口 `Customer` 到达 |
| 动作 | 52 个，另有 1 个结束节点 `close_consultation` |
| 可追踪属性 | 7 个：既被产出、又被消费的属性；动作之间的全部依赖只经过它们 |
| ZSZQ 编译器 | `ontology_mcp.compiler.compile_map` 检查通过 |
| 数据流覆盖 | 99.5%（3,351 / 3,367），其中"挑选" 99.7%，"直接引用" 98.6% |

**数据流覆盖**的算法：参考轨迹里每一次"上游返回值 → 下游参数"，如果两端被映射到同一个 对象.属性，这次依赖就算这张图能解释。剩下没覆盖的十几次，主要是参考轨迹本身的问题，例如把策略代码当作基金代码传给基金公告接口。

依赖高度集中：

- `Fund.fundCode`：2,486 次，其中 2,362 次要从候选中挑选；
- `Strategy.strategyCode`：765 次，其中 339 次要挑选。

这两个属性就是 Jev-as-IF 主要发挥作用的位置。

## 文件

| 文件 | 内容 |
| --- | --- |
| `finmcp-object-v04.json` | 本体，结构与组合详解 v0.4 相同（顶层字段、对象节点、关系边、动作节点、结束节点） |
| `finmcp-tool-mapping.json` | 工具注册映射，与 ZSZQ `tool-registry.json` 同构，见下 |
| `report.json` | 统计、编译结果、数据流覆盖和未覆盖的数据流 |
| `graph-data.json` | 页面用的汇总数据：本体、映射、推导出的数据流边（附观测次数）、工具调用次数 |
| `graph.html` | 可视化页面：本体视图（对象本体与操作本体同图）和 Schema 对照两个标签页，见下 |
| `build_graph.py` / `render_page.py` / `graph_template.html` | 生成脚本和页面模板 |

### 本体视图的画法

沿用 ZSZQ 最小运行包 v0.8 工作台（`workbench/graph-layout.js`）的本体视图：

- 左边是操作本体（动作），右边是对象本体（对象），两区分开画；
- 两区之间只按动作的 `input` / `output` 连线：
  - 输入参数是 `对象.属性`，画成对象 → 动作；
  - 输出是 `对象.属性`，画成动作 → 对象；
  - 同一个对象既是输入又是输出时，画一条双向线。
- 同一个动作和同一个对象之间只画一条线，具体属性在悬停或选中时显示。

和工作台不同的两处：

- 工作台把所有动作排在一列。这里有 52 个动作，所以按业务领域分成横带，动作在带内按所连对象的位置排序；
- 对象关系画成对象列右侧的弧线。

另外，回显不画成输出线；客户标识和查询条件的输入画成虚线，可以关掉。

### 工具注册映射

本体里只出现中文语义和 `对象.属性`。工具名、参数名、返回字段名都放在 `finmcp-tool-mapping.json`：

- `params`：每个工具参数对应一个本体属性，包括：
  - `apiField`：工具原参数名；
  - `fill`：填写方式，`identity` 表示会话注入，`agent` 表示智能体填写；
  - `check`：`traced` 表示取值必须来自上游输出。
- `fields`：每个输出属性对应返回结果里的字段路径。路径写法：
  - `[*]` 表示列表元素；
  - `<code>`、`<category>` 等表示以代码或类别为键的字典；
  - `**` 表示任意层级。
- `constants`：固定取值、不暴露给智能体的参数；
- `dropped`：不建模的参数，如分页大小。

## 建模取舍

1. **入口动作是 `GetCurrentTime`**。组合详解以确定服务范围的工具作入口，FinMCP 没有这类工具。相对日期都要以当前时间换算，所以用它作入口。
2. **查询条件单独成为对象 `QueryCondition`**。关键词、日期、分页、排序等都放在这里，因为 v0.4 要求每个参数都是某个对象的属性。
3. **投顾组合持仓的产品代码就是策略代码**。所以：
   - 持仓、收益类工具同时产出 `Holding.productCode` 和 `Strategy.strategyCode`；
   - `GetLatestProfit`、`GetProfitProd` 的 `prodCodeList` 绑定到 `Strategy.strategyCode`。
4. **备选基金代码也作为 `Fund.fundCode` 产出**。组合成分里的备选基金代码，下游会拿去查费率、交易规则。
5. **`GetFundDiagnosis.fundNameOrCode` 绑定 `Fund.fundCode`**。子集里它的值多数来自上游返回的基金代码。只有 `GuessFundCode` 接收客户原话里的名称，即 `QueryCondition.fundNameOrCode`。
6. **回显不算数据来源**。回显指工具把输入原样带回，例如批量查询返回里的 `fundCode`。映射保留这些字段，但动作图不把回显画成数据来源。因此 `Announcement.announcementType` 虽然在编译器的可追踪列表里，实际只有回显，图中不单独画出。
7. **对象和动作的颜色分组只用于展示**，不属于本体。分组定义在 `render_page.py`。

## 许可

源数据为 FinMCP-Bench（CC BY-NC-SA 4.0），本文件夹里的衍生数据沿用同一许可，仅限非商业用途。
