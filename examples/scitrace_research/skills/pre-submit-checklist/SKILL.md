---
name: pre-submit-checklist
description: Writer 在写出或修订论文草稿前的逐项自查清单；核对必备章节、最低词数、RQ 小节、四表四图、引用结构与禁用词，全部通过后再交付。
---

# 论文提交前自查清单

## 目标

在交付草稿前完成结构与表述自查，使稿件进入确定性审计门禁时不再出现可预防的结构性错误。

- 核心目标：所有检查项 status=pass 后才提交。
- 关键成功条件：不引入任何输入工件之外的数字、区间或结论。

## 工作流

### 1. 章节结构自查

- 10 个必备章节齐全：Abstract、Introduction、Related Work、Method、Experimental Setup、Results、Discussion、Limitations、Conclusion、References。
- 正文体词数（表格、图注、引用条目、代码不计）：Method ≥ 300、Experimental Setup ≥ 180、Results ≥ 550、Discussion ≥ 180。
- Results 内含标注的 RQ1、RQ2、RQ3 三个子小节（RQ1: Matched-budget effectiveness；RQ2: Redistribution and query risk；RQ3: Sensitivity and boundary conditions）。

### 2. 表格与图自查

- Results 按出现顺序 4 张表、各配实质性 Table 1–4 图注：Table 1 主对比（含 candidate、baseline、difference、95% interval 列）；Table 2 描述性自适应基线（Raw Jaccard、Binned calibration、kNN allocation 列）；Table 3 查询结果计数（Wins、Ties、Losses、Rescued、Newly missed 列）；Table 4 特征消融（All features、Rank overlap、Jaccard only、RBO only 列）。
- 4 张图：figures/method_overview.pdf 嵌入 Method；figures/recall_by_budget.pdf、figures/budget_effect.pdf、figures/query_risk.pdf 嵌入 Results；均需自足图注（说明训练/测试边界或逐点区间语义）。

### 3. 引用自查

- References ≥ 6 个 evidence ID；正文以反引号 ID 引用且与文献表一一对应；不虚构文献，不引用未提供的条目。

### 4. 禁用词警示（正文任何位置、否定句也不得出现）

- 不得出现 "zero-shot"；FiQA 边界写作 "separately trained finance-domain lexical evaluation"。
- 不得将 allocation 与 pathways / sub-queries 并用；正确表述是 allocate document depth jointly across a known batch after fusion。
- 不得在 depth 附近使用 "continuous"（含否定句）；写 discrete integer depths 与 real-valued recall predictions。
- 不得出现 "preregistered / preregistration / pre-specified"（含否定句）；改为说明无 timestamped protocol 的选择过程。
- 需要确定性核对时运行 `scripts/checklist.py --draft <draft.md>`：输出 JSON 报告，全过 exit 0，任一失败 exit 1。

## 输出要求

- 任一未通过项必须在交付前修复；已正确的表数、图注与引用不得改写。
- 任何数字、区间、计数只来自输入工件，不得为描述性对比虚构置信区间。
