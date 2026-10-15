# 统计表述规范详解（notation-rules）

本参考展开 SKILL.md 各条规则的判定要点，并给出英文正文风格的正误例句。规则本身以 SKILL.md 为准；本文件供 Writer 修订与 Auditor 审查时查阅。

## 1. 逐点区间语义

paired-bootstrap 95% interval 是对每个 budget 单独重采样查询得到的逐点区间，条件于已实现的排序与分配。

- 正确："Every budget has pointwise paired-bootstrap 95% intervals that condition on the realized batch allocation; they are not simultaneous intervals."
- 正确："The intervals describe uncertainty over query resampling for this fixed allocation, not for reallocating a fresh batch."
- 错误："The 95% confidence bands show simultaneous coverage across all budgets."
- 错误："The interval tells us how the method would perform on a new batch of queries."

## 2. macro 主张范围

主结论 = SciFact 与 NFCorpus 在 budget 5 的非加权宏平均；SciFact 单独显著（次要结果）；NFCorpus 不确定。

- 正确："The primary supported result is the unweighted macro mean over SciFact and NFCorpus at budget 5; SciFact is individually significant as a secondary result, while NFCorpus is inconclusive."
- 错误："The method significantly improves retrieval on both datasets."
- 错误："Do not extend the effect to any individual dataset."（SciFact 有单独显著结果，不能一概禁止外推。）
- 边界：FiQA 是 separately trained finance-domain lexical evaluation，只作边界条件讨论，不进入 macro，也不得称为 zero-shot transfer。

## 3. 描述性对比用词

自适应基线与特征消融只有点估计，没有配对区间：

- 正确："The kNN allocation attains higher descriptive recall than the raw Jaccard baseline on SciFact; no paired interval was computed for this comparison."
- 正确："The rank-overlap-only ablation slightly exceeds the full feature model on SciFact, so the all-feature representation is not uniformly best."
- 错误："The kNN allocation significantly outperforms the binned calibration."
- 错误："Feature ablations are statistically comparable across datasets."

## 4. 数字来源与精度

- 所有数字仅来自输入工件字段（required_results_table_rows、required_adaptive_baseline_rows、required_query_outcome_rows、required_train_subsampling_summary、feature_ablation recalls）。
- 显示值 4–6 位小数、全文一致；精确值仅存于 Structured References JSON block。
- 错误示例：为 Table 2 / Table 4 补造 "±" 或区间；引用工件未提供的 compute savings、latency 或 answer-quality 数字。

## 5. 种子与子集表述

- 正确："The primary experiment and bootstrap used seed 42; the five 80% training-subset draws used seeds 42 through 46 and form a limited sensitivity check, not five independent model seeds."
- 错误："All runs use seed 42."（与子集抽样种子矛盾。）
- 错误："We reran the experiment five times."（子集检查在已完成的实验内部完成，未重跑流水线。）

## 6. Auditor 应用方式

- 发现越界统计语言（significant 滥用、区间语义错误、macro 外推）：severity=error，quote 原句，suggested_fix 给出符合上述规则的替换句。
- 仅风格问题（如小数位数不齐）：severity=warning；不得要求新实验或新测量。
- 无可行动问题返回 findings=[]，不虚构问题。
