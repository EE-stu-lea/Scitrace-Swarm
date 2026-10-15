---
name: stats-notation
description: Writer 与 Auditor 共用的统计表述规范；约束置信区间语义、macro 主张范围与描述性对比用词，防止越界统计语言。
---

# 统计表述规范

## 目标

统计证据的措辞必须与实验工件严格一致：只有真正计算过区间或检验的比较才允许使用确定性统计语言。

- 核心目标：区间语义、主张范围与显著性用词全部可溯源到输入工件。
- 关键成功条件：描述性对比绝不使用 significant / significance 语言。

## 决策规则

- paired-bootstrap 95% 区间一律写 "pointwise paired-bootstrap 95% intervals"：逐点区间、条件于已实现的 batch allocation；不得写成 simultaneous intervals，也不得外推为新批次重分配的不确定度。
- 主结论是 SciFact 与 NFCorpus 在 budget 5 的 unweighted macro mean；SciFact 是单独显著的次要结果，NFCorpus 是 inconclusive；既不得把 macro 结论当作每个数据集都显著，也不得声称结论对任何单个数据集都不成立。
- 自适应基线（Raw Jaccard、Binned calibration、kNN allocation）与特征消融是 descriptive comparisons：未计算对它们的置信区间，只用 higher / lower / descriptive 等词，禁用 significant。
- 全部数字只来自输入工件（required_results_table_rows、required_train_subsampling_summary、required_adaptive_baseline_rows、required_query_outcome_rows）；为描述性对比补造区间视为造假。
- 显示值保持 4–6 位小数的一致精度；精确值仅保留在 Structured References JSON block，正文不得提及该隐藏块或内部文件路径。
- 五个 80% training-query subset 是有限的敏感性检查，不是五个独立模型种子；主实验/bootstrap 用 seed 42，子集抽样用 seeds 42–46，两处表述必须区分。
- Auditor 发现违反上述措辞的问题必须报 error 并附 quote 与 suggested_fix；Writer 修订时不得改写已正确的统计表述。

详细判定要点与正误例句读取：`references/notation-rules.md`。

## 输出要求

- 正文中的每个区间、显著性判断和宏平均数字都能指回输入工件字段。
- Auditor 只在有证据支撑的 correctness/completeness 缺陷上报 error；不得要求补造未提供的测量。
