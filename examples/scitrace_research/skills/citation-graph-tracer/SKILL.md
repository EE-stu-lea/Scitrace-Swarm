---
name: citation-graph-tracer
description: 第一轮检索的相关文献不足或需要补充直接可比工作时的 OpenAlex 引文图一跳扩展流程；含种子选择、重过滤、去重与溯源规则。
---

# 引文图一跳扩展

## 目标

关键词检索未能覆盖足够直接相关文献时，沿 OpenAlex 引文图从高相关种子做一跳扩展，补充可溯源的新文献。

- 核心目标：只保留通过同一 focus_terms 门槛且能追溯到种子的新结果。
- 关键成功条件：扩展失败只降级为告警，绝不阻断文献主流程。

## 工作流

### 1. 选择种子

- 从已过滤的第一轮结果中，取相关性最高且带 OpenAlex work id（形如 W…）的前 3 篇作种子；没有任何可用 id 则跳过扩展（status=skipped）。

### 2. 一跳拉取

- 施引文献：请求 api.openalex.org/works?filter=cites:<W-id>，每种子上限 10 条。
- 相关文献：读取种子 raw_payload.related_works，逐个 GET /works/<W-id>，每种子上限 5 个。

### 3. 重过滤与合并

- 用与第一轮完全相同的 focus_terms 与 minimum_focus_terms 对候选重新过滤（title+summary 词干匹配）。
- 以 OpenAlex id / DOI / 规范化 title 三重身份键与基础结果及已保留候选去重；保留上限 20 条，按匹配分排序。

### 4. 溯源与审计

- 每篇保留文献记录其种子的 OpenAlex id（evidence metadata 的 expansion_seed_openalex_id）。
- 每次 API 调用计入 retrieval_call_count；literature_citation_expansion 审计事件写入 evidence/literature_events.jsonl。

## 决策规则

- 单请求超时 20 秒、重试 2 次；重试耗尽或无可用种子 → status=skipped，主流程继续使用基础结果。
- 扩展结果不得伪造 DOI、年份或摘要；解析失败的字段留空。
- 需要独立于流水线运行时执行：
  `scripts/expand_by_citations.py --seeds seeds.json --config filter.json --out expanded.json`（仅访问 OpenAlex；exit 0 = expanded，exit 1 = skipped）。

## 输出要求

- 扩展报告含 status、seed_openalex_ids、kept 列表、kept_provenance（work key → seed id）与 api_call_count。
- kept 中每条记录携带 title、authors、year、doi、source_url、summary 与自身 openalex_id。
