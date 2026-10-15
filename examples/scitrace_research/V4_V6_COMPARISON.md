# SciTrace-Swarm v4 与 v6 运行对比

两轮 strict-real 全流程运行的逐一对照：**v4**（零 skill 基线，提交演示）与 **v6**（writer-only skill 注入 + 引文扩展修复，消融处理组）。三轮消融的完整叙事（含 v5）见 `ABLATION.md`，本文聚焦 v4 vs v6 的两两差异。

所有数字与运行产物逐一核对（`run_manifest.json`、`resource_usage.json`、`validation_report.json`、`evidence/literature_events.jsonl`、`team_prompt_audit.jsonl`），复核路径见第 9 节。

## 1. 一句话结论

v4 以最短路径（2 轮修订、9 次调用）原生通过并导出 PDF；v6 首轮质量更优（14 vs 16 个错误）、证据池多 7 条溯源文献，但被确定性门禁自身的否定盲区误报阻断，修复豁免逻辑后经 0 次模型调用的离线重校验翻转 success 并导出 PDF。两份论文产物均可用，v4 为提交演示首选。

## 2. 公共实验条件（完全一致）

| 项 | 值 |
|---|---|
| 模式 | `real_team` + `strict-real` + `no-mock-fallback`（两轮 `fixture_used` / `mock_fallback_used` 均为 false） |
| 模型 | DeepSeek `deepseek-v4-pro`，`temperature: 0.0` |
| Topic | Continuous difficulty calibration for budgeted hybrid scientific evidence retrieval |
| 一致性循环 | 审稿人 + 确定性门禁双通道，门禁错误归零才放行导出 |
| 运行日期 | 2026-10-14 |

## 3. 配置差异（全部变量）

| 配置项 | v4 | v6 |
|---|---|---|
| writer 侧 skill（draft/revision 阶段） | 无 | pre-submit-checklist + stats-notation |
| review 侧 skill | 无 | 无（保持干净） |
| OpenAlex 引文扩展 | 关 | 开（三层隔离修复后） |
| `revision.max_rounds` | 2 | 3（resume 时放宽） |
| `budget.max_model_calls` | 12 | 17（resume 时放宽） |

## 4. 核心指标对照

| 指标 | v4 | v6 |
|---|---|---|
| 一致性轨迹（门禁+审稿错误数） | 16 → 2 → **0** ✓ | **14** → 3 → 2 → 2 ✗ → 门禁修复后离线重校验 **0** ✓ |
| 实际修订轮次 | 2 | 3（含 1 轮 resume 续跑） |
| 模型调用 | 9 | 17（`resumed_run: true`，含失败阶段重放与 1 次 retry，非纯净单次成本） |
| 总 token | 217,320（输入 188,096 / 输出 29,224） | 494,300（输入 436,435 / 输出 57,865），约 v4 的 2.28× |
| 真实检索调用 | 10（literature_search） | 23 = 10 search + 13 citation_expansion |
| 文献证据池 | 10 条 | **17 条**（+7 篇采纳，全部带 seed 溯源） |
| 墙钟时长 | 886 s（06:21:49Z–06:36:35Z） | 活跃 816 s + resume 263 s（`resource_usage.json` 的 13,898 s 含 resume 间隙，其 `runtime_accounting` 字段已注明） |
| 最终状态 | success | success（`revalidated_from_failure: true`，`model_calls_during_revalidation: 0`） |
| draft_sha256 | `99672d57…` | `234f944c…` |

## 5. 首轮质量：writer 侧 skill 的可见改善（16 vs 14）

两轮 round 0 的确定性门禁错误构成（每条均被审稿人复述一次，故总错误 = 2×确定性数）：

| round-0 确定性错误 | v4 | v6 |
|---|---|---|
| `paper_results_table_caption_missing`（Table 1–4 caption 缺失） | 4 | 4 |
| `paper_references_insufficient` + `paper_in_text_citations_insufficient`（5 条可溯源引用 < 6） | 2 | 2 |
| `paper_section_underdeveloped`（Discussion 141 词 < 180） | 1 | **0** |
| `paper_context_budget_missing`（budget 未按 evidence-context 口径定义） | 1 | **0** |
| `paper_pathway_allocation_claim`（方法范围表述错误） | 0 | 1 |
| **合计** | **8**（总 16） | **7**（总 14） |

三点观察：

1. v6 首轮**规避**了 v4 的"Discussion 字数不足"与"budget 定义缺失"两类错误——二者恰与 pre-submit-checklist 的检查项对应，是 writer 侧 skill 最具体的正向证据；首轮正文也更充实（3,212 词 vs v4 的 2,625 词）。
2. 改善是**部分**的：表格 caption 与参考文献引用两类错误两轮完全相同，skill 并未阻止 writer 首轮漏写 caption。
3. v6 新增了 pathway 表述错误（writer 措辞与 method 契约有张力），该错误随后暴露了门禁自身的缺陷（见第 7 节）。

注意：单轮对照（n=1），16→14 指示方向，不构成统计显著结论。

## 6. 修订收敛路径对比

- **v4**：16 → 2 → 0。round 1 剩余 2 条错误**均为审稿人假阳性**（finding 原文自述 "No numerical inconsistency is present … No change required"）；round 2 审稿降为 info、门禁归零，通过。
- **v6**：14 → 3 → 2 → 2。writer 每轮单调改善，round 3 已把审稿人处方**逐字**写入 draft（修正句被门禁正确豁免）；剩余 1 条确定性错误 `paper_pathway_allocation_claim` 为误报，另 1 条 reviewer finding 是指向同一句的自指性复述——修订循环被**验证器缺陷**而非写作残留阻断。

对比结论：v6 writer 的收敛能力不弱于 v4（每轮错误严格下降且 round 1 即从 14 降到 3），"差一轮收敛"的表象是门禁 bug 所致。两轮共同暴露的另一个事实：审稿人假阳性在 v4 也出现过，只是未被确定性门禁放大成死循环——决定成败的是确定性门禁的豁免质量，而非审稿噪声本身。

## 7. v6 独有增益与代价

**引文扩展**（三层隔离修复后的生产验证）：

- 13 次真实 OpenAlex 扩展调用、27 个候选、**7 篇采纳且全部带 seed 溯源**（`kept_provenance`）；
- 证据池 10 → 17 条。两轮 round 0 都只有 5 条可溯源引用（瓶颈在 writer 的引用行为而非证据供给），但 v6 修订时有充足的真实来源可引，v4 曾因此触发 `paper_references_insufficient`。

**门禁否定盲区（v6 失败根因，已闭环）**：`paper_pathway_allocation_claim` 的豁免只扫匹配前 60 字符前缀、词表缺 rather than 等修正句式，对 3 句科学正确的否定/对比句误报——与 `method_contract._negated` 同一 bug 类。修复豁免语义（前缀 ∪ 匹配区间内、术语之前）并补回归单测后，`revalidate_run.py` 在**不改 draft 一个字节、0 次模型调用**下重放门禁 → 0 错误 → manifest 翻转 success → PDF 导出。原始失败产物五件套以 `revalidation_original_*` 封存。

## 8. 最终论文产物对比

| | v4 | v6 |
|---|---|---|
| 标题框架 | Continuous difficulty calibration … | Predicted-recall allocation … |
| 终稿字数（门禁统计口径） | 2,885 词 | 3,366 词 |
| 可溯源参考文献 | 6 条 | 6 条 |
| ICLR PDF | `paper/paper.pdf`（242,849 B） | `paper/paper.pdf`（247,305 B） |
| draft_sha256 | `99672d5720932fd1…` | `234f94470695d690…` |

## 9. 结论

1. **提交演示**：v4 首选——2 轮、9 次调用、217,320 token 原生收敛，全程零人工干预。
2. **v6 净增益**：首轮错误 −2 且规避两类格式错误（writer skill 小幅正向）；证据池 +7 条溯源（扩展生产可用）；暴露并闭环一个真实门禁 bug（附回归单测，工程收益超出运行本身）。
3. **v6 成本**：token 约 2.28×、调用 9→17（含 resume 重放，非纯净对照）。
4. **配置走向**：当前代码库默认采用 v6 的注入位置（writer-only + 扩展开启）+ 修复后的门禁豁免；门禁盲区修复后，预期同等配置的收敛轮次将回落至 v4 量级。

## 10. 证据复核路径

| 证据 | 路径 |
|---|---|
| v4 / v6 PDF | `runs/route_b2_deepseek_20261014_v{4,6}/paper/paper.pdf` |
| 运行清单（调用数/状态/哈希） | `runs/route_b2_deepseek_20261014_v{4,6}/run_manifest.json` |
| token / 检索 / 墙钟 | `runs/route_b2_deepseek_20261014_v{4,6}/resource_usage.json` |
| 各轮门禁轨迹 | `runs/route_b2_deepseek_20261014_v{4,6}/validation_report.json` |
| v6 重校验封存产物 | `runs/route_b2_deepseek_20261014_v6/revalidation_original_*` |
| 扩展审计事件 | `runs/route_b2_deepseek_20261014_v6/evidence/literature_events.jsonl` |
| Prompt 哈希审计 | `runs/route_b2_deepseek_20261014_v{4,6}/team_prompt_audit.jsonl` |
| 门禁修复与单测 | `examples/scitrace_research/workflow/paper_quality.py`、`tests/unit_tests/agents/harness/research/test_paper_quality.py` |
| 三轮完整消融叙事 | `examples/scitrace_research/ABLATION.md` |
