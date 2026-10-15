# SciTrace-Swarm Skill 注入消融报告（v4 / v5 / v6）

三轮 strict-real 全流程对照运行，回答一个问题：**把 Skill 知识包内联到直连专家提示词的哪个位置有价值——writer 侧、review 侧，还是双侧？**

所有数字均与运行产物核对（`run_manifest.json`、`resource_usage.json`、`validation_report.json`、`literature_events.jsonl`、`team_prompt_audit.jsonl`），可按第 9 节指引复核。

## 0. 公共实验条件

| 项 | 值 |
|---|---|
| 模式 | `real_team` + `strict-real` + `no-mock-fallback`（无任何 fixture 兜底） |
| 模型 | DeepSeek `deepseek-v4-pro`，`temperature: 0.0` |
| Topic | Continuous difficulty calibration for budgeted hybrid scientific evidence retrieval |
| 一致性循环 | `max_rounds: 2`（v6 resume 放宽到 3） |
| 预算上限 | 12 次模型调用（v6 resume 放宽到 17） |
| 运行日期 | 2026-10-14 |

## 1. 设计矩阵与结果

| | v4（基线） | v5（双侧注入） | v6（writer-only + 扩展修复） |
|---|---|---|---|
| writer 侧 skill（draft/revision） | 无 | pre-submit-checklist + stats-notation | pre-submit-checklist + stats-notation |
| review 侧 skill | 无 | stats-notation | **无** |
| 引文扩展（OpenAlex 一跳） | 关 | 开（旧代码） | 开（三层隔离修复后） |
| 一致性轨迹（门禁错误数） | 16 → 2 → **0** ✓ | 16 → 3 → 2 ✗ | **14** → 3 → 2 → 2 ✗ → 门禁修复后离线重校验 **0** ✓ |
| 模型调用 | 9 | 10（含 1 次阶段重试） | 17（含 resume 重放） |
| 总 token | 217,320 | 256,979 | 494,300 |
| 墙钟时长 | 886 s | 1,041 s | 816 s + resume 263 s |
| 真实检索调用 | 10 | 10（扩展 0 次，全灭） | 10 + **13 次扩展** |
| 文献证据池 | 10 条 | 10 条 | **17 条**（+7 来自扩展） |
| 最终状态 | success，ICLR PDF 导出 | failed（consistency_loop） | success（门禁误报修复后离线重校验 + ICLR PDF 导出，0 次额外调用） |
| draft_sha256 | `99672d57…` | `960e8f41…` | `234f944c…`（round-3 终稿） |

## 2. 发现一：review 侧 skill 注入破坏修订循环收敛

v5 首次将 stats-notation 注入审稿人提示词（prompt 较 v4 增大 2.5KB，`team_prompt_audit.jsonl` 字节级确认），修订轨迹从 v4 的 16→2→0 恶化为 16→3→2。v5 round 2 的两个剩余错误中，一个是**审稿人自身引入的假问题**（把 SciFact budget 2 的方向写反），属于规则性文本诱发的主观 findings 膨胀。

v6 仅摘除 review 侧注入（writer 侧保持不变）后，该现象消失：v6 round 2 仅剩的 1 个 reviewer finding 是**自指性复述**（指向同一句被确定性门禁 `paper_pathway_allocation_claim` 命中的措辞），不构成独立新问题。失败原因回到 writer 的真实写作残留，而非审稿人噪声。

**归因成立：v5 的不收敛主因是 review 侧注入，而非 writer 侧。**

## 3. 发现二：writer 侧 skill 注入小幅改善首轮质量

v4 与 v5 的 round 0 错误数均为 16；v6（writer 侧注入 + 干净 review）首轮降至 **14**，为三次运行最佳首轮。resume 将 `max_rounds` 放宽到 3（预算 17）后，轨迹为 14 → 3 → 2 → 2：writer 在 round 3 已把审稿人建议的修正句**逐字**写入 draft，但门禁对另外 3 句科学正确的否定句继续误报——修订循环被**验证器缺陷**而非写作残留阻断（归因见发现四）。

## 4. 发现三：OpenAlex 悬空引用隔离（工程修复，生产验证通过）

**v5 故障**：seed 的 `related_works` 数组含悬空 ID `W2376932109`（OpenAlex 自己列出、自己 404）。旧代码对 404 盲目重试 3 次后将**单个**失败升级为**整个扩展放弃**（审计事件：`status: skipped, seed_count: 0`）。

**三层修复**（`citation_expansion.py`，单测 57/57 覆盖）：

1. 404 快失败——确定性失败不重试（1 次调用 vs 旧 3 次）
2. 悬空 related work 逐个跳过——单个死引用不拖累同 seed 其余扩展
3. 种子级隔离——单个死 seed 不拖累其余 seed

**v6 生产验证**：13 次真实 OpenAlex 调用、27 个扩展候选、**7 篇采纳且全部带 seed 溯源**（`kept_provenance`），文献证据池从 10 条增至 17 条。v4 曾因参考文献仅 4 条可溯源触发 `paper_references_insufficient` 门禁，该瓶颈被扩展直接缓解。

## 5. 发现四：确定性门禁的否定盲区（v6 失败根因，离线重校验闭环）

**故障形态**：v6 round 3 的确定性错误只剩 `paper_pathway_allocation_claim` 一个。writer 已按审稿人处方把修正句逐字写入（"it does not allocate budget among retrievers, retrieval pathways, or sub-queries"，该句被正确豁免），但同一段落里另外 3 句**科学上正确**的否定/对比句继续触发同一正则：

1. "The **allocat**or does not rerank documents, …, or allocate budget among retrievers, retrieval **pathways**…"——模式把名词 *allocator* 当动词匹配，否定词落在匹配区间内部；
2. "…redistributes a known total capacity **rather than** choosing a retrieval pathway…"——对比连词不在豁免词表；
3. "…allocation method, **not** a per-query … retrieval-pathway router"——否定词在区间内部。

**根因**：`paper_quality.py` 的 forbidden_patterns 豁免只扫匹配前 60 字符前缀，且词表缺少 rather than 等修正句式——与此前修复的 `method_contract.py` `_negated` 属同一 bug 类。v4 通过纯属措辞运气：其唯一命中句的否定词恰落在前缀窗口内。

**修复**（与 `_negated` 语义对齐）：豁免检查扩展为"前缀 60 字符 ∪ 匹配区间内、禁用术语之前"，词表补齐 rather than / instead of / as opposed to / unlike / distinct from；禁止短语**自身内含否定**的规则（如 `do not extend …`）不入术语表，保持可执行。新增单测覆盖 3 句误报豁免、阳性对照与内含否定规则不回归（374/374 全过，顺带修复 test_paper_export 转义语法错、RealResult 假对象缺 `raw_payload`、4 处 sqlite3 `with` 不关闭连接的 Windows 句柄泄漏）。

**闭环**：`revalidate_run.py`（仓库自带的"门禁实现 bug 修复后专用"离线通道）在**不改一个字节 draft、0 次模型调用**的前提下重放确定性门禁 → v6 round-3 终稿 0 错误 → manifest 翻转 success（`revalidated_from_failure: true`）→ ICLR PDF 导出。原始失败产物五件套（draft/manifest/quality/validation/research.db）以 `revalidation_original_*` 完整封存，审计事件 `offline_gate_revalidation` 记录全过程。

## 6. Prompt 体量影响（`team_prompt_audit.jsonl`，v5 vs v4 字节级对比）

| 阶段 | 注入后增量 |
|---|---|
| draft | +4.4 KB |
| review | +2.5 KB |
| revision | +5.7 KB |

v6 的 review 提示词与 v4 字节级一致（opt-in 注入：无 skill 契约即空块，`skill_context.py` 保证）。总 token 对比显示双侧注入（v5）比基线多耗 ~18%，其中相当部分花在未产生收敛收益的 review 侧。

## 7. 结论与当前默认配置

1. **提交演示**：v4 运行（无 skill、门禁 16→2→0、ICLR PDF 成功导出），见 `runs/route_b2_deepseek_20261014_v4/paper/paper.pdf`；v6 round-3 终稿经门禁修复后离线重校验同样导出 PDF，见 `runs/route_b2_deepseek_20261014_v6/paper/paper.pdf`
2. **消融价值**：v5/v6 构成完整消融证据链——review 侧注入有害（发现一）、writer 侧注入有限正向（发现二）、扩展修复生产可用（发现三）、门禁否定盲区暴露并闭环（发现四）
3. **固化默认**：v6 配置已作为代码库新默认——review 契约 skill 置空（`team_contract.py` 内附消融依据注释）、引文扩展开启；确定性门禁的否定豁免语义与 `method_contract._negated` 对齐（`paper_quality.py`，含否定豁免回归单测）。v6 终稿经离线重校验 0 错误

## 8. 三轮故事线（评审叙事）

> 基线（v4）先证明系统在 strict-real 下端到端可靠；随后引入 skill 机制做双侧消融（v5），发现 review 侧注入令修订循环不收敛——这是一个有价值的负结果；据此做单变量修正（v6），确认归因、拿到 writer 侧首轮改善，并顺带暴露-修复了 OpenAlex 悬空引用的真实 API 缺陷（三层隔离 + 57 项单测 + 生产验证 7 篇溯源扩展）。
>
> 尾声（v6 闭环）：writer 按审稿处方写出的否定/对比句反过来暴露了确定性门禁自身的否定盲区（发现四）——与 `method_contract` 属同一 bug 类。修复豁免语义、补齐单测后，经仓库自带的离线重校验通道在 **0 次模型调用**下重放门禁，v6 翻转 success 并导出 PDF；全程不改 draft 一个字节，原始失败产物五件套封存可查。

## 9. 证据复核指引

| 证据 | 路径 |
|---|---|
| v4 最终 PDF | `runs/route_b2_deepseek_20261014_v4/paper/paper.pdf` |
| v6 重校验 PDF | `runs/route_b2_deepseek_20261014_v6/paper/paper.pdf` |
| 重校验翻转字段 | `runs/route_b2_deepseek_20261014_v6/run_manifest.json`（`revalidated_from_failure: true`、`model_calls_during_revalidation: 0`） |
| 重校验前封存产物 | `runs/route_b2_deepseek_20261014_v6/revalidation_original_*`（manifest/draft/paper_quality/validation/research.db 五件套） |
| 各轮门禁轨迹 | `runs/route_b2_deepseek_20261014_v{4,5,6}/validation_report.json` |
| 模型/检索/审计计数 | `runs/route_b2_deepseek_20261014_v{4,5,6}/run_manifest.json`、`resource_usage.json` |
| 扩展审计事件 | `runs/route_b2_deepseek_20261014_v{5,6}/evidence/literature_events.jsonl` |
| Prompt 哈希审计 | `runs/route_b2_deepseek_20261014_v{4,5,6}/team_prompt_audit.jsonl` |
| 扩展修复代码 | `jiuwenswarm/agents/harness/research/citation_expansion.py` |
| 门禁否定豁免修复 | `examples/scitrace_research/workflow/paper_quality.py`（对齐 `method_contract._negated`） |
| 契约默认与依据 | `jiuwenswarm/agents/harness/research/team_contract.py`（review 契约注释） |
| Skill 包 | `examples/scitrace_research/skills/{pre-submit-checklist,stats-notation,citation-graph-tracer}` |
| 单测 | `tests/unit_tests/agents/harness/research/test_citation_expansion.py`（57 项扩展隔离）、`test_paper_quality.py`（否定豁免回归） |
