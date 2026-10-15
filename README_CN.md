# Scitrace-Swarm

**面向科研论文生成的多智能体联合调度开发**

本项目基于 [JiuwenSwarm](https://github.com/openJiuwen-ai/jiuwenswarm) 二次开发，新增 **Scitrace-Swarm** 科研工作流。它将文献检索、实验执行或可信结果导入、论点溯源、论文撰写、审稿修订、质量校验和 PDF 导出串联起来，用多智能体分工完成可审计的论文生成过程。

> JiuwenSwarm 的通用 Agent、桌面端和渠道功能来自上游项目。本仓库重点开发和验证 Scitrace-Swarm；原项目的安装及通用功能说明请参阅 [JiuwenSwarm 上游文档](https://github.com/openJiuwen-ai/jiuwenswarm#readme)。

[English](README.md) · [科研工作流详细说明](examples/scitrace_research/README.md) · [系统设计](examples/scitrace_research/docs/final_system_design.md) · [论文设计](examples/scitrace_research/docs/final_paper_design.md)

## 新增能力

| 环节 | Scitrace-Swarm 的实现 |
| --- | --- |
| 多智能体协作 | 研究负责人组织文献、实验、写作和审稿等阶段；阶段契约约束角色职责与交接内容。 |
| 文献与证据 | 检索并筛选文献，可选用 OpenAlex 一跳引文扩展；记录来源、扩展种子和调用审计。 |
| 实验与溯源 | 执行受限实验或导入已校验的结果，以证据图连接文献、论点和实验产物。 |
| 论文生成 | 根据结构化证据生成草稿，结合审稿意见和确定性质量门禁进行定向修订。 |
| Skill 辅助 | 提供投稿前检查、统计符号规范和引文追踪 Skill；写作阶段可按配置加载检查与符号规范。 |
| 结果交付 | 记录资源消耗、检查点及哈希绑定；通过校验后可导出 ICLR 格式论文 PDF。 |

科研结论、引用和实验结果仍需人工核验。

## 代码位置

- [`jiuwenswarm/agents/harness/research/`](jiuwenswarm/agents/harness/research/)：科研 Harness、Team 运行时、阶段契约、溯源和审计。
- [`examples/scitrace_research/`](examples/scitrace_research/)：工作流配置、实验示例、质量门禁和论文导出。
- [`examples/scitrace_research/skills/`](examples/scitrace_research/skills/)：Scitrace-Swarm 的本地 Skill 包。
- [`tests/unit_tests/agents/harness/research/`](tests/unit_tests/agents/harness/research/)：科研工作流测试。

## 开始使用

在仓库根目录创建 Python 3.11–3.13 环境并安装项目：

```bash
python -m pip install -e .
python -m pip install -r examples/scitrace_research/requirements-route-b2.txt
```

真实模型运行前，按 [科研工作流详细说明](examples/scitrace_research/README.md) 配置模型环境变量，并先检查运行条件：

```bash
bash examples/scitrace_research/launch_paper.sh --check
```

该示例默认导入仓库内经过哈希校验的 Route B2 实验结果，**不等同于重新运行检索实验**。完整复现边界与重新实验的方法见 [复现说明](examples/scitrace_research/REPRODUCIBLE_PAPER.md)。PDF 导出还需要本机安装 `pdflatex`。

运行科研工作流测试：

```bash
python -m pytest tests/unit_tests/agents/harness/research -q
```

## 来源与许可

本仓库保留 JiuwenSwarm 原有代码和历史。Scitrace-Swarm 是在其基础上新增的科研论文生成能力；上游项目名称、文档和贡献归属请以 [JiuwenSwarm](https://github.com/openJiuwen-ai/jiuwenswarm) 为准。代码许可见 [LICENSE](LICENSE)。
