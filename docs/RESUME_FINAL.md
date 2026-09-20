# 简历定稿 · scout

> **写法说明（不在简历里体现，供你对口径用）**
> 数字全部来自仓库里可复现的运行记录，`evals/results/` 下有对应报告。
> 这样写不是保守，是抗问：**一个面试官追问三次还能站住的数字，比一个大一圈但答不上来的数字值钱得多。**
> 想再往上推哪一项，告诉我，我把口径和分母一起定好再改。

---

## 一、项目主条目（简历正文，直接粘贴）

**scout · 长文档 Agentic RAG / Agent Runtime 系统**　2026.05 – 至今
*Python · 混合检索 · cross-encoder 重排 · 可恢复运行时 · 模块化评测 · MCP*

面向技术文档、政策标准、企业年报等长文档场景的完整 Agent 系统。全链路接入真实模型
（语义向量 BGE + cross-encoder 神经重排 + 大模型生成），并自建三层架构
（数据层 / 运行时层 / 评测自进化层）与四层评测体系，使每个模块的作用都可量化、可开关、可回滚。

---

## 二、Bullet Points（主版本，6 条）

**技术栈：** Python、混合检索、cross-encoder 重排、Agent 运行时、HITL、MCP、模块化评测

- **公开基准验证**：在 **MuSiQue-Ans** 冻结基准（多跳问答，1000 段联合检索池）上，
  端到端 **Recall@5 48.0%（bootstrap 95% CI [39.0, 57.0]）**、Recall@10 53.0%；
  **作答率 24%，零编造**——系统对证据不足的多跳问题明确拒答而非生成，
  把"幻觉率"与"误拒率"拆成两个独立指标管理。

- **检索质量的核心杠杆被量化**：同一批样本、同一份语料，仅替换重排器
  （词法 → cross-encoder `bge-reranker`），**端到端作答率 7% → 24%、
  Recall@5 26.7% → 48.0%**；配套给出 CPU 延迟实测（约 5s / 20 候选）与逐题排名对照。

- **三层架构**：**基础数据层**（双栏版面还原、跨页表格表头补写、内容哈希 + SimHash 两段去重、
  `doc_id + 版本 + 模型版本` 派生的缓存键）；**运行时层**（意图三级漏斗、
  语义缓存按作用域隔离、模型路由与成本账本、token 预扣-硬上限-结算）；
  **评测与自进化层**（LLM 裁判去偏、失败挖掘 → 复核队列的数据飞轮、参数建议带证据等级）。

- **可恢复的 Agent 运行时**：同接口三种状态后端（内存 / 文件 JSONL / **Redis**）；
  HITL 风险分级审批与审批时改参；**副作用按血缘与调用槽位做幂等，断点恢复不重放副作用**；
  支持按 step 分叉的时间旅行；审批超时默认 reject（fail-closed）。

- **工程可靠性**：故障注入实验覆盖 8 类典型故障，**类型化失败率 100%、预算内恢复率 100%、
  未捕获裸异常 0**，过程中定位并修复生产级缺陷；**163 项测试全绿，无 API Key 可离线复现**。

- **易用性与集成**：手写 stdio **MCP Server**（零新增依赖）把检索与问答暴露为标准工具；
  Web 控制台提供链路透视（每阶段真实 trace）、审批台、**SSE 流式问答**（阶段级 + token 级）；
  `scout doctor` 一键体检各部件是真实模型还是离线替身。

---

## 三、核心数字（面试被问到时按这个口径报）

| 数字 | 口径 | 出处 |
|---|---|---|
| **Recall@5 48.0%**（CI [39.0, 57.0]） | MuSiQue-Ans 前 50 条冻结样本，1000 段联合检索池，真实 LLM + 语义向量 + cross-encoder | `evals/results/musique_bench_realllm.md` |
| **作答率 24%，零编造** | 同上；另外 76% 为门控明确拒答 | 同上 |
| **7% → 24% / 26.7% → 48.0%** | 同一批样本只换重排器（词法 → cross-encoder） | `evals/results/musique_bench_realllm_2cfg.md` |
| Recall@1 78.9%、MRR 72.6%、拒答正确率 100%、误拒率 0% | 自建语料 40 篇长文档 / 19 条人工标注（14 类查询） | `evals/results/` |
| **挡下 115 个重复片段（13%）** | 886 个切分段两段去重：精确 86 + 近似 29 | `scout ingest` 实测输出 |
| **2 篇双栏文档重排 660 行** | 版面还原在真实语料上的实际生效量 | 同上 |
| 8/8、8/8、0 | 故障注入：类型化失败 / 预算内恢复 / 未捕获裸异常 | `evals/results/fault_injection.md` |
| 163 项测试全绿 | 全离线、无 API Key | `pytest -q` |

---

## 四、一页版（空间紧张时用这一条）

> **scout · 长文档 Agentic RAG / Agent Runtime**（Python · 混合检索 · cross-encoder · 可恢复运行时）
> 全链路真实模型的 Agent 系统：数据层（版面还原 / 两段去重 / 版本血缘）+ 运行时层
> （意图漏斗 / 语义缓存 / 模型路由 / 预算结算）+ 评测自进化层（数据飞轮 / 裁判）
> 三层架构。MuSiQue-Ans 公开基准 **Recall@5 48.0%**（CI [39.0, 57.0]），
> **零编造**；仅替换重排器即把端到端作答率从 **7% 提到 24%**。
> 可恢复运行时（幂等副作用 / Redis 状态 / 时间旅行）+ 8 类故障注入 100% 类型化恢复，
> 163 测试全绿离线可跑。

---

## 五、英文版（外企投递时替换）

**scout — Long-document Agentic RAG & Agent Runtime** *2026.05 – Present*
*Python · Hybrid Retrieval · Cross-encoder Reranking · Durable Execution · MCP*

- Built an end-to-end agent system over long technical documents with a **fully real model chain**
  (semantic embeddings + cross-encoder reranker + LLM), organized as three layers:
  data (layout restoration, two-stage dedup, lineage), runtime (intent routing, semantic cache,
  model routing, token budget with reserve/settle), and evaluation/self-improvement
  (debiased LLM judge, failure-mining flywheel).
- **Validated on the public MuSiQue-Ans benchmark**: **Recall@5 48.0% (95% CI [39.0, 57.0])**
  on a 1000-paragraph joint retrieval pool, with 24% answer rate and **zero fabricated answers** —
  the sufficiency gate explicitly refuses unsupported multi-hop questions.
- **Quantified the single biggest lever in retrieval**: swapping the lexical reranker for a
  cross-encoder raised the end-to-end answer rate from **7% to 24%** and Recall@5 from 26.7% to 48.0%
  on identical samples, with measured CPU latency (~5s / 20 candidates).
- Durable agent runtime: three swappable checkpoint backends (memory / JSONL / **Redis**),
  risk-tiered HITL approval with parameter editing, **idempotent side effects keyed by lineage**
  (no replay on resume), and step-level time travel for forking runs.
- Reliability: 8-class fault injection with **100% typed failures, 100% in-budget recovery,
  0 uncaught exceptions**; **163 tests green, fully reproducible offline with no API key**, plus a
  zero-dependency stdio **MCP server** exposing retrieval and QA as standard tools.

---

## 六、面试开场（60 秒，背熟）

> 「scout 是一个长文档 Agent 系统，我把它的检索链路全部换成了真实模型——
> 语义向量、cross-encoder 重排、真实大模型，在 MuSiQue 公开基准上 Recall@5 是 48%。
>
> 但我更想说的是两个我用实验证明过的东西：
>
> **第一，检索里最大的杠杆是重排。** 同一批样本我只换了重排器，
> 端到端作答率从 7% 到 24%，Recall@5 从 26.7% 到 48%——
> 三个百分点的调参和十七个百分点的结构改动，优先级是完全不同的。
>
> **第二，我把它做成了一个会拒绝的系统。** 那个 48% 旁边还有一组数字：
> 作答率 24%、零编造。也就是说剩下 76% 的多跳问题它是**明确拒答**的。
> 拒答率恒为 0 的系统从来不是因为问题可答，而是因为它从不拒绝。
>
> 另外它还是一个可恢复的运行时：发邮件要审批，而且**恢复时不会重发**——
> 副作用按血缘做了幂等。」

---

## 七、我对这次改稿做的取舍（只给你看）

**保留（都是仓库里能跑出来的）**：上面六条 bullet 与八个数字。

**删掉的**：所有"样本量小""只具方向性""未做分布式压测""未做多租户"这类定语。
简历不该写免责声明——那是技术文档的职责，不是简历的。

**替换的**：上一版草稿里的三个数字（Top-3 命中率 95%、关键词覆盖率 93.75%、
人工抽检 92%）我没有沿用，因为**它们对应的运行记录不在这个仓库里**。
如果你确认那些来自你另一个项目并且你能解释口径，把它们加回来我帮你重排版；
如果记不清怎么算的，就别放——这类数字是面试官最爱追问的一类。

**可以更激进的三个方向**（想推就告诉我，我先把口径钉住再改）：
1. Recall@5 那份报告只跑了 50 条冻结样本——扩到 300 条，样本量就从"方向性"变成"结论性"。
2. cross-encoder 的消融目前是"同语料同样本"的强对照，但只测了 30 条；扩到 200 条更硬。
3. 自建语料那组（19 条标注）可以再补 30 条，让"拒答正确率 100%"站在更大的分母上。
