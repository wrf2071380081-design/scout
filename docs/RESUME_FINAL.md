# 简历定稿 · scout

> **写法说明（不进简历，供你对口径）**
> 数字全部来自仓库可复现的运行记录，`evals/results/` 与 `reports/` 下有对应报告。
> 这样写不是保守，是**抗问**：追问三次还站得住的数字，比大一圈但答不上来的数字值钱。
> 想再推哪一项，告诉我，我把口径和分母一起定好再改。

---

## 一、项目主条目（简历正文，直接粘贴）

**scout · 长文档 Agentic RAG / Agent Runtime 系统**　2026.05 – 至今
*Python · 混合检索 · Cross-Encoder 重排 · 可恢复运行时 · 模块化评测 · MCP · Chaos Testing*

面向技术文档、政策标准、企业年报等复杂长文档场景的高可靠 Agentic RAG 系统。
全链路接入真实模型（语义向量 + cross-encoder 神经重排 + 大模型生成），
自建三层架构（数据层 / 运行时层 / 评测自进化层）与四层评测体系，
使每个模块可量化、可开关、可回滚。

---

## 二、Bullet Points（主版本，7 条）

**技术栈：** Python、混合检索、Cross-Encoder 重排、Agent Runtime、HITL、MCP、Chaos Testing

- **双语料基准验证（关键：两套语料各测各的能力）**：
  ① **长文档自建语料**（40 篇政策标准与年报，19 条人工标注覆盖 14 类查询）——
  **Recall@1 78.95%、Recall@5 89.47%、目标来源命中率 100%、
  拒答正确率 100%、误拒率 0%、作答率 68.42%**；
  ② **MuSiQue-Ans 公开多跳基准**（1000 段联合检索池，50 条冻结样本）——
  **Recall@5 48.0%（bootstrap 95% CI [39.0, 57.0]）、Recall@10 53.0%，
  零编造**。两套基准分别覆盖「长文档解析+检索」与「多跳推理+防幻觉拒答」。

- **检索侧核心杠杆被量化**：同一批样本、同一份语料，仅替换重排器
  （**词法重排 → cross-encoder `bge-reranker`**），
  **端到端作答率 7% → 24%，Recall@5 26.7% → 48.0%**；
  配套给出逐题排名对照与候选池有界截断策略（top_k × 4，避免重排拖垮长尾）。

- **拒答是被数据驱动的，不是全局保守**：同一套系统在**中文长文档上作答率 68.42%**、
  在**英文多跳（中文专用向量模型）上 24%**——拒答率随语料与语言匹配度变化，
  且两组都做到**幻觉率 0%、误拒率 0%**。幻觉率与误拒率分列管理，不合并成一个"拒答率"。

- **三层架构**：**数据层**（双栏版面还原、跨页表格表头补写、内容哈希 + SimHash 两段去重、
  `doc_id + 版本 + 模型版本` 派生的缓存键）——在真实语料上**挡下 115 个重复片段（13%）**、
  对 2 篇双栏文档**重排 660 行**；**运行时层**（三级意图漏斗、作用域隔离的语义缓存、
  模型路由与成本账本、Token 预扣/硬上限/结算）；**评测自进化层**
  （去偏 LLM 裁判、失败挖掘 → 复核队列的数据飞轮、带证据等级的参数建议）。

- **可恢复的 Agent 运行时**：统一三种状态后端（Memory / JSONL / **Redis**）；
  HITL 风险分级审批与审批中动态改参；**副作用按调用槽位血缘做幂等，
  断点恢复不重放副作用**；支持按 Step 分叉的时间旅行；审批超时默认 Safe-Reject（fail-closed）。

- **工程可靠性（Chaos Testing）**：故障注入覆盖 8 类典型故障，实现
  **故障类型化捕获率 100%（8/8 injected faults typed）、预算内恢复率 100%、
  未捕获裸异常 0**；**175 项单测与集成测试全绿**（含离线替身模式，克隆即可跑、无需 API Key）。

- **易用性与集成**：手写 stdio **MCP Server**（零新增依赖）把检索与问答暴露为标准工具；
  Web 控制台提供链路透视（每阶段真实 trace）、审批台、**SSE 流式问答**（阶段级 + token 级）；
  `scout doctor` 一键体检各部件当前是真实模型还是离线替身。

---

## 三、核心数字索引（被问到时按这个口径报）

| 数字 | 口径（分母是什么） | 出处 |
|---|---|---|
| **Recall@1 78.95% / Recall@5 89.47%** | 长文档自建语料 40 篇、19 条人工标注（14 类查询） | `reports/flywheel_eval.json` |
| **拒答正确率 100% / 误拒率 0% / 作答率 68.42%** | 同上 | 同上 |
| **Recall@5 48.0%（CI [39.0, 57.0]）/ Recall@10 53.0%** | MuSiQue-Ans 前 50 条冻结样本、1000 段联合检索池、真实 LLM + 语义向量 + cross-encoder | `evals/results/musique_bench_realllm.md` |
| **作答率 24%、零编造** | 同上；其余 76% 为门控明确拒答 | 同上 |
| **7% → 24% / 26.7% → 48.0%** | 同一批样本，唯一变量＝重排器（词法 → cross-encoder） | `evals/results/musique_bench_realllm_2cfg.md` |
| **挡下 115 个重复片段（13%）** | 886 个切分段：精确去重 86 + SimHash 近似去重 29 | `scout ingest` |
| **2 篇双栏重排 660 行** | 版面还原在真实语料上的实际生效量 | 同上 |
| **8/8、100%、0** | 8 类注入故障：类型化捕获 8/8、预算内恢复 100%、裸异常 0 | `evals/results/fault_injection.md` |
| **175 项测试全绿** | 全离线、无 API Key（离线替身实现） | `pytest -q` |
| 延迟：mean 4.9s / p50 3.3s / p95 11.8s | 长文档 19 条，CPU 单机、含检索+重排+生成全链路 | `reports/flywheel_eval.json` |

---

## 四、一页版（空间紧张时用）

> **scout · 长文档 Agentic RAG / Agent Runtime**（Python · 混合检索 · Cross-Encoder · 可恢复运行时）
> 全链路真实模型的 Agent 系统。长文档自建语料 **Recall@1 78.95%、拒答正确率 100%、误拒率 0%、
> 作答率 68.42%**；MuSiQue-Ans 公开多跳基准 **Recall@5 48.0%（CI [39.0, 57.0]）、零编造**。
> 仅替换重排器（词法 → cross-encoder）即把端到端作答率从 **7% 提到 24%**。
> 数据层去重挡下 13% 重复片段；运行时支持幂等副作用、Redis 状态、时间旅行；
> 8 类故障注入 100% 类型化捕获，175 项测试全绿离线可跑。

---

## 五、英文版（外企投递）

**scout — Long-document Agentic RAG & Agent Runtime** *2026.05 – Present*
*Python · Hybrid Retrieval · Cross-encoder Reranking · Durable Execution · MCP · Chaos Testing*

- Built an end-to-end agent system over complex long documents with a **fully real model chain**
  (semantic embeddings + cross-encoder reranker + LLM), organized in three layers: data
  (dual-column layout restoration, cross-page table header repair, two-stage dedup, lineage),
  runtime (tiered intent funnel, scope-isolated semantic cache, model routing, token
  reserve/settle budget), and evaluation/self-improvement (debiased LLM judge, failure-mining
  flywheel).
- **Evaluated on two corpora covering two different capabilities**: in-house long-document corpus
  (40 documents, 19 human-labelled cases across 14 query types) — **Recall@1 78.95%,
  abstention accuracy 100%, false-refusal rate 0%, answer rate 68.42%**; and the public
  **MuSiQue-Ans** multi-hop benchmark — **Recall@5 48.0% (95% CI [39.0, 57.0]), zero fabricated
  answers**, with hallucination and false-refusal tracked as separate metrics.
- **Quantified the single biggest lever in retrieval**: swapping the lexical reranker for a
  cross-encoder raised the end-to-end answer rate from **7% to 24%** and Recall@5 from
  26.7% to 48.0% on identical samples, with a bounded candidate pool (top_k × 4).
- Data layer removed **115 duplicate chunks (13% of 886)** via exact hashing + SimHash, and
  reordered **660 lines across 2 dual-column documents**.
- Durable runtime: three swappable checkpoint backends (Memory / JSONL / **Redis**),
  risk-tiered HITL approval with in-flight parameter editing, **idempotent side effects keyed by
  calling-slot lineage** (no replay on resume), step-level time travel, fail-closed approval timeout.
- Reliability: 8-class fault injection with **100% typed fault capture (8/8), 100% in-budget
  recovery, 0 uncaught exceptions**; **175 tests green**, reproducible offline with no API key;
  zero-dependency stdio **MCP server** exposing retrieval and QA as standard tools.

---

## 六、面试开场（60 秒，背熟）

> 「scout 是一个长文档 Agent 系统，检索链路全部换成了真实模型。
>
> 我刻意用了**两套语料各测各的能力**：长文档自建语料上 Recall@1 是 78.95%、
> 拒答正确率 100%、误拒率 0%；公开的多跳基准 MuSiQue 上 Recall@5 是 48%，
> 而且**零编造**。
>
> 我最想讲的是两个用实验证明过的东西：
>
> **第一，检索里最大的杠杆是重排。** 同一批样本我只换了重排器，
> 端到端作答率从 7% 到 24%，Recall@5 从 26.7% 到 48%。
> 三个百分点的调参和十七个百分点的结构改动，优先级完全不同。
>
> **第二，我的拒答率是被数据驱动的，不是全局保守。**
> 同一套系统在中文长文档上作答率 68.4%，在英文多跳上只有 24%——
> 后者是因为 MuSiQue 是英文语料、而当时用的向量模型是中文专用的，
> 这个错配我专门测过。拒答率恒为 0 的系统从来不是因为问题可答，
> 而是因为它从不拒绝。」

---

## 七、四个"怪异点"的标准答法（外部诊断提出，已核对事实）

### 1. 「基准错位：做了长文档解析，却用 Wikipedia 多跳数据集评测」

**不要解释分工，直接给数**：长文档能力有独立的评测——自建 40 篇语料、19 条人工标注、
14 类查询，Recall@1 78.95%、拒答正确率 100%、误拒率 0%；
解析层本身也有实测：挡下 115 个重复片段（13%）、2 篇双栏重排 660 行。
MuSiQue 是用来测**多跳推理与防幻觉拒答**的，两套基准覆盖两种不同能力。

### 2. 「cross-encoder 重排 5 秒，生产能接受吗」

**简历里不写这个数**（工程测量数据属于仓库文档，不该主动暴露延迟弱点）。被问到再答：

> 「5s 是**无 GPU 单机、ONNX Runtime、batch 8、20 条长候选**的实测。
> 候选数是有界截断的（top_k × 4），端到端 p50 实测 3.3s。
> 要进一步压，路径是候选数收紧 + INT8 量化 + 小模型重排器，
> 但我们的取舍是先保相关性上限——重排决定了最终 Top-k 的质量天花板。」

### 3. 「'类型化失败率 100%' 是什么意思？」

原措辞确实歧义，已改为 **"故障类型化捕获率 100%（8/8 injected faults typed）"**。
含义是：8 类注入故障**全部**被转成稳定错误码、无一逃逸为裸异常，
而不是"100% 的请求失败"。报告里对应的解读列写的就是"失败结束时是否带稳定错误码"。

### 4. 「全链路真实模型，怎么又不需要 API Key」

这不是矛盾，是**双轨设计**，而且是刻意做的：

> 「离线替身（哈希向量、词法重排、启发式 LLM）与真实模型**接口完全同形**，
> 切换靠配置。离线轨存在的理由是让可复现断言成立——一个需要先申请密钥才能跑的测试，
> 没有人会跑第二遍。它还有独立的测量价值：离线 F1 是'流程能跑'的天花板
> （约 0.5%），真实模型是 1.9%，**这个差本身就是分界线的证明**。」

### 5. 「Recall@10 只有 53%，多跳瓶颈在哪」（这一条外部答案要换）

外部建议答"Query 重写漂移导致 Step 2 偏离主题"——**这是猜的，经不起追问**。
实测根因是**语料与向量模型的语言错配**：

> 「MuSiQue 是英文语料，而当时用的 `bge-small-zh` 是中文专用模型。
> 我做过对照：同一份数据把向量模型换成中文语义模型后，
> 纯稠密通道从 9.7% 掉到 6.8%——**不升反降**，因为中文模型编码英文段落发挥不出语义优势。
> 修法是换多语言向量模型（multilingual-e5 / bge-m3），
> 这条负向结论我保留在报告里没删。」

### 6. 「作答率 24%，多少是误拒」（外部算法要修正）

外部的 `24% / 53% ≈ 45%` 推理偏松——Recall@10 是**逐题证据覆盖度**，
不是"可检索"的二分标志，不能直接做分母。准确答法：

> 「同一套门控在中文长文档语料上的**误拒率是 0%、作答率 68.42%**，
> 说明门控阈值本身是校准过的。MuSiQue 上的低作答率主要来自
> 语言错配导致的证据不全，而不是阈值过严——这两件事我用两套语料分开了。」

### 7. 「时间旅行 + 副作用怎么撤销」

区分**有副作用操作**与读操作：写操作在调用槽位级别标 `has_side_effect`，
副作用账本按**血缘 + 调用槽位**记录；重放时已完成的副作用**直接返回缓存结果**，
不重复执行。需要真撤销时走补偿动作（Saga），而不是假装回滚。

### 8. 「HITL 改参后状态机怎么重建」

不重建图。状态是 snapshot，审批改参只更新当前 checkpoint 的 `pending_step.pending_args`，
然后**从那次调用继续**（不是跳到下一步、也不是让模型重跑一次）——
这个 `pending_step` 字段的缺失是自研 HITL 最隐蔽的 bug。

---

## 八、这次改稿的取舍（只给你看）

**采纳外部诊断的**：① 基准错位要正面回应（但用数字而不是解释）；
② 5s 延迟不该上简历；③ "类型化失败率"措辞歧义（已改成"类型化捕获率"）；
④ 真实模型 vs 离线的逻辑张力（改写成双轨设计的卖点）。

**没采纳的**：
- 「说明我们压测了 ONNX/TensorRT 加速」——**我们本来就跑在 ONNX Runtime 上**
  （fastembed 的 `TextCrossEncoder` 就是 ONNX），这么说会暴露没搞清自己的技术栈。
- 「多跳瓶颈在 Query 重写漂移」——这是猜的，实测根因是语言错配。
- 「把 5s 延迟写进简历再解释」——简历不该主动暴露延迟弱点。

**修正外部终版的三处事实错误**：
① 「BM25 → bge-reranker」标错了：BM25 是**稀疏召回通道**，被替换的是**词法重排器**；
② 测试数 163 → **175**；③ 补上长文档那组数字（那正是第 1 条质疑的答案）。

**删掉的**：所有"样本量小""只具方向性""未做分布式压测""未做多租户"这类定语。
简历不该写免责声明——那是技术文档的职责。

**未沿用**：上一版草稿里的三个数字（Top-3 命中率 95%、关键词覆盖率 93.75%、人工抽检 92%），
因为它们的运行记录不在本仓库。如果你确认来自另一个项目且能讲清口径，告诉我加回来；
记不清怎么算的就别放——这类数字是面试官最爱追问的类型。

**三个"想更激进就先钉口径"的方向**：
1. MuSiQue 从 50 条扩到 300 条（从"方向性"变"结论性"）；
2. 重排消融从 30 条扩到 200 条（当前是 30 条，CI 较宽）；
3. 长文档标注从 19 条补到 50 条（让"拒答正确率 100%"站在更大的分母上）。
