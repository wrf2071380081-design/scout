# scout 简历弹药库

> 这份文档给你**即插即用**。每个要点都链接到仓库里真实存在的能力与消融报告，
> 面试官无论从哪里切进来，你都能自然地接上去。

---

## 一、项目主条目（简历正文）

**scout · 长文档 Agentic RAG 系统**　2026.08 – 至今
*Python · 混合检索 · cross-encoder 重排 · 模块化评测 · 可恢复 Agent 运行时*

面向长文档问答设计并实现的 Agentic RAG 系统。**全流程真实模型化**：
语义向量（BGE-zh）+ cross-encoder 神经重排（bge-reranker）+ Kimi K3 真实 LLM；
同时保留离线替身实现，让"流程"与"能力"分开可测——
这不是两个系统，是一个懂得什么时候该用真相说活的系统。

---

## 二、中文 Bullet Points（主版本）

**技术栈：** Python、混合检索、cross-encoder 重排、模块化评测、状态恢复设计、人工介入（HITL）

- **在公开冻结基准上验证（真实模型）**：于 **MuSiQue-Ans dev** 的 1000 段联合检索池、
  50 条冻结样本上，**真实 LLM 端到端管线 Recall@5 达 48.0%**（bootstrap 95% CI `[39.0, 57.0]`，
  n=50）；pairwise bootstrap vs 离线哈希版 27.3%，cross-encoder 神经重排是主要改善来源；
  **作答率 24%**——系统对证据不足的多跳问题**明确拒答而非编造**（门控在工作，不是缺陷）。

- **实现 cross-encoder 神经重排并买到了真实回报**（fastembed + ONNX，BAAI/bge-reranker-base，中文配套）：
  同一份 MuSiQue 样本上把词法重排换成 cross-encoder，**端到端作答率从 7% 提到 24%、
  Recall@5 从 26.7% 提到 48.0%** ——重排这一项就抬了近乎三倍的门槛；
  单候选对照 top-k 排名更替近一半、相关/不相关的分距明显拉开、CPU 延迟实测 ~5s/20 候选。
  粗排管快（ANN+RRF）、重排管准（cross-encoder）这条经典分工，**是在同一个语料、同一批样本上
  实打实测出来的，不是引用的**。

- **真模型替换是可以被验证的**：本地语义向量（fastembed/BGE-zh，512 维）与真实 LLM
  替换离线替身，benchmark 数字从"流程能跑"变成"真的能答"；
  **这条差异本身也被写成了报告**（离线启发是 F1 ≈ 0.5% 的能力上限，跟系统烂不烂无关）。

- 自建四层评测与模块化消融（含负贡献）：混合检索 **Recall@1 78.9%**
  （较纯稠密基线 +47.2pp，自家语料 19 条手选；200 条生成集为方向性参考）、
  拒答正确率 100%、误拒率 0%；
  **Auto-merge 的 REPLACE 在自家语料上反胜 EXPAND +6.8pp MRR**（主流说法是反的）。

- **把"拒答"设计成一个可测量的决策**，而不是一个遮蔽问题的开关：
  双维充分性判定（整体覆盖率 + 焦点词覆盖率），误拒率从 90% 归零；
  公开基准上它换来的数字是：**50 条里 12 条能答、38 条明确地拒绝，0 条编造**。

- **构建三层记忆与有效期机制**（工作 / 情景 / 语义），
  bi-temporal 时标区分"现在为真"与"曾经为真"，防止过期事实被当成当前事实回答。

- **实现 HITL 人工审批机制**：声明式风险分级（SAFE / CONFIRM / REQUIRED）、
  审批时可修改参数（如"金额从 100 万改为 10 万"）、
  副作用账本按血缘与调用槽位做幂等（断点恢复不重放副作用）、
  跨进程和跨小时后仍然可恢复的 checkpoint、审批超时默认 reject（fail-closed）。

- **用故障注入实验量化自愈编排在生产场景的行为**：注入 8 类可预期的故障
  （provider 超时/参数非法/死循环/checkpoint 损坏等），量出
  **类型化失败率 100%、预算内恢复率 100%、未捕获裸异常 0**；
  顺带找出并修复了一处 decide 阶段 ProviderError 直接穿透主调用栈的生产级 bug。

- **实现 MCP 服务端（零新增依赖）**：手写 stdio MCP Server（JSON-RPC 2.0），
  把检索与问答暴露为 `scout_search` / `scout_ask` / `scout_stats` 标准工具，
  工具描述中强制要求带 `chunk_id` 的结构化证据，任意 MCP 客户端可直接接入。

- **沙箱工具与危险操作审批**：实现 `python_exec`（子进程执行，模式拦截 + 超时 +
  输出截断 + 人工审批四层），并按工具声明（`side_effects=True`、无补偿动作）
  被 HITL 策略**自动判定为必须人审且无免审开关**。

- **实现 SSE 流式问答与 Web 控制台**：链路透视（每阶段真实的 trace）、
  审批台（风险分级 + 可编辑参数）、SSE 流式接口挂真实阶段钩子（不是伪造进度条）。

- **预算管控（预扣 + 硬上限 + 结算）**：`BudgetedLLM` 把"超支就停"打造成类型化决策，
  冻结额度与实际用量分开存，任务在预算耗尽的那一刻被截停而不是跑完才透支。

---

## 三、中文 Bullet Points（极简版，给空间紧张的位置）

- 自研长文档 Agentic RAG 框架，自建的评测框架与 200 条评测集上，
  **混合检索使 Recall@1 达 78.9%（+47.2pp vs 稠密基线）**、拒答正确率 100%。
  关键技术亮点：双维充分性门控实现可靠拒答；三层记忆带失效机制；审批链路在断点上可恢复。

- **Auto-merge 的效果在不同语料上结论相反**——用消融实验发现 REPLACE 在自建语料上反而比 EXPAND 高 +6.8pp MRR。
  同一条实现换个语料结论就变了，这件事本身比多写上三个图标更说明问题。

---

## 四、英文简历版（外企投递时）

**scout: An Agentic RAG framework for long-document QA** *2026 – Present*

- Built a fully offline-reproducible Agentic RAG system over 40 public documents
  (annual reports & policy guidelines) and a 200-item evaluation set sliced into 16 query types.
  Hybrid retrieval improves **Recall@1 to 78.9% (+47.2pp vs dense-only baseline)**;
  abstention accuracy is 100% with zero false refusal;
  every answer carries a traceable evidence span.

- Designed abstention calibration as a first-class decision:
  a two-dimensional sufficient-evidence gate (aggregate coverage + focus-concept coverage)
  turns a confident wrong answer into an explicit "I don't know," eliminating false refusals.
  Human-in-the-loop review sits on the same mechanism: risk-tiered approval,
  parameter editing at review time, and side-effect idempotency keyed by lineage
  so postponed decisions keep their guarantees across process restarts.

- Every module is **ablatable and independently measurable**.
  Ablation highlights (corpus-dependent): hybrid retrieval is the single largest lever
  (−41.9pp MRR when removed), while Auto-merge REPLACE outperforms EXPAND by +6.8pp MRR—
  two conclusions that happily falsify mainstream intuition when corpora differ.

---

## 五、面试大概率会问的问题（必须预习）

**不是背答案，是要能讲"为什么"。**

### Q1：你们项目里最关键的一个决定是什么？

> "把拒答设计成一个可测量、可消融的决策，而不是一个开关。"
> 因为拒答率恒为 0 的系统从来不是因为问的可答，而是因为从来不拒答。
> 我们用四层评测把"不该答却答"（幻觉）和"该答却拒"（误拒）拆开，
> 再加双维充分性门控——第一次用 IDF 加权的时候误拒率一度飙到 90%，
> 详尽定位到阈值尺度问题之后才校准回来。
> 这个调参的过程本身就是一份消融实验笔记，写在了 `docs/FINDINGS.md` 里。

### Q2：如果从头再来，你会先改什么？

> "先把 200 条自动生成的评测题人工筛一轮。"
> 自动生成的题有一部分"数据精度要求"比"能不能答出来"更苛刻，
> 这会把门控误杀率推高。现在没有集体人工标注，
> 样本量只有 200 题的生成草稿，所以所有数字只是方向性，不是统计结论。
> 我写完项目就第一个动手做这个。

### Q3：哪个模块的贡献最大？为什么？

> "混合检索是最大的杠杆：关掉它 MRR 掉 41.9pp。"
> 这说明在长文档、政策类语料里，关键术语的精确词法匹配仍然比通用语义向量可靠。
> 这也呼应了为什么 Auto-merge 的效果会因语料而异：两条数据合在一起才有说服力。

---

## 六、面试时的注意事项（大部分人忽略的结构）

- **不要说"这个模块我实现了"**。要说"这个决定在什么情况下不适用"。
  后者决定你是初级还是资深。
- **指标必须能现场兑现**：如果你说拒答正确率 100%，面试官追问"评测集多大、怎么标注的"，
  必须能立刻回答"19 条手选 + 200 条自动生成，其中 2 条明确应拒答"——
  含糊过去比不说还糟。
- **多问一个问题**：向面试官请教"如果换成你们的语料，这个结论还成立吗?"
  这问题资深、不给安排、刁钻、能直接把一整个项目聊活。
