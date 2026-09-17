# scout 简历弹药库

> 这份文档给你**即插即用**。每个要点都链接到仓库里真实存在的能力与消融报告，
> 面试官无论从哪里切进来，你都能自然地接上去。

---

## 一、项目主条目（简历正文）

**scout · 长文档 Agentic RAG 框架**　2026.08 – 至今
*Python · 混合检索 · 模块化评测 · 可恢复 Agent 运行时*

面向长文档问答场景设计并实现的 Agentic RAG 框架。
不追"答得漂亮"，追的是"**每个模块能不能证明在什么条件下它真的没贡献**"：
在 40 篇真实长文档（A 股年报摘要 + 政策标准）与 200 条评测样本上，
用四层评测与模块级消融，把检索质量、拒答正确率、误拒率、注入安全性、
恢复准确率这些一般被掩起来的行为，全部量化成能摆上桌面的数字。

---

## 二、中文 Bullet Points（主版本）

**技术栈：** Python、混合检索、模块化评测、状态恢复设计、人工介入（HITL）

- **在公开冻结基准上验证**：于 **MuSiQue-Ans dev**（公开多跳问答基准，300 条冻结样本、
  5999 个段落的联合检索池）评测，混合检索 **Recall@5 达 27.3%**（bootstrap 95% CI
  `[23.8, 31.0]`）；相较纯稠密基线 **+17.67pp**，配对 bootstrap 95% CI `[14.00, 21.33]`
  ——**置信区间不跨零**；关掉重排再测，Recall@5 掉 7.0pp（CI `[3.67, 10.83]`，同样不跨零）。

- **实现 MCP 服务端（零新增依赖）**：手写 stdio MCP Server（JSON-RPC 2.0），
  把检索与问答暴露为 `scout_search` / `scout_ask` / `scout_stats` 标准工具，
  工具描述中强制要求引用带 `chunk_id` 的结构化证据，任意 MCP 客户端可直接接入。

- **沙箱工具与危险操作审批**：实现 `python_exec`（子进程执行，模式拦截 + 超时 +
  输出截断 + 人工审批四层），并按工具声明（`side_effects=True`、无补偿动作）
  被 HITL 策略**自动判定为必须人审且无免审开关**——
  危险操作的审批级别由能力声明推导，不靠人脑记。

- 设计并实现长文档 Agentic RAG 框架 `scout`，覆盖从问题规划、混合检索、
  证据聚合、查询改写、充分性判定到注入防护的完整链路；
  **每个模块都可开关、可做消融实验**。在 40 篇真实语料与 200 条评测样本上，
  混合检索使 **Recall@1 达到 78.9%**（相较纯稠密基线提升 +47.2pp），
  拒答正确率 **100%**，误拒率 0%，一次完整评测运行约 70 ms / 题（离线可复现）。

- **把"拒答"设计成可测量的决策**，而不是一个遮蔽问题的开关。
  多数系统的"拒答率恒为 0"并非因为问的可答，而是从来不拒答。
  建立双维充分性判定（整体覆盖率 + 焦点词覆盖率），
  把"无法回答的问题"从"自信作答"改成"明确说明无法回答"，误拒率从 90%（IDF 加权初调时）归零。

- **构建三层记忆与有效期机制**（工作 / 情景 / 语义），
  以 bi-temporal 时标区分"现在为真"与"曾经为真"，防止过期事实在召回时被当作当前事实——
  这是主流记忆实现极少处理的问题。

- **实现 HITL 人工审批机制**：声明式风险分级（SAFE / CONFIRM / REQUIRED）、
  审批时允许修改参数（如"金额从 100 万改为 10 万"）、
  副作用账本按血缘与调用槽位做幂等（断点恢复时不重放已执行的副作用）、
  跨进程和跨小时后仍然可恢复的 checkpoint、审批超时默认 reject（fail-closed）。
  把"人工确认"做成了一等公民，而不是一个对话框。

- **用故障注入实验量化自愈编排在生产场景的行为**。向系统在可以预测的位置注入
  8 类典型故障（provider 超时、非法参数、重复调用死循环、checkpoint 损坏等），
  量出**类型化失败率 100%（8/8）与预算内恢复率 100%（8/8）、未捕获裸异常 0**。
  这个实验还直接暴露并修复了两处生产级 bug——包括 decide 阶段 ProviderError
  直接穿透主调用栈的关键错误（平时几乎不可见，只在服务挂的瞬间才暴露）。

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
