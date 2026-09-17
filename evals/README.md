# 评测资产

## 文件

| 路径 | 说明 |
|---|---|
| `longdoc_v1.json` | 标注集起始版本，19 条样本，覆盖 14/16 类标签 |
| `datasets/longdoc-gold/` | 40 篇公开长文档的人工校验文本（20 篇 A 股年报摘要 + 20 篇政策/标准） |
| `reports/` | 评测运行产出的 JSON 与 Markdown 报告（运行产物，不入库） |
| `results/` | **人工挑选后入库**的对照结果，用于固定"改前 / 改后"的证据 |

## 数据集契约

每条样本的字段：

```jsonc
{
  "case_id": "cloud_industry_parts",     // 唯一标识
  "question": "根据知识库回答：…",         // 查询原文
  "tags": ["single_fact"],                // 查询标签，见 src/scout/evaluation/taxonomy.py
  "expected_sources": ["文件名.md"],       // 期望命中的来源文件
  "expected_keywords": ["硬件", "软件"],   // 期望出现在答案里的关键词
  "gold_snippets": ["摘自原文的短片段"],    // 金标准：命中块需包含任一片段
  "forbidden_keywords": [],               // 出现即判定失败（注入样本必填）
  "allow_unknown": false,                 // 期望拒答
  "notes": "设计意图"
}
```

### 为什么 gold 用文本片段而不是 chunk_id

`chunk_id` 依赖分块参数。把叶子块从 800 字符改成 600，所有 id 当场失效，
标注集整份作废。而"这段文字应当被检索到"是与分块策略解耦的。

代价是判定比 id 相等稍慢，换来的是**标注集能在分块消融中反复复用**——
做消融实验时这一点是决定性的，否则每换一次参数就要重标一遍全部样本。

片段请保持简短（建议 < 60 字），确保能完整落在单个叶子块内；
跨块的长片段会因分块边界切分而永远匹配不上。

## 标签体系

16 类标签定义在 `src/scout/evaluation/taxonomy.py`，每类都对应一个**已知会翻车的机制**：

| 标签 | 期望行为 | 考察点 |
|---|---|---|
| `single_fact` | 作答 | 主链路基线 |
| `definition` | 作答 | 摘要能力 |
| `parameter` | 作答（数字必须精确） | 重排与精确匹配 |
| `cross_document` | 作答 | 多路召回与融合 |
| `multi_hop` | 作答 | 子问题分解 |
| `comparison` | 作答（须覆盖双方） | 只答一方即为片面 |
| `time_version` | 作答（区分时点） | 快照与时效 |
| `table` | 作答 | 分块是否破坏结构 |
| `code` | 作答 | 大小写与符号原样匹配 |
| `ambiguity` | **澄清或拒答** | 拒答校准 |
| `no_knowledge` | **明确拒答** | 编造即为失败 |
| `source_conflict` | 如实呈现分歧 | 不擅自择一 |
| `near_entity` | 作答且不被误导 | 表面相似度干扰 |
| `typo` | 作答（经改写纠正） | 缺陷诊断式改写 |
| `long_question` | 作答且要点齐全 | 复杂度路由 |
| `prompt_injection` | 作答且不泄漏 | 消毒与归因门控 |

## 当前缺口

- **`source_conflict` 与 `code` 两类暂空**。当前语料中不存在互相冲突的来源，
  也没有代码类文档。**刻意不编造凑数的样本**——一条不是冲突的"冲突样本"
  会让这个标签失去全部意义。补齐需要先扩语料。
- 样本量 19 条，**只具方向性，不构成统计结论**。扩到 200 条是本项目下一步的首要工作。
- 尚无标注一致性（Cohen's κ）报告。

## 常用命令

```bash
# 校验数据集（会检查标签合法性、拒答样本是否误带 gold、注入样本是否有禁用词）
scout eval validate --dataset evals/longdoc_v1.json

# 跑一次评测
scout eval run --dataset evals/longdoc_v1.json --corpus datasets/longdoc-gold \
  --out reports/pipeline.json --markdown reports/pipeline.md

# 只跑 Agent 模式
scout eval run --mode agent --out reports/agent.json --markdown reports/agent.md

# 模块级消融（产出边际贡献对照表，含负贡献）
scout eval ablation --outdir reports/ablation

# 只跑指定配置
scout eval ablation --only full,-merge,merge-replace,baseline-dense-only

# 当作回归门禁用（Recall@5 低于阈值时退出码非零）
scout eval run --fail-under 0.8
```
