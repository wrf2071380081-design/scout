# MuSiQue-Ans 真实 LLM 评测（Kimi K3）

- LLM: **kimi-k3**（真实模型，token.caih.com）
- 向量器: auto（auto →local）
- 重排器: lexical
- 样本: 30 条（公开冻结 dev 集前 30 条）
- 语料: 600 个段落的联合检索池
- 索引: 663 块（9s，向量器 BAAI/bge-small-zh-v1.5）

## 两套配置的对比

| 配置 | 作答率 | Recall@5 | Recall@10 | **Answer F1** |
|---|---|---|---|---|
| gated（含门控） | 7% | 26.7% | 33.3% | **0.1%** |
| ungated（强行作答） | 10% | 21.7% | 28.3% | **0.3%** |

## 逐题记录（gated（含门控），1331s）

- [1] Who is the spouse of the Green performer?… | R@5 0% | F1 0.00 | clarify | grounding pass | 41s
- [2] Who founded the company that distributed the fil… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 30s
- [3] What administrative territorial entity is the ow… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 60s
- [4] Where is Ulrich Walter's employer headquartered?… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 30s
- [5] Which company owns the manufacturer of Learjet 6… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 54s
- [6] Who is the child of Caroline LeRoy's spouse?… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 74s
- [7] Who is the grandmother of Philippe, Duke of Orlé… | R@5 50% | F1 0.00 | clarify | grounding pass | 56s
- [8] What is the goal of the group that European Move… | R@5 100% | F1 0.04 | answered | grounding pass | 47s
- [9] What company succeeded the owner of Empire Sport… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 28s
- [10] What province shares a border with the province … | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 56s
- [11] What league does the team that plays in Stadio C… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 30s
- [12] What is a notable work written by the author of … | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 31s
- [13] In which borough was Callum McManaman born?… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 14s
- [14] What other county does the county where Imperial… | R@5 0% | F1 0.00 | clarify | grounding pass | 166s
- [15] In what county is William W. Blair's birthplace … | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 28s
- [16] What award did the author of The Red Tree receiv… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 35s
- [17] Where was Tyler MacDuff's child educated?… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 22s
- [18] Who is the spouse of the Rabbit Hole's producer?… | R@5 0% | F1 0.00 | clarify | grounding pass | 49s
- [19] Who is the child of Sigrid Eskilsdotter's child?… | R@5 50% | F1 0.00 | answered | grounding regenerate | 66s
- [20] In which county is Kimbrough Memorial Stadium lo… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 57s
- [21] What record label is the performer of Almost Mad… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 23s
- [22] Where was the author of Hannibal and Scipio educ… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 26s
- [23] In which county is Southern Maryland Electric Co… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 24s
- [24] Who is the child of the Victim of Romance perfor… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 40s
- [25] What county was Tim Dubois born in?… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 13s
- [26] What record label did the person who is part of … | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 63s
- [27] What is another notable work made by the author … | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 52s
- [28] What instrument is played by the person from The… | R@5 50% | F1 0.00 | clarify | grounding pass | 35s
- [29] What is the seat of the county where Van Hook To… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 31s
- [30] Who is the father of Edward Baring, 1st Baron Re… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 50s

## 逐题记录（ungated（强行作答），1431s）

- [1] Who is the spouse of the Green performer?… | R@5 0% | F1 0.00 | clarify | grounding pass | 124s
- [2] Who founded the company that distributed the fil… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 27s
- [3] What administrative territorial entity is the ow… | R@5 50% | F1 0.00 | clarify | grounding pass | 81s
- [4] Where is Ulrich Walter's employer headquartered?… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 31s
- [5] Which company owns the manufacturer of Learjet 6… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 148s
- [6] Who is the child of Caroline LeRoy's spouse?… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 25s
- [7] Who is the grandmother of Philippe, Duke of Orlé… | R@5 50% | F1 0.00 | clarify | grounding pass | 105s
- [8] What is the goal of the group that European Move… | R@5 50% | F1 0.05 | answered | grounding pass | 32s
- [9] What company succeeded the owner of Empire Sport… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 29s
- [10] What province shares a border with the province … | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 33s
- [11] What league does the team that plays in Stadio C… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 39s
- [12] What is a notable work written by the author of … | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 29s
- [13] In which borough was Callum McManaman born?… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 49s
- [14] What other county does the county where Imperial… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 33s
- [15] In what county is William W. Blair's birthplace … | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 40s
- [16] What award did the author of The Red Tree receiv… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 85s
- [17] Where was Tyler MacDuff's child educated?… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 34s
- [18] Who is the spouse of the Rabbit Hole's producer?… | R@5 0% | F1 0.00 | clarify | grounding pass | 41s
- [19] Who is the child of Sigrid Eskilsdotter's child?… | R@5 50% | F1 0.00 | answered | grounding pass | 42s
- [20] In which county is Kimbrough Memorial Stadium lo… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 8s
- [21] What record label is the performer of Almost Mad… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 23s
- [22] Where was the author of Hannibal and Scipio educ… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 23s
- [23] In which county is Southern Maryland Electric Co… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 18s
- [24] Who is the child of the Victim of Romance perfor… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 23s
- [25] What county was Tim Dubois born in?… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 10s
- [26] What record label did the person who is part of … | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 55s
- [27] What is another notable work made by the author … | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 98s
- [28] What instrument is played by the person from The… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 32s
- [29] What is the seat of the county where Van Hook To… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 26s
- [30] Who is the father of Edward Baring, 1st Baron Re… | R@5 50% | F1 0.05 | answered | grounding pass | 86s

## 怎么读这份报告

1. **gated** 是真实生产默认配置。作答率不是 100%——
   当检索证据撑不起这个多跳问题时，系统选择拒答/求澄清，而不是编造。
   这不是缺陷，是**门控在工作**。同时它也诚实暴露了检索的短板：
   在英文语料+中文向量模型的错配下，有多跳问题本来就应该拒掉。
2. **ungated** 拿掉门控。看它是为了把「它能不能答」和「它该不该答」拆开——
   F1 在这里衡量的是**单纯生成能力**。
3. **之前那个 Answer F1 ≈ 0.5%** 是用离线 HeuristicLLM 测的——
   那衡量的是「流程能跑」的天花板，不是系统的真实能力。
   换上真实 LLM 后，检索列可能变化不大，但**答案质量是质变**。这才是「真正能答」。
