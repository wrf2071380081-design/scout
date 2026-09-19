# MuSiQue-Ans 真实 LLM 评测（Kimi K3）

- LLM: **kimi-k3**（真实模型，token.caih.com）
- 向量器: auto（auto → local）
- 重排器: cross → BAAI/bge-reranker-base
- 样本: 50 条（公开冻结 dev 集前 50 条）
- 语料: 1000 个段落的联合检索池
- 索引: 1098 块（18s，向量器 BAAI/bge-small-zh-v1.5）
- 重排器实例: BAAI/bge-reranker-base

## 结果

| 指标 | 均值 | bootstrap 95% CI |
|---|---|---|
| Recall@5 | 48.0% | [39.0, 57.0] |
| Recall@10 | 53.0% | [44.0, 62.0] |
| **Answer F1** | 1.9% | [0.9, 3.0] |

作答率：24%（12/50）
总耗时：2342s（50 条）

## 与离线启发式实现的对比

| 指标 | 离线 HeuristicLLM | 真实 Kimi K3 |
|---|---|---|
| Answer F1 | ~0.5% | **见上表** |
| 作答率 | ~50% | 见上 |

- 离线启发式是**词法抽取**，它不理解答案，只是从证据里拼词——
  F1 ≈0.5% 是从“流程能跑通”的测量天花板，不是真实能力。
- 换上真实 LLM 后 F1 的跳升，正好反证了那套分层评测的价值——
  之前那个 ~0.5% 不是系统烂，而是**离线替身的能力天花板**。

## 逐题记录

- [1/50] Who is the spouse of the Green performer?… | R@5 50% | F1 0.00 | clarify | grounding pass | 67s
- [2/50] Who founded the company that distributed the film … | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 22s
- [3/50] What administrative territorial entity is the owne… | R@5 0% | F1 0.00 | clarify | grounding pass | 55s
- [4/50] Where is Ulrich Walter's employer headquartered?… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 25s
- [5/50] Which company owns the manufacturer of Learjet 60?… | R@5 100% | F1 0.09 | answered | grounding pass | 33s
- [6/50] Who is the child of Caroline LeRoy's spouse?… | R@5 50% | F1 0.00 | clarify | grounding pass | 33s
- [7/50] Who is the grandmother of Philippe, Duke of Orléan… | R@5 50% | F1 0.00 | clarify | grounding pass | 25s
- [8/50] What is the goal of the group that European Moveme… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 34s
- [9/50] What company succeeded the owner of Empire Sports … | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 56s
- [10/50] What province shares a border with the province wh… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 30s
- [11/50] What league does the team that plays in Stadio Cir… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 48s
- [12/50] What is a notable work written by the author of Th… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 29s
- [13/50] In which borough was Callum McManaman born?… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 17s
- [14/50] What other county does the county where Imperial i… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 86s
- [15/50] In what county is William W. Blair's birthplace lo… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 35s
- [16/50] What award did the author of The Red Tree receive?… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 30s
- [17/50] Where was Tyler MacDuff's child educated?… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 43s
- [18/50] Who is the spouse of the Rabbit Hole's producer?… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 32s
- [19/50] Who is the child of Sigrid Eskilsdotter's child?… | R@5 50% | F1 0.00 | answered | grounding pass | 60s
- [20/50] In which county is Kimbrough Memorial Stadium loca… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 49s
- [21/50] What record label is the performer of Almost Made … | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 24s
- [22/50] Where was the author of Hannibal and Scipio educat… | R@5 0% | F1 0.00 | clarify | grounding pass | 36s
- [23/50] In which county is Southern Maryland Electric Coop… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 21s
- [24/50] Who is the child of the Victim of Romance performe… | R@5 100% | F1 0.05 | answered | grounding regenerate | 108s
- [25/50] What county was Tim Dubois born in?… | R@5 100% | F1 0.09 | answered | grounding regenerate | 31s
- [26/50] What record label did the person who is part of Th… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 25s
- [27/50] What is another notable work made by the author of… | R@5 100% | F1 0.03 | answered | grounding regenerate | 50s
- [28/50] What instrument is played by the person from The B… | R@5 50% | F1 0.00 | clarify | grounding pass | 39s
- [29/50] What is the seat of the county where Van Hook Town… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 42s
- [30/50] Who is the father of Edward Baring, 1st Baron Reve… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 67s
- [31/50] What group was the performer of Be the One a membe… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 84s
- [32/50] Who is the lead singer of the band performing Bang… | R@5 100% | F1 0.07 | answered | grounding regenerate | 30s
- [33/50] What is the network which National Cycle Route 57 … | R@5 100% | F1 0.14 | answered | grounding regenerate | 55s
- [34/50] Who is the spouse of the child of Peter Andreas He… | R@5 100% | F1 0.09 | answered | grounding regenerate | 100s
- [35/50] The Unwinding author volunteered for which organis… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 23s
- [36/50] What is the capital of the county that Pine Spring… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 37s
- [37/50] What district is the headquarter of Julia's House … | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 41s
- [38/50] Who is the spouse of Young Man Luther's author?… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 55s
- [39/50] What record label did the performer of The Place a… | R@5 50% | F1 0.02 | answered | grounding pass | 56s
- [40/50] Who is the child of the person who followed Tihomi… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 66s
- [41/50] What team was Anna Benson's husband on?… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 47s
- [42/50] What place does the administrative territorial ent… | R@5 0% | F1 0.00 | insufficient_evidence | grounding pass | 55s
- [43/50] Where was the spouse of Frances Tupper born?… | R@5 100% | F1 0.00 | insufficient_evidence | grounding pass | 61s
- [44/50] Who founded the political party of Dimuthu Bandara… | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 112s
- [45/50] Bancroft's county borders what county?… | R@5 50% | F1 0.00 | clarify | grounding pass | 58s
- [46/50] In which county is Mark Dismore's birthplace locat… | R@5 100% | F1 0.10 | answered | grounding regenerate | 46s
- [47/50] Who was the sibling of Nannina de' Medici?… | R@5 50% | F1 0.06 | answered | grounding pass | 26s
- [48/50] What county is the NRHEG High School located in?… | R@5 100% | F1 0.20 | answered | grounding pass | 29s
- [49/50] What league does the team that occupies the Rabat … | R@5 50% | F1 0.00 | insufficient_evidence | grounding pass | 66s
- [50/50] Who was married to the star of No Escape?… | R@5 50% | F1 0.00 | clarify | grounding pass | 40s
