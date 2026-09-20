"""生成评测集人工复核清单：让人只做"人才能做的那部分"。

**为什么不是让人从零标注。**
草稿里 181 条新样本的 gold 已经写好，并且通过了机械校验
（gold 片段能在语料里逐字命中 → 至少忠实于语料，不是编的）。
所以人不需要"写 gold"，只需要判断一件**机器判断不了的事**：

    **这段 gold 真的回答了这个问题吗？**

忠实 ≠ 正确。"这段话确实在语料里"和"这段话是这道题的答案"是两件事——
后者需要读懂问题与证据的关系，这正是人工复核不可替代的部分。

复核清单按标签分组、按关联度排序，让人可以批量看同类题，降低切换成本。
"""

from __future__ import annotations

import json
from pathlib import Path

POOL = Path(r"C:\agent-project\scout\evals\longdoc_v2_verified_pool.json")
OUT = Path(r"C:\agent-project\scout\reports\dataset_review_sheet.md")

TAG_ORDER = [
    "single_fact", "parameter", "definition", "table", "time_version",
    "cross_document", "multi_hop", "comparison", "long_question",
    "near_entity", "typo", "ambiguity", "no_knowledge", "prompt_injection",
    "source_conflict", "code",
]


def main() -> int:
    payload = json.loads(POOL.read_text(encoding="utf-8"))
    cases = payload["cases"]

    grouped: dict[str, list[dict]] = {}
    for case in cases:
        primary = (case.get("tags") or ["unknown"])[0]
        grouped.setdefault(primary, []).append(case)

    lines: list[str] = []
    lines.append("# 评测集人工复核清单")
    lines.append("")
    lines.append(f"共 **{len(cases)}** 条待复核。gold 已通过机械校验（能在语料里逐字命中），")
    lines.append("所以你要判断的只有一件事：**这段 gold 真的回答了这个问题吗？**")
    lines.append("")
    lines.append("## 怎么用（预计 30–45 分钟）")
    lines.append("")
    lines.append("在每条末尾的 `判定：` 后面填 **对 / 错 / 改**：")
    lines.append("")
    lines.append("- `对` —— gold 能回答问题，直接采纳")
    lines.append("- `错` —— gold 不回答问题（请在后面用一句话说明哪里不对）")
    lines.append("- `改` —— 方向对但不精确（请把更准的原文片段贴上去，可从语料里复制）")
    lines.append("")
    lines.append("填完把文件发回来，我按判定结果生成最终数据集并跑一次评测。")
    lines.append("**不确定的就标 `错`**，宁缺毋滥——一条错的 gold 会污染整份数据集的可信度。")
    lines.append("")
    lines.append("---")
    lines.append("")

    counter = 0
    for tag in TAG_ORDER + [key for key in grouped if key not in TAG_ORDER]:
        items = grouped.get(tag)
        if not items:
            continue
        lines.append(f"## {tag}（{len(items)} 条）")
        lines.append("")
        for case in items:
            counter += 1
            gold = (case.get("gold_snippets") or [""])[0]
            lines.append(f"**{counter}. `{case['case_id']}`**")
            lines.append("")
            lines.append(f"- 问题：{case['question'].replace('根据知识库回答：', '')}")
            lines.append(f"- gold：{gold}")
            if case.get("expected_sources"):
                lines.append(f"- 来源：{case['expected_sources'][0]}")
            lines.append(f"- 判定：")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"合计 {counter} 条。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"复核清单 -> {OUT}（{counter} 条，按 {len(grouped)} 个标签分组）")
    for tag, items in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        print(f"   {tag:<20} {len(items):>3} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
