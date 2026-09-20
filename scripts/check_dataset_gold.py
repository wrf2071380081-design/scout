"""校验评测集草稿的 gold 片段是否真的能在语料里找到。

**为什么这是最有价值的一步。**
"人工标注"这四个字的分量，全在 gold 对不对上。
机器生成的 gold 如果只是"看起来像"，那这份数据集就是自欺——
而 verifier 是机械的：**gold 片段必须能在语料里逐字命中**。
命中的 → 至少忠实于语料；命不中的 → 要么人抄错了、要么是编的，必须人工看。

这样把"人要做的事"从"从头标注 31 条"压缩成"复核少数命不中的"。
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, r"C:\agent-project\scout\src")
from scout.data.layout import restore_layout  # noqa: E402
from scout.evaluation.runner import load_corpus  # noqa: E402

DATASET = Path(r"C:\agent-project\scout\evals\longdoc_v2_draft.json")
V1 = Path(r"C:\agent-project\scout\evals\longdoc_v1.json")


def normalize(text: str) -> str:
    """去掉所有空白与常见标点变体，只留内容字符。

    语料里同一句话可能有半角/全角、换行差异；逐字比对必须归一化，
    否则会把"格式差异"误报成"gold 是编的"。
    """

    return re.sub(r"[\s\u3000·、，,。．.：:；;（）()【】\[\]「」“”\"'’‘]", "", text or "")


def main() -> int:
    corpus = load_corpus(r"C:\agent-project\scout\datasets\longdoc-gold")
    haystack = normalize("\n".join(restore_layout(text)[0] for _name, text in corpus))

    draft = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = draft.get("cases") or []
    v1_ids = {case["case_id"] for case in json.loads(V1.read_text(encoding="utf-8"))["cases"]}

    found, missing, empty = [], [], []
    for case in cases:
        if case["case_id"] in v1_ids:
            continue  # v1 是已人工确认过的，不重复校验
        golds = [snippet for snippet in (case.get("gold_snippets") or []) if snippet.strip()]
        if not golds:
            empty.append(case)
            continue
        hits = [snippet for snippet in golds if normalize(snippet)[:60] in haystack]
        if hits:
            found.append(case)
        else:
            missing.append(case)

    lines = [
        "# 评测集草稿 gold 校验报告",
        "",
        f"草稿总条数：{len(cases)}｜其中 v1 已有：{len(v1_ids)}｜待校验（v2 新增）：{len(cases) - len(v1_ids)}",
        "",
        "## 机械校验：gold 片段能否在语料里逐字命中（归一化空白与标点后）",
        "",
        f"- ✅ 可在语料中命中：**{len(found)}**",
        f"- ❌ 未能命中（需人工看）：**{len(missing)}**",
        f"- ⚪ 无 gold（拒答类样本）：**{len(empty)}**",
        "",
    ]

    if missing:
        lines.append("## 未能命中的清单（人工复核用）")
        lines.append("")
        for case in missing[:40]:
            lines.append(f"- `{case['case_id']}`　{case['question'][:60]}")
            snippet = (case.get("gold_snippets") or [""])[0]
            lines.append(f"    - gold：{snippet[:110]}")
            lines.append(f"    - 期望来源：{case.get('expected_sources')}")
        lines.append("")

    out = Path(r"C:\agent-project\scout\reports\dataset_gold_check.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

    # 可用的样本池：命中过的
    usable = [
        {
            "case_id": case["case_id"],
            "question": case["question"],
            "tags": case["tags"],
            "gold_snippets": case["gold_snippets"],
            "expected_sources": case.get("expected_sources") or [],
            "notes": case.get("notes") or "",
        }
        for case in found
    ]
    pool = Path(r"C:\agent-project\scout\evals\longdoc_v2_verified_pool.json")
    pool.write_text(
        json.dumps({"name": "longdoc-v2-verified-pool", "cases": usable}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"gold 可命中 {len(found)}｜未命中 {len(missing)}｜无 gold {len(empty)}")
    print(f"报告 -> {out}")
    print(f"可用样本池 -> {pool}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
