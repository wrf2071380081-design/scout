"""基础数据层测试：版面还原、去重、版本与入库流水线（全部离线、毫秒级）。"""

from __future__ import annotations

from scout.data import (
    VersionRegistry,
    content_hash,
    deduplicate,
    detect_columns,
    hamming,
    ingest,
    merge_cross_page_tables,
    reading_order_ratio,
    reorder_columns,
    simhash,
    split_sections,
)

# —— 双栏版面 ——

DUAL_COLUMN = "\n".join(
    [
        "第一章 总体要求        第二章 重点任务",
        "本标准规定了术语定义        本标准明确了实施路径",
        "术语应保持前后一致        路径应分阶段推进",
        "定义需覆盖全部场景        推进要有量化指标",
        "第二章内容参见右栏        具体指标见附录 A",
        "附录 A 给出完整清单        附录 B 给出责任分工",
        "本页结束        本页结束",
        "编制说明另附        修订记录另附",
    ]
)

SINGLE_COLUMN = "\n".join(
    [
        "第一章 总体要求",
        "本标准规定了术语定义。",
        "术语应保持前后一致。",
        "定义需覆盖全部场景。",
        "具体指标见附录 A。",
        "附录 A 给出完整清单。",
        "编制说明另附。",
        "修订记录另附。",
    ]
)


def test_detect_dual_column() -> None:
    columns, pairs = detect_columns(DUAL_COLUMN)
    assert columns == 2
    assert pairs, "双栏文本应当能被切开"


def test_single_column_not_split() -> None:
    columns, _pairs = detect_columns(SINGLE_COLUMN)
    assert columns == 1, "单栏文本被误判成双栏，会把正文切碎"


def test_reorder_puts_left_column_first() -> None:
    reordered, count = reorder_columns(DUAL_COLUMN)
    assert count > 0
    lines = [line for line in reordered.splitlines() if line.strip()]
    # 左栏整列在前：第一行应当是左栏的第一条内容
    assert lines[0].startswith("第一章 总体要求")
    # 右栏内容被推到后半部分
    assert any("第二章 重点任务" in line for line in lines[: len(lines) // 2 + 2])


def test_cross_page_table_repeats_header() -> None:
    text = "\n".join(
        [
            "指标    2024    2025",
            "营收    100     120",
            "--- 第 2 页 ---",
            "利润     20      25",
        ]
    )
    merged, merged_count, repeated = merge_cross_page_tables(text)
    assert merged_count >= 1
    assert repeated >= 1
    assert merged.count("指标    2024    2025") >= 2, "续页应当补写表头"


def test_reading_order_ratio_prefers_coherent_text() -> None:
    coherent = "\n".join(f"这是第{index}段完整的说明文字，结尾有标点。" for index in range(25))
    scrambled = "\n".join(f"这是第{index}段被硬断开的文字没有结尾标点" for index in range(25))
    assert reading_order_ratio(coherent) > 0.9
    assert reading_order_ratio(scrambled) < 0.2


def test_halfwidth_punctuation_is_recognised() -> None:
    """半角句号 ｡ 是 PDF 转文本的常见变体，必须认。

    真实语料里有一篇用全篇半角句号的文档，早期标点集只认 `。`，
    结果它每一行都被算成"硬断"，被打 0 分——那不是版面错误，
    是标点变体没被识别，但报告上看起来像是文档有问题。
    """

    halfwidth = "\n".join(f"这是第{index}段说明文字，用半角句号结尾｡" for index in range(25))
    assert reading_order_ratio(halfwidth) > 0.9


def test_table_heavy_document_is_not_judged() -> None:
    """表格密集文档 → "无法判断"（返回 1.0），而不是"不合格"。

    这是真实误杀换来的结论：表格行本来就不以句号结尾，
    用散文的规矩去量表格，低分反映的是**体裁**，不是错误。
    用"测不准"去拒绝入库，是把不确定性当成否定证据。
    """

    rows = ["项目名称 本期金额 上期金额"] + [
        f"科目{index} 1234567.89 9876543.21" for index in range(40)
    ]
    table = "\n".join(rows)
    assert reading_order_ratio(table) == 1.0, "散文行不足时不该下结论"

    sections, report, _registry = ingest([("table.md", table)])
    assert report.low_quality == [], "默认不应拒收"
    assert report.kept >= 1
    assert "table.md" in report.reading_order, "分数要落到报告里供人工判断"


def test_quality_gate_only_acts_when_explicitly_enabled() -> None:
    """要拒收就必须显式设阈值——让"拦掉一些文档"成为有人签字的动作。"""

    scrambled = "\n".join(f"这是第{index}段被硬断开的文字没有结尾标点字数够了" for index in range(25))
    sections, report, _registry = ingest([("bad.md", scrambled)], min_reading_order=0.5)
    assert report.low_quality, "显式设了阈值就该拦"
    assert sections == []


# —— 去重 ——


def test_exact_duplicate_dropped() -> None:
    items = [("a", "云计算标准体系包括六个部分"), ("b", "云计算标准体系包括六个部分")]
    kept, report = deduplicate(items)
    assert len(kept) == 1
    assert report.exact_duplicates == 1
    assert report.dropped[0]["reason"] == "exact"


def test_near_duplicate_dropped_by_simhash() -> None:
    base = "本标准规定了云计算的术语、参考架构、服务模式与实施路径等基础性内容，适用于各类组织实施。"
    tweaked = "本标准规定了云计算的术语、参考架构、服务模式与实施路径等基础性内容，适用各类组织实施。"  # 差 1 字
    distance = hamming(simhash(base), simhash(tweaked))
    assert distance <= 3, f"改动一个字不应让指纹大幅漂移（实际差 {distance} 位）"

    kept, report = deduplicate([("a", base), ("b", tweaked)], near_threshold=3)
    assert len(kept) == 1
    assert report.near_duplicates == 1


def test_unrelated_text_not_deduped() -> None:
    kept, report = deduplicate(
        [
            ("a", "云计算标准体系包括基础、技术、服务、应用、管理和安全六个部分。"),
            ("b", "低空经济标准体系涵盖低空航空器、起降设施与运行服务等领域。"),
        ]
    )
    assert len(kept) == 2
    assert report.near_duplicates == 0


def test_content_hash_ignores_whitespace() -> None:
    assert content_hash("云计算  标准") == content_hash("云计算\n标准")


# —— 版本 ——


def test_version_bumps_only_on_change() -> None:
    registry = VersionRegistry()
    first, changed = registry.upsert("doc1", "内容 A")
    assert changed and first.version == 1
    same, changed_again = registry.upsert("doc1", "内容 A")
    assert not changed_again and same.version == 1, "内容没变就不该升版本"
    second, changed_third = registry.upsert("doc1", "内容 B")
    assert changed_third and second.version == 2


def test_soft_delete_keeps_history() -> None:
    registry = VersionRegistry()
    registry.upsert("doc1", "内容 A")
    tombstone = registry.soft_delete("doc1")
    assert tombstone is not None and tombstone.deleted
    assert registry.history("doc1"), "软删除必须保留历史（可审计、可恢复）"
    assert not list(registry.active())


def test_cache_key_covers_content_and_model() -> None:
    registry = VersionRegistry()
    record, _ = registry.upsert("doc1", "内容 A")
    key_v1 = record.cache_key(model_version="v1")
    key_v2 = record.cache_key(model_version="v2")
    assert key_v1 != key_v2, "换模型必须让缓存失效，否则会撞上旧维度向量"


# —— 结构切分与入库 ——


def test_split_sections_breaks_on_heading() -> None:
    text = "# 一\n第一段内容。\n# 二\n第二段内容。"
    sections = split_sections("doc", text)
    assert len(sections) == 2
    assert "第一段内容" in sections[0].text
    assert "第二段内容" in sections[1].text


def test_ingest_reports_and_rejects_garbled() -> None:
    good = "# 一\n云计算标准体系包括六个部分。\n# 二\n低空经济标准体系涵盖航空器。"
    sections, report, registry = ingest([("good.md", good)])
    assert report.documents == 1
    assert report.kept >= 1
    assert report.versions_created == 1
    assert report.layout, "版面报告必须落盘，否则语料质量问题无从追查"
    assert all(section.doc_id == "good.md" for section in sections)


def test_ingest_dedups_across_documents() -> None:
    same = "# 一\n这段内容在库里出现了两次，会让 Top-K 被同一条占满。"
    sections, report, _registry = ingest([("a.md", same), ("b.md", same)])
    assert report.documents == 2
    # 两个文档各自切出一节，但内容一致 → 精确去重掉一个
    assert report.dropped_exact >= 1
    assert len(sections) == 1, "重复内容不得同时进入检索库"
