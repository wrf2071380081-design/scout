"""分块与父块合并的测试。

这里最重要的是 ``test_expand_preserves_ranking_identity``：它把本项目相对
经典 Auto-merging 的核心差异**固化成可执行断言**。
如果哪天有人把 EXPAND 模式改回"替换"，这条测试会失败。
"""

from __future__ import annotations

import re

from scout.rag.chunking import ChunkCatalog, split_document
from scout.rag.merge import EvidenceUnit, MergeMode, auto_merge

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _long_text(sections: int = 400) -> str:
    return "".join(
        f"第{i}节：这是一段用于测试分块层级结构的中文内容，长度足够触发多级切分。\n\n"
        for i in range(sections)
    )


def _dense_parent_fixture():
    """构造一个"同一父块下命中多个叶子块"的场景。"""

    chunk_set = split_document(
        _long_text(),
        document_id="doc-1",
        document_version="v1",
        filename="sample.md",
    )
    catalog = ChunkCatalog()
    catalog.add(chunk_set)
    for parent_id, children in catalog.children_of.items():
        parent = catalog.get(parent_id)
        if parent is not None and parent.level == 2 and len(children) >= 3:
            return chunk_set, catalog, parent, children
    raise AssertionError("未能构造出密集父块，请调整测试文本长度")


def test_three_levels_with_parent_links() -> None:
    chunk_set = split_document(
        _long_text(),
        document_id="doc-1",
        document_version="v1",
        filename="sample.md",
    )
    assert set(chunk_set.levels) == {1, 2, 3}
    assert len(chunk_set.levels[1]) >= 2
    assert len(chunk_set.levels[2]) > len(chunk_set.levels[1])
    assert len(chunk_set.levels[3]) > len(chunk_set.levels[2])

    for level1 in chunk_set.levels[1]:
        assert level1.parent_chunk_id is None
    for level2 in chunk_set.levels[2]:
        assert level2.parent_chunk_id is not None
        assert chunk_set.get(level2.parent_chunk_id).level == 1
    for level3 in chunk_set.levels[3]:
        assert level3.parent_chunk_id is not None
        assert chunk_set.get(level3.parent_chunk_id).level == 2
        assert len(chunk_set.ancestors(level3.chunk_id)) == 2


def test_chunk_identity_is_complete() -> None:
    chunk_set = split_document(
        _long_text(30),
        document_id="doc-9",
        document_version="v7",
        filename="x.md",
        metadata={"page": 3},
    )
    leaf = chunk_set.leaves()[0]
    identity = leaf.identity
    assert identity["document_id"] == "doc-9"
    assert identity["document_version"] == "v7"
    assert _HEX64.match(str(identity["content_hash"]))
    assert identity["characters"] > 0
    # 身份里不得出现正文，否则 trace 会撑爆且可能泄露敏感内容。
    assert "text" not in identity


def test_expand_preserves_ranking_identity() -> None:
    """EXPAND 模式的核心承诺：不改变检索单元，只附加父块上下文。"""

    _chunk_set, catalog, parent, child_ids = _dense_parent_fixture()
    children = [catalog.get(item) for item in child_ids]
    units = [EvidenceUnit(chunk=child, score=1.0 - index * 0.1) for index, child in enumerate(children)]

    expanded, expand_outcome = auto_merge(units, catalog, mode=MergeMode.EXPAND, threshold=2)
    replaced, replace_outcome = auto_merge(units, catalog, mode=MergeMode.REPLACE, threshold=2)

    # EXPAND：数量与顺序不变，检索身份仍是叶子块。
    assert len(expanded) == len(units)
    assert [unit.chunk.chunk_id for unit in expanded] == [unit.chunk.chunk_id for unit in units]
    assert expand_outcome.applied is True

    # 但上下文被扩展到了父块。
    assert expanded[0].merge_source == "parent_expanded"
    assert expanded[0].context_level == parent.level
    assert len(expanded[0].context_text) > len(expanded[0].chunk.text)
    assert expand_outcome.context_growth_chars > 0

    # REPLACE：单元数减少，且排序身份被换成了父块——这正是排序被破坏的原因。
    assert len(replaced) < len(units)
    assert any(unit.chunk.level == parent.level for unit in replaced)
    assert replace_outcome.replaced_chunks > 0

    # 分数在替换模式下被保留为子块最大值，避免"替换即降权"。
    assert replaced[0].score == max(unit.score for unit in units)


def test_merge_off_is_a_true_noop() -> None:
    _chunk_set, catalog, _parent, child_ids = _dense_parent_fixture()
    units = [EvidenceUnit(chunk=catalog.get(item), score=1.0) for item in child_ids]
    merged, outcome = auto_merge(units, catalog, mode=MergeMode.OFF, threshold=2)
    assert merged == units
    assert outcome.applied is False
    assert outcome.mode == "off"


def test_merge_threshold_suppresses_trigger() -> None:
    _chunk_set, catalog, _parent, child_ids = _dense_parent_fixture()
    units = [EvidenceUnit(chunk=catalog.get(item), score=1.0) for item in child_ids]
    _merged, outcome = auto_merge(units, catalog, mode=MergeMode.EXPAND, threshold=len(child_ids) + 1)
    assert outcome.applied is False


def test_catalog_is_cross_document() -> None:
    """跨文档候选必须能找到各自父块——这是单 ChunkSet 实现会静默出错的地方。"""

    catalog = ChunkCatalog()
    sets = [
        split_document(
            _long_text(60),
            document_id=f"doc-{index}",
            document_version="v1",
            filename=f"f{index}.md",
        )
        for index in range(3)
    ]
    for chunk_set in sets:
        catalog.add(chunk_set)
    assert catalog.size == sum(len(item.all_chunks()) for item in sets)
    for chunk_set in sets:
        for leaf in chunk_set.leaves():
            assert catalog.parent_of[leaf.chunk_id] == leaf.parent_chunk_id
            assert catalog.document_of[leaf.chunk_id] == chunk_set.document_id
