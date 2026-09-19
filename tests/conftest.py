"""测试全局配置。

**为什么测试必须锁定离线向量器。**

``build_index`` 的默认向量器是 ``auto``：装了 fastembed 就用本地语义模型
（``BAAI/bge-small-zh-v1.5``）。这对真实使用是对的，对测试是错的：

- 模型是否安装随环境而变 → 同一份测试在不同机器上跑出不同结果；
- 首次使用会触发约 90MB 权重下载 → 测试依赖网络；
- 语义向量的排序与词法向量不同 → 任何断言"某个块应该被检索到"的用例都会摇摆。

所以这里在导入 scout 之前把后端钉死为哈希向量器。
**要测语义向量本身**，请用 ``test_embed.py`` 里显式构造 ``LocalEmbedder`` 的用例——
显式构造是刻意写出来的，它不是隐式默认。
"""

from __future__ import annotations

import os

# 必须在 scout.config 被导入之前设置：get_settings() 会缓存首次读到的环境。
os.environ.setdefault("SCOUT_EMBED_BACKEND", "hashing")
# 同理把重排钉死为词法实现，避免测试触发 1GB 级重排模型下载。
os.environ.setdefault("SCOUT_RERANK_BACKEND", "lexical")
# 顺带钉死 LLM：不配置 base_url → 走离线启发式实现，测试不触网。
os.environ.pop("SCOUT_LLM_BASE_URL", None)
os.environ.pop("SCOUT_LLM_API_KEY", None)
