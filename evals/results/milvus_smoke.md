# Milvus 后端联机冒烟报告

- Milvus：`http://127.0.0.1:19530`（索引 HNSW / COSINE）
- 语料：3 篇（longdoc-gold）
- 检索深度：top-10｜查询：8 条

- 向量器：BAAI/bge-small-zh-v1.5（dim=512）

## 结论：Milvus 不可用

```
ScoutError: Milvus 同步失败且 strict=True，拒绝降级：MilvusException: <MilvusException: (code=2, message=Fail connecting to server on 127.0.0.1:19530, illegal connection params or server unavailable)>
```

**先把容器起来**：`docker start milvus-etcd milvus-minio milvus-standalone`
（Milvus standalone 依赖 etcd 与 MinIO，缺一不可。）