"""scout — 长文档 Agentic RAG 框架。

设计目标：把「检索质量」和「Agent 决策」放在同一套可观测、可评测、可恢复的运行时里。

模块划分：

- ``scout.errors``        typed 错误分类：区分「无知识」「证据不足」「Provider 故障」
- ``scout.config``        环境变量配置
- ``scout.trace``         结构化链路追踪（可序列化、可脱敏）
- ``scout.llm``           LLM 客户端（OpenAI 兼容 / 可脚本化）
- ``scout.rag``           分块、混合检索、RRF、Auto-merge、重写、流水线
- ``scout.agent``         ReAct 循环与工具注册表
- ``scout.memory``        三层记忆（工作 / 情景 / 语义）与失效机制
- ``scout.verify``        证据接地校验、引用校验、检索侧注入防护
- ``scout.orchestrator``  失败分类与自愈恢复
- ``scout.evaluation``    评测数据集、指标、运行器与报告
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
