"""MCP 服务器（stdio，零新增依赖）。"""

from __future__ import annotations

from .server import MCPServer, run_mcp

__all__ = ["MCPServer", "run_mcp"]
