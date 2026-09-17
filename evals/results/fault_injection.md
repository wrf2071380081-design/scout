# 故障注入报告

> 用脚本化故障在可预期的位置注入失败，验证自愈编排的行为。
> 数据由 `scripts/fault_injection.py` 生成，可离线复现。

## 汇总

| 指标 | 值 | 解读 |
|---|---|---|
| 场景数 | 8 | 注入的故障类型数 |
| 类型化失败率 | 100.0% | 失败结束时是否带稳定错误码 |
| 预算内恢复率 | 100.0% | 重试 ≤ 预算后完成或优雅降级 |
| 未捕获裸异常 | 0 | 崩溃逃逸（应为 0） |

## 场景明细

| 场景 | 注入的故障 | 期望行为 | 实际结果 | 恢复 |
|---|---|---|---|---|
| `provider_timeout_once` | ProviderError(retryable=True) at task=decide ×1 | 重试一次后正常完成，恢复统计里可见这次重试 | answered | ✓ |
| `provider_persistent_failure` | ProviderError(retryable=True) 永久 | 预算耗尽后类型化失败（provider_timeout），不崩溃不静默 | provider_timeout | ✓ |
| `malformed_tool_args` | calculator(expression='1+(') | Schema 校验失败 → 错误回灌 → 模型修正或诚实停止 | answered | ✓ |
| `repeat_call_loop` | 同一 calculator 调用 ×6 | 重复调用被抑制，预算终止或最终作答，不崩溃不烧穿预算 | answered | ✓ |
| `unknown_tool` | ToolCall(name='nonexistent_tool') | 类型化错误（tool_not_found），Agent 继续或诚实停止 | answered | ✓ |
| `corrupted_checkpoint` | 从外部写入与 schema 不符的检查点 | ValidationError 显式失败，不渲染半对状态 | — | ✓ |
| `crash_mid_jsonl` | 最后一个 JSONL 行截断 | 跳过损坏行、从上一份完好的检查点恢复，不丢全部状态 | — | ✓ |
| `hitl_timeout_reject` | timeout_policy=REJECT + approval_deadline=0 | 自动拒绝该调用并继续，副作用不发生 | completed | ✓ |

> 这不是“系统不会出错”的证明——是“系统出错时**败得可控、败得可解释**”的证明。
> 任何一行 “✗” 都表示那一条机制的分支尚未实现或还有 bug，应该直接被提 issue。