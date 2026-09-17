"""HITL（中断/恢复/时间旅行）的测试。

测试聚焦**行为契约**，不是"能跑通"：

- 只读工具不打扰人
- 写工具触发审批，审批通过才执行
- 审批人可以改参数
- 恢复时**不重复执行**已完成的副作用（这是整件事里最重要的一条）
- 时间旅行分叉不重复副作用
- 没人审批时按 fail-closed 策略拒绝，而不是静默放行
- 错配的审批被拒绝
"""

from __future__ import annotations

import pytest

from scout.hitl import (
    FileCheckpointStore,
    HumanDecision,
    InMemoryCheckpointStore,
    ResumableAgent,
    RiskLevel,
    RunStatus,
    SideEffectLedger,
    TimeoutPolicy,
    build_default_policy,
)
from scout.llm.base import LLMResponse, ToolCall
from scout.llm.scripted import ScriptedLLM
from scout.tools.actions import InMemoryActionExecutor, build_action_tools
from scout.tools.builtin import build_default_tools
from scout.tools.registry import ToolRegistry


def _agent(
    responses: list[LLMResponse],
    *,
    store=None,
    timeout_policy: TimeoutPolicy = TimeoutPolicy.REJECT,
    approval_deadline_seconds: float = 900.0,
) -> tuple[ResumableAgent, InMemoryActionExecutor]:
    executor = InMemoryActionExecutor()
    registry = ToolRegistry([*build_default_tools(), *build_action_tools(executor)])
    agent = ResumableAgent(
        ScriptedLLM(responses),
        registry,
        store=store or InMemoryCheckpointStore(),
        timeout_policy=timeout_policy,
        approval_deadline_seconds=approval_deadline_seconds,
    )
    return agent, executor


def _approve(request, *, edited_arguments=None, by="reviewer") -> HumanDecision:
    return HumanDecision(
        request_id=request.request_id,
        approved=True,
        edited_arguments=edited_arguments,
        decided_by=by,
    )


def _reject(request, *, comment="不行") -> HumanDecision:
    return HumanDecision(request_id=request.request_id, approved=False, comment=comment)


# —— 中断 ——


def test_readonly_tool_does_not_interrupt() -> None:
    """只读工具不应触发任何审批。"""

    agent, executor = _agent(
        [
            LLMResponse(tool_calls=[ToolCall(name="calculator", arguments={"expression": "1+2"})]),
            LLMResponse(content="结果是 3。"),
        ]
    )
    outcome = agent.start("算一下 1+2")
    assert outcome.status is RunStatus.COMPLETED
    assert outcome.answer == "结果是 3。"
    assert outcome.meta["approvals_required"] == 0


def test_side_effect_tool_triggers_approval() -> None:
    """写工具必须触发审批，且中断时状态已落盘。"""

    agent, executor = _agent(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="send_email",
                        arguments={"to": "boss@example.com", "subject": "周报", "body": "…"},
                    )
                ]
            ),
        ]
    )
    outcome = agent.start("把周报发给老板")
    assert outcome.status is RunStatus.AWAITING_APPROVAL
    assert outcome.needs_human
    assert outcome.request is not None
    assert outcome.request.tool_name == "send_email"
    # 中断时副作用尚未执行。
    assert executor.committed() == []
    # 状态已持久化，能被重新加载。
    assert agent.load_state(outcome.run_id) is not None


def test_unregistered_tool_defaults_to_confirm() -> None:
    """未登记的工具按"有副作用"处理——fail-closed，不是默认放行。"""

    policy = build_default_policy(ToolRegistry([]))
    assessment = policy.assess("some_unknown_tool", {})
    assert assessment.level is RiskLevel.CONFIRM


def test_destructive_tool_is_required_not_confirmable() -> None:
    """不可撤销的操作必须是 REQUIRED 级，且不能被 auto_approve 降级。"""

    agent, _executor = _agent(
        [
            LLMResponse(tool_calls=[ToolCall(name="delete_file", arguments={"path": "/tmp/x.txt"})]),
        ]
    )
    outcome = agent.start("删掉 /tmp/x.txt")
    assert outcome.request is not None
    assert outcome.request.assessment.level is RiskLevel.REQUIRED


def test_pii_escalation_by_argument() -> None:
    """参数里出现个人信息，比工具名更优先升级——按工具名分级会漏掉这一类。"""

    agent, _executor = _agent(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(name="create_ticket", arguments={"title": "处理用户 13800001111 的投诉"})
                ]
            ),
        ]
    )
    outcome = agent.start("创建工单")
    assert outcome.request is not None
    assert any("手机号" in reason for reason in outcome.request.assessment.escalations)


# —— 审批通过 / 拒绝 / 改参数 ——


def test_resume_executes_after_approval() -> None:
    """审批通过后继续执行，副作用恰好发生一次。"""

    agent, executor = _agent(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="send_email",
                        arguments={"to": "boss@example.com", "subject": "周报", "body": "…"},
                    )
                ]
            ),
            LLMResponse(content="已发送。"),
        ]
    )
    first = agent.start("把周报发给老板")
    assert first.needs_human

    approved = _approve(first.request)
    resumed = agent.resume(first.run_id, approved)
    assert resumed.status is RunStatus.COMPLETED
    assert resumed.answer == "已发送。"

    sent = [record for record in executor.history() if record.tool == "send_email"]
    assert len(sent) == 1
    assert sent[0].arguments["to"] == "boss@example.com"


def test_resume_with_rejection_skips_the_side_effect() -> None:
    """审批拒绝后，副作用不发生，模型拿到可读的拒绝原因。"""

    agent, executor = _agent(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="send_email",
                        arguments={"to": "boss@example.com", "subject": "周报", "body": "…"},
                    )
                ]
            ),
            LLMResponse(content="那我换个方式。"),
        ]
    )
    first = agent.start("把周报发给老板")
    resumed = agent.resume(first.run_id, _reject(first.request, comment="时间还早，别催"))

    assert resumed.status is RunStatus.COMPLETED
    assert executor.history() == []
    rejected_calls = [
        call for call in resumed.state.call_log if call.get("approved") is False
    ]
    assert rejected_calls, "拒绝必须留痕"


def test_reviewer_can_edit_arguments() -> None:
    """批准但改参数：审批不只是个开关，而是可以修正模型输出的介入点。"""

    agent, executor = _agent(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="create_ticket",
                        arguments={"title": "生产线告警", "priority": "critical"},
                    )
                ]
            ),
            LLMResponse(content="已建。"),
        ]
    )
    first = agent.start("给生产线告警建个工单")
    resumed = agent.resume(
        first.run_id,
        _approve(first.request, edited_arguments={"title": "生产线告警", "priority": "medium"}),
    )
    assert resumed.status is RunStatus.COMPLETED
    ticket = [record for record in executor.history() if record.tool == "create_ticket"]
    assert len(ticket) == 1
    assert ticket[0].arguments["priority"] == "medium"
    assert resumed.meta["argument_edits"] == 1


def test_wrong_request_id_is_rejected() -> None:
    """决策与请求对不上必须显式失败——错配的审批会批准一件没被审过的事。"""

    agent, _executor = _agent(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="send_email",
                        arguments={"to": "boss@example.com", "subject": "周报", "body": "…"},
                    )
                ]
            ),
        ]
    )
    first = agent.start("发邮件")
    mismatched = HumanDecision(request_id="req-wrong", approved=True, decided_by="x")
    from scout.errors import ScoutError

    with pytest.raises(ScoutError):
        agent.resume(first.run_id, mismatched)


# —— 幂等与副作用 ——


def test_resume_does_not_duplicate_side_effects() -> None:
    """恢复时不会重复执行已完成的副作用——整个 HITL 机制里最重要的一条。

    场景：第一步模型要发邮件（已批准执行），又要删文件（待审批）。
    恢复后循环从"删文件"那次调用继续，**邮件不能被再发一次**。
    """

    agent, executor = _agent(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="send_email",
                        arguments={"to": "boss@example.com", "subject": "周报", "body": "…"},
                    ),
                    ToolCall(name="delete_file", arguments={"path": "/tmp/x.txt"}),
                ]
            ),
            LLMResponse(content="都做完了。"),
        ]
    )
    first = agent.start("发周报并清理临时文件")
    assert first.needs_human
    # 第一个待审批的是 send_email
    assert first.request.tool_name == "send_email"

    resumed1 = agent.resume(first.run_id, _approve(first.request))
    # 现在停在 delete_file（REQUIRED 级）
    assert resumed1.status is RunStatus.AWAITING_APPROVAL
    assert resumed1.request.tool_name == "delete_file"

    resumed2 = agent.resume(resumed1.run_id, _approve(resumed1.request))
    assert resumed2.status is RunStatus.COMPLETED

    emails = [record for record in executor.history() if record.tool == "send_email"]
    deletes = [record for record in executor.history() if record.tool == "delete_file"]
    assert len(emails) == 1, "邮件不能被重复发送"
    assert len(deletes) == 1, "文件不能被重复删除"


def test_crash_restart_via_file_store() -> None:
    """换一个 Agent 实例、共享同一份文件存储，恢复不丢状态、不重复副作用。

    这模拟"进程崩溃/重启"，是"跨进程恢复"真正成立的检验——
    只有状态是纯数据且检查点持久化，这条才能成立。
    """

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = FileCheckpointStore(tmp)
        agent, executor = _agent(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall(
                            name="send_email",
                            arguments={"to": "a@example.com", "subject": "通知", "body": "…"},
                        )
                    ]
                ),
                LLMResponse(content="完成。"),
            ],
            store=store,
        )
        first = agent.start("发通知")
        assert first.needs_human

        # 重启：新实例，同一个 store。
        registry = ToolRegistry([*build_default_tools(), *build_action_tools(executor)])
        from scout.llm.scripted import ScriptedLLM as _Scripted

        restarted = ResumableAgent(
            _Scripted([LLMResponse(content="完成。")]),
            registry,
            store=store,
        )
        resumed = restarted.resume(first.run_id, _approve(first.request))
        assert resumed.status is RunStatus.COMPLETED
        emails = [record for record in executor.history() if record.tool == "send_email"]
        assert len(emails) == 1


# —— 时间旅行 ——


def test_fork_time_travel_does_not_duplicate_side_effects() -> None:
    """从过去分叉出一条新运行线：新 run 不能重复父 run 已经执行的副作用。

    这是把"时间旅行"做对的关键：历史状态回退了，但**世界没有回退**。
    所以分叉要继承父 run 的副作用账本，幂等键建在血缘上。
    """

    # ScriptedLLM 是顺序回放的：第一跑消耗前两条，分叉跑消耗后两条。
    email_call = ToolCall(
        name="send_email",
        arguments={"to": "boss@example.com", "subject": "周报", "body": "…"},
    )
    agent, executor = _agent(
        [
            LLMResponse(tool_calls=[email_call]),
            LLMResponse(content="已发送。"),
            LLMResponse(tool_calls=[email_call]),
            LLMResponse(content="已发送。"),
        ]
    )
    first = agent.start("发周报")
    resumed = agent.resume(first.run_id, _approve(first.request))
    assert resumed.status is RunStatus.COMPLETED
    assert len([record for record in executor.history() if record.tool == "send_email"]) == 1

    # 从 step 0 分叉（在发邮件之前）
    history = agent.history(first.run_id)
    assert history, "必须有检查点历史才能时间旅行"
    earliest = min(checkpoint.step for checkpoint in history)
    forked = agent.run_fork(first.run_id, at_step=earliest)

    # 分叉后的运行重新跑了一遍，但 send_email 的槽位已经在血缘账本上 → 只回放。
    assert forked.run_id != first.run_id
    emails = [record for record in executor.history() if record.tool == "send_email"]
    assert len(emails) == 1, "分叉不能重复发送邮件"
    assert forked.meta["effects_replayed"] >= 1, "分叉应当对副作用做幂等回放"


def test_fork_of_nonexistent_step_fails() -> None:
    agent, _executor = _agent([LLMResponse(content="ok")])
    outcome = agent.start("随便")
    from scout.errors import ScoutError

    with pytest.raises(ScoutError):
        agent.fork(outcome.run_id, at_step=999)


# —— 超时策略 ——


def test_timeout_policy_rejects_by_default() -> None:
    """审批超时且没人处理时，默认 fail-closed：拒绝该调用，运行继续。"""

    agent, executor = _agent(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="send_email",
                        arguments={"to": "boss@example.com", "subject": "周报", "body": "…"},
                    )
                ]
            ),
            LLMResponse(content="那我就先不发了。"),
        ],
        timeout_policy=TimeoutPolicy.REJECT,
        approval_deadline_seconds=0.0,
    )
    first = agent.start("发周报")
    assert first.needs_human
    assert first.request.expired, "deadline=0 应当立即可见为超时"

    # 没有人工决策，但请求已超时 → 按策略自动拒绝并继续
    resumed = agent.resume(first.run_id)
    assert resumed.status is RunStatus.COMPLETED
    assert executor.history() == []
    assert resumed.meta["decisions_rejected"] == 1
    assert resumed.state.meta["last_timeout_policy"] == "reject"
    # 必须可审计：自动决策的 source 是 timeout_policy，不是 human
    log = resumed.state.meta.get("approval_log", [])
    assert log and log[-1]["source"] == "timeout_policy"


def test_resume_without_decision_and_not_expired_fails() -> None:
    """没超时不允许静默继续——那等于把"等待审批"变成"默认放行"。"""

    agent, _executor = _agent(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="send_email",
                        arguments={"to": "boss@example.com", "subject": "周报", "body": "…"},
                    )
                ]
            ),
        ],
        approval_deadline_seconds=3600.0,
    )
    first = agent.start("发周报")
    from scout.errors import ScoutError

    with pytest.raises(ScoutError):
        agent.resume(first.run_id)


# —— 账本（单元级） ——


def test_ledger_slot_and_compensation_ordering() -> None:
    ledger = SideEffectLedger()
    first = ledger.record(
        run_id="run-1", step=1, call_index=0, tool_name="send_email", arguments={"to": "a"},
        compensator="recall_email", observation="已发送",
    )
    ledger.record(
        run_id="run-1", step=1, call_index=1, tool_name="create_ticket", arguments={"title": "t"},
        compensator="close_ticket", observation="已创建",
    )
    ledger.record(
        run_id="run-1", step=1, call_index=2, tool_name="delete_file", arguments={"path": "/x"},
    )

    assert ledger.already_executed("run-1:1:0")
    assert not ledger.already_executed("run-1:1:9")

    # 补偿计划必须逆序：先撤销后发生的。
    plan = ledger.compensation_plan()
    assert [item["compensator"] for item in plan] == ["close_ticket", "recall_email"]
    # 不可补偿的操作被单独列出来，正是审批人最该看到的信息。
    irreversible = ledger.irreversible()
    assert [record.tool_name for record in irreversible] == ["delete_file"]
    assert first.slot == "run-1:1:0"
