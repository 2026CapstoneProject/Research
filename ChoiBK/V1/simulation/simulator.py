"""
시뮬레이션 드라이버
에피소드 전체를 실행하고 로그를 반환한다.
"""

import os
from datetime import datetime
from typing import Callable, Dict, List, Optional

from data.loader import WIPData, JobData
from data.params import MAX_SIM_STEPS, SHIFT_END, STACK_TO_NODE
from env.state import State
from env.actions import (
    Action, CRANE_MOVE, CRANE_TEMP_MOVE, CRANE_RESTORE, CRANE_PRE_POSITION,
    CRANE_WAIT, PROD_CONTINUE, PROD_NONE,
)
from env.transition import transition, get_tau
from env.cost import step_cost, terminal_cost, episode_summary


def run_episode(
    initial_state: State,
    wip_data:      Dict[int, WIPData],
    job_data:      Dict[int, JobData],
    inter_times:   Dict,
    machine_times: Dict,
    policy:        Callable,
    verbose:       bool = True,
    output_path:   Optional[str] = None,
) -> List[dict]:
    """
    에피소드 전체 시뮬레이션.

    policy: Callable(state, wip_data, job_data, machine_times) → (Action, float)
    output_path: 결과를 저장할 .md 또는 .txt 파일 경로 (None이면 저장 안 함)

    Returns:
        log: 스텝별 기록 리스트
          {step, state_before, action, cost, lookahead_cost, state_after}
    """
    s = initial_state.copy()
    log: List[dict] = []

    header_lines = [
        "=" * 60,
        "시뮬레이션 시작",
        f"총 WIP: {sum(len(v) for v in s.stacks.values())}",
        f"총 Job: {len(s.Q_rem)}",
        "=" * 60,
    ]
    if verbose:
        for line in header_lines:
            print(line)

    step_lines: List[str] = []

    for step_num in range(MAX_SIM_STEPS):
        if s.is_terminal():
            end_lines = [
                f"\n[Step {step_num}] 에피소드 종료",
                f"  완료 job: {sorted(s.Q_done)}",
                f"  미완료 job: {sorted(s.Q_rem)}",
                f"  최종 clock: {s.clock:.1f}분",
            ]
            if verbose:
                for line in end_lines:
                    print(line)
            step_lines.extend(end_lines)
            break

        action, lh_cost = policy(s, wip_data, job_data, machine_times)

        # 무인가공 시간대 WAIT|NONE 상태: 다음 사이클에서 할 수 있는 run이
        # 전혀 없으면 더 진행해도 상태가 개선되지 않으므로 종료한다.
        # (원자재 run이나 야드 WIP이 남아 있으면 다음 사이클에서 재개 가능)
        if (
            s.is_unm(SHIFT_END)
            and action.crane.type == CRANE_WAIT
            and action.prod.type == PROD_NONE
            and _no_future_work(s, wip_data, job_data)
        ):
            end_lines = [
                f"\n[Step {step_num}] 더 이상 처리 가능한 job 없음. 종료",
                f"  완료 job: {sorted(s.Q_done)}",
                f"  미완료 job: {sorted(s.Q_rem)}",
                f"  최종 clock: {s.clock:.1f}분",
            ]
            if verbose:
                for line in end_lines:
                    print(line)
            step_lines.extend(end_lines)
            break

        tau = get_tau(action.crane, s, inter_times, machine_times)

        cost = step_cost(s, action, job_data, tau)

        s_next = transition(s, action, wip_data, job_data, inter_times, machine_times)

        log.append({
            "step":           step_num,
            "state_before":   s,
            "action":         action,
            "tau":            tau,
            "cost":           cost,
            "lookahead_cost": lh_cost,
            "state_after":    s_next,
        })

        line = _format_step(step_num, s, action, cost, tau)
        step_lines.append(line)
        if verbose:
            print(line)

        s = s_next

    t_cost = terminal_cost(s, job_data=job_data)
    t_line = f"\nTerminal cost: {t_cost:.2f}"
    if verbose:
        print(t_line)
    step_lines.append(t_line)

    if output_path is not None:
        _save_result(
            output_path, log, wip_data, job_data,
            header_lines, step_lines, t_cost,
        )
        print(f"\n결과 저장 완료: {output_path}")

    return log


def _format_step(step: int, state: State, action: Action, cost: float, tau: float) -> str:
    """스텝 정보 문자열 반환"""
    phase = state.phase.name
    return (
        f"[{step:3d}] t={state.clock:6.1f}min | {phase:8s} | "
        f"{str(action):55s} | cost={cost:+.2f} | τ={tau:.1f}min"
    )


def _print_step(step: int, state: State, action: Action, cost: float, tau: float):
    """스텝 정보 출력 (하위 호환)"""
    print(_format_step(step, state, action, cost, tau))


def _build_summary_lines(log: List[dict], job_data: Optional[Dict[int, JobData]] = None) -> List[str]:
    """에피소드 요약 문자열 리스트 생성 (출력·저장 공용)"""
    if not log:
        return ["로그가 비어있습니다."]

    summary = episode_summary(log, job_data=job_data)

    lines = [
        "",
        "=" * 60,
        "에피소드 요약",
        "=" * 60,
        f"  총 스텝:           {summary['n_steps']}",
        f"  완료 job:          {summary['jobs_done']}",
        f"  미완료 job:        {summary['jobs_remain']}",
        f"  최종 시각:         {summary['clock_end']:.1f}분",
        f"  PICKING 횟수:         {summary['n_pickings']}",
        f"  START 횟수:        {summary['n_starts']}",
        f"  STORE 횟수:        {summary['n_stores']}",
        f"  MOVE 횟수:         {summary['n_moves']}",
        f"  TEMP_MOVE 횟수:    {summary['n_temp_moves']}",
        f"  RESTORE 횟수:      {summary['n_restores']}",
        f"  PRE_POSITION 횟수: {summary['n_pre_positions']}",
        f"  WAIT 횟수:         {summary['n_waits']}",
        f"  누적 비용:         {summary['total_cost']:.2f}",
        "=" * 60,
        "",
        "완료된 Job 상세:",
    ]

    start_entries = [e for e in log if e["action"].prod.type == "START_PROCESS"]
    for e in start_entries:
        s = e["state_before"]
        q = e["action"].prod.job_id
        lines.append(
            f"  Job {q:3d} | t={s.clock:6.1f}min | "
            f"K_mach={sorted(s.K_mach)} | "
            f"fill={s.u_short:.0f}/{s.u_long:.0f}mm"
        )

    return lines


def print_summary(log: List[dict], job_data: Optional[Dict[int, JobData]] = None):
    """에피소드 요약 출력"""
    for line in _build_summary_lines(log, job_data=job_data):
        print(line)


# 결과 파일 저장

def _save_result(
    output_path: str,
    log:         List[dict],
    wip_data:    Dict[int, WIPData],
    job_data:    Dict[int, JobData],
    header_lines: List[str],
    step_lines:   List[str],
    t_cost:       float,
) -> None:
    """
    시뮬레이션 결과를 마크다운(.md) 또는 텍스트(.txt) 파일로 저장한다.

    파일 구성:
      1. 메타 정보 (실행 시각, 정책, Job 목록)
      2. 전체 스텝 로그
      3. 적재 순서 요약 (어떤 WIP을 어떤 순서로 적재했는지)
      4. Run별 배치 결과 (Job ID, 투입 WIP, fill)
      5. 이동 이력 (MOVE / TEMP_MOVE / RESTORE)
      6. 에피소드 요약 통계
    """
    is_md = output_path.endswith(".md")

    lines: List[str] = []

    def h1(text):  return f"# {text}" if is_md else f"\n{'=' * 60}\n{text}\n{'=' * 60}"
    def h3(text):  return f"### {text}" if is_md else f"\n[{text}]"
    def bold(text): return f"**{text}**" if is_md else text
    def code(text): return f"`{text}`" if is_md else text

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines += [
        h1("시뮬레이션 결과"),
        "",
        f"- 실행 시각: {now}",
        f"- 총 WIP 수: {sum(len(v) for v in log[0]['state_before'].stacks.values()) if log else 'N/A'}",
        f"- 총 Job 수: {len(job_data)}",
        f"- Job 목록:  {sorted(job_data.keys())}",
        "",
    ]

    lines += [h2("Job 목록 및 입력재"), ""]
    if is_md:
        lines += [
            "| Job ID | 입력 WIP | 스택 | 레벨 | Spec | 공정시간(분) | C_short | C_long |",
            "|--------|----------|------|------|------|------------|---------|--------|",
        ]
        for jid, job in sorted(job_data.items()):
            wip = wip_data.get(job.input_wip_id)
            loc = f"{wip.stack_id}" if wip else "-"
            lv  = f"{wip.level}"   if wip else "-"
            lines.append(
                f"| {jid} | {job.input_wip_id} | {loc} | {lv} | "
                f"{job.spec} | {job.process_time:.1f} | "
                f"{job.cap_short:.0f} | {job.cap_long:.0f} |"
            )
    else:
        for jid, job in sorted(job_data.items()):
            wip = wip_data.get(job.input_wip_id)
            loc = f"stack={wip.stack_id} lv={wip.level}" if wip else "N/A"
            lines.append(
                f"  Job {jid:3d}: WIP {job.input_wip_id:3d} [{loc}] | "
                f"{job.spec:20s} | ptime={job.process_time:.1f}분 | "
                f"C_s={job.cap_short:.0f} C_l={job.cap_long:.0f}"
            )
    lines.append("")

    lines += [h2("크레인 적재 순서 (PICKING 이벤트)"), ""]
    picking_entries = [e for e in log if e["action"].crane.type == "PICKING"]
    if is_md:
        lines += [
            "| # | 시각(분) | WIP ID | 출처 | Job ID | 적재 후 u_short | 적재 후 u_long |",
            "|---|--------|--------|------|--------|--------------|--------------|",
        ]
        for idx, e in enumerate(picking_entries, 1):
            s   = e["state_before"]
            a   = e["action"].crane
            src = f"Stack {a.src_stack}" if a.src_stack is not None else "Buffer"
            wip = wip_data.get(a.wip_id)
            s_next = e["state_after"]
            lines.append(
                f"| {idx} | {s.clock:.1f} | {a.wip_id} | {src} | {a.job_id} | "
                f"{s_next.u_short:.0f} | {s_next.u_long:.0f} |"
            )
    else:
        for idx, e in enumerate(picking_entries, 1):
            s   = e["state_before"]
            a   = e["action"].crane
            src = f"Stack {a.src_stack}" if a.src_stack is not None else "Buffer"
            wip = wip_data.get(a.wip_id)
            spec = f"{wip.thickness:.0f}*{wip.short_side:.0f}*{wip.long_side:.0f}" if wip else "?"
            s_next = e["state_after"]
            lines.append(
                f"  [{idx:2d}] t={s.clock:6.1f}min | WIP {a.wip_id:3d}({spec}) "
                f"← {src} | Job {a.job_id} | "
                f"u={s_next.u_short:.0f}/{s_next.u_long:.0f}mm"
            )
    lines.append("")

    lines += [h2("Run별 배치 투입 결과"), ""]
    start_entries = [e for e in log if e["action"].prod.type == "START_PROCESS"]
    if is_md:
        lines += [
            "| Job ID | 시작 시각(분) | 투입 WIP 목록 | u_short(mm) | u_long(mm) | 비용 |",
            "|--------|------------|-------------|-----------|----------|------|",
        ]
        for e in start_entries:
            s = e["state_before"]
            q = e["action"].prod.job_id
            job = job_data.get(q)
            fill_pct = ""
            if job:
                pct_s = s.u_short / job.cap_short * 100 if job.cap_short else 0
                pct_l = s.u_long  / job.cap_long  * 100 if job.cap_long  else 0
                fill_pct = f" ({pct_s:.0f}%/{pct_l:.0f}%)"
            lines.append(
                f"| {q} | {s.clock:.1f} | {sorted(s.K_mach)} | "
                f"{s.u_short:.0f}{fill_pct} | {s.u_long:.0f} | {e['cost']:+.2f} |"
            )
    else:
        for e in start_entries:
            s = e["state_before"]
            q = e["action"].prod.job_id
            job = job_data.get(q)
            fill_pct = ""
            if job:
                pct_s = s.u_short / job.cap_short * 100 if job.cap_short else 0
                fill_pct = f" (단변 {pct_s:.0f}% 충진)"
            lines.append(
                f"  Job {q:3d} | t={s.clock:6.1f}min | "
                f"WIP {sorted(s.K_mach)}{fill_pct} | "
                f"u_short={s.u_short:.0f}mm u_long={s.u_long:.0f}mm"
            )
    lines.append("")

    move_entries = [
        e for e in log
        if e["action"].crane.type in (
            CRANE_MOVE, CRANE_TEMP_MOVE, CRANE_RESTORE, CRANE_PRE_POSITION
        )
    ]
    lines += [h2(f"크레인 재배치 이력 (MOVE/TEMP_MOVE/RESTORE/PRE_POSITION, 총 {len(move_entries)}건)"), ""]
    if move_entries:
        if is_md:
            lines += [
                "| # | 시각(분) | 행동 | WIP ID | 출처 | 목적지 | 비용 |",
                "|---|--------|------|--------|------|------|------|",
            ]
            for idx, e in enumerate(move_entries, 1):
                s = e["state_before"]
                a = e["action"].crane
                src = f"Stack {a.src_stack}" if a.src_stack is not None else "Buffer"
                dst = f"Stack {a.dst_stack}" if a.dst_stack is not None else "Buffer"
                lines.append(
                    f"| {idx} | {s.clock:.1f} | {a.type} | {a.wip_id} | "
                    f"{src} | {dst} | {e['cost']:+.2f} |"
                )
        else:
            for idx, e in enumerate(move_entries, 1):
                s = e["state_before"]
                a = e["action"].crane
                src = f"Stack {a.src_stack}" if a.src_stack is not None else "Buffer"
                dst = f"Stack {a.dst_stack}" if a.dst_stack is not None else "Buffer"
                lines.append(
                    f"  [{idx:2d}] t={s.clock:6.1f}min | {a.type:10s} | "
                    f"WIP {a.wip_id:3d} | {src} → {dst} | cost={e['cost']:+.2f}"
                )
    else:
        lines.append("  (재배치 없음)")
    lines.append("")

    # step_lines 구조: [스텝 로그 len(log)줄] + [에피소드종료 4줄] + [terminal cost 1줄]
    # zip 길이가 len(log)에서 끊기므로, 접미 줄을 미리 분리해야 유실되지 않는다.
    n_step_lines = len(log)
    step_log_lines  = step_lines[:n_step_lines]    # 스텝별 로그
    step_tail_lines = step_lines[n_step_lines:]    # 에피소드종료 + terminal cost

    filtered_log = [
        entry for entry in log
        if not _is_unmanned_wait_step(entry)
    ]
    filtered_step_lines = [
        line for line, entry in zip(step_log_lines, log)
        if not _is_unmanned_wait_step(entry)
    ]
    filtered_step_lines += step_tail_lines

    lines += [h2("전체 스텝 로그"), ""]
    if is_md:
        lines.append("```")
    lines += filtered_step_lines
    if is_md:
        lines.append("```")
    lines.append("")

    lines += [h2("에피소드 요약 통계"), ""]
    if len(filtered_log) != len(log):
        lines += [
            f"- 숨김 처리된 무인가공 WAIT 스텝 수: {len(log) - len(filtered_log)}",
            "- 아래 요약은 실제 전체 episode 기준이며, 위 스텝 로그 본문에서만 무인가공 WAIT 스텝을 생략했습니다.",
            "",
        ]
    lines += _build_summary_lines(log, job_data=job_data)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _is_unmanned_wait_step(entry: dict) -> bool:
    """
    결과 md에서는 무인가공 구간의 단순 WAIT step을 생략한다.
    설비가 BUSY이든 EMPTY이든 무인가공 시간대이고 크레인이 WAIT이면 대상이다.
    """
    state_before = entry["state_before"]
    action = entry["action"]
    return (
        state_before.is_unm(SHIFT_END)
        and action.crane.type == CRANE_WAIT
    )


def _no_future_work(
    state,
    wip_data: dict,
    job_data:  dict,
) -> bool:
    """
    남은 job 중 다음 교대 사이클(유인 근무 시간)에 처리 가능한 run이
    하나도 없으면 True를 반환한다.

    처리 가능 조건:
      1. has_external_input=True  → 원자재 공급으로 언제든 시작 가능
      2. unique input WIP가 야드 어딘가에 존재

    Phase 5: BLOCKED 상태(O_wait≠∅)이면 출력재 STORE가 아직 남아 있으므로 False 반환.
    """
    # 설비가 BLOCKED(출력재 대기) 상태면 아직 할 일이 있다
    if len(state.O_wait) > 0:
        return False

    from env.feasibility import _matches_job_template

    for jid in state.Q_rem:
        job = job_data.get(jid)
        if job is None:
            continue
        # 조건 1: 원자재 job (input_wip_id==0 포함, has_external_input=True)
        #   → 원자재 공급으로 언제든 DIRECT_START 가능
        if job.has_external_input:
            return False
        # 조건 2: unique job — 해당 WIP가 야드 어딘가에 존재
        for stk in state.stacks.values():
            if job.input_wip_id in stk:
                return False

    return True  # 처리 가능한 job 없음
