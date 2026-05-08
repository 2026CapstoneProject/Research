"""
전이 함수: S_{t+1} = S^M(S_t, x_t, W_{t+1})

가정:
  - output WIP은 unique하게 생성·적재되지만, 후속 job 입력으로 재사용되지는 않는다.
"""

from typing import Dict, Optional
import numpy as np
from data.params import (
    DELTA_MIN, SHIFT_END, STACK_TO_NODE, MACHINE_NODE, SIGMA_PTIME
)
from data.loader import WIPData, JobData, get_crane_time
from env.state import State, MachinePhase
from env.actions import (
    Action, CraneAction, ProdAction,
    CRANE_PICKING, CRANE_STORE, CRANE_MOVE, CRANE_TEMP_MOVE,
    CRANE_RESTORE, CRANE_PRE_POSITION, CRANE_WAIT,
    PROD_START, PROD_DIRECT_START, PROD_CONTINUE, PROD_NONE,
)

# 확률적 모드 플래그 (main.py의 --stochastic 옵션으로 활성화 가능)
_stochastic_mode: bool = False


def set_stochastic(enabled: bool) -> None:
    """확률적 생산시간 모드 전환 (main.py에서 호출)"""
    global _stochastic_mode
    _stochastic_mode = enabled


# ─────────────────────────────────────────────────────────────────
# 크레인 행동별 소요시간 τ(x_t^crane)
# ─────────────────────────────────────────────────────────────────

def get_tau(
    crane: CraneAction,
    state: State,
    inter_times: Dict,
    machine_times: Dict,
) -> float:
    """
    크레인 행동 소요시간 τ(x_t^crane) 계산 (분)
    V5_SDAM Section 6.4: c_{t+1} = c_t + τ(x_t^crane)
    τ(WAIT) = DELTA_MIN > 0 (무한루프 방지)
    """
    ctype = crane.type

    if ctype == CRANE_WAIT:
        # DIRECT_START도 CRANE_WAIT을 사용하므로 별도 처리 불필요
        return DELTA_MIN

    if ctype == CRANE_PICKING:
        # 야드 PICKING: 현재 위치 → 스택 → 설비
        # 버퍼 PICKING(src_stack=None): 현재 위치 → 버퍼(B-1) → 설비
        src_node = "B-1" if crane.src_stack is None else STACK_TO_NODE.get(
            crane.src_stack, state.crane_loc
        )
        # 스택까지 이동 + 설비까지 이동 (단순화: 설비 이동시간만 사용)
        t_to_stack = get_crane_time(state.crane_loc, src_node,
                                    inter_times, machine_times)
        t_to_mach  = machine_times.get(src_node, 5.0)
        return t_to_stack + t_to_mach

    if ctype == CRANE_STORE:
        # 설비 → 스택 이동
        dst_node = STACK_TO_NODE.get(crane.dst_stack, state.crane_loc)
        t_to_stack = machine_times.get(dst_node, 5.0)
        return t_to_stack

    if ctype == CRANE_MOVE:
        src_node = STACK_TO_NODE.get(crane.src_stack, state.crane_loc)
        dst_node = STACK_TO_NODE.get(crane.dst_stack, state.crane_loc)
        return get_crane_time(src_node, dst_node, inter_times, machine_times)

    if ctype == CRANE_TEMP_MOVE:
        # 스택 → 버퍼 (B-1로 이동)
        src_node = STACK_TO_NODE.get(crane.src_stack, state.crane_loc)
        return get_crane_time(src_node, "B-1", inter_times, machine_times)

    if ctype == CRANE_RESTORE:
        # 버퍼(B-1) → 스택
        dst_node = STACK_TO_NODE.get(crane.dst_stack, state.crane_loc)
        return get_crane_time("B-1", dst_node, inter_times, machine_times)

    if ctype == CRANE_PRE_POSITION:
        # Phase 3: 버퍼 → 전략적 스택 (RESTORE와 동일한 이동 시간)
        dst_node = STACK_TO_NODE.get(crane.dst_stack, state.crane_loc)
        return get_crane_time("B-1", dst_node, inter_times, machine_times)

    return DELTA_MIN


# ─────────────────────────────────────────────────────────────────
# 크레인 위치 갱신
# ─────────────────────────────────────────────────────────────────

def _new_crane_loc(crane: CraneAction, old_loc: str) -> str:
    """행동 후 크레인이 있을 노드 이름"""
    ctype = crane.type
    if ctype == CRANE_PICKING:
        return MACHINE_NODE
    if ctype == CRANE_STORE:
        dst = STACK_TO_NODE.get(crane.dst_stack)
        return dst if dst else old_loc
    if ctype in (CRANE_MOVE,):
        dst = STACK_TO_NODE.get(crane.dst_stack)
        return dst if dst else old_loc
    if ctype == CRANE_TEMP_MOVE:
        return "B-1"
    if ctype in (CRANE_RESTORE, CRANE_PRE_POSITION):
        dst = STACK_TO_NODE.get(crane.dst_stack)
        return dst if dst else old_loc
    return old_loc   # WAIT


# ─────────────────────────────────────────────────────────────────
# 메인 전이 함수
# ─────────────────────────────────────────────────────────────────

def transition(
    state: State,
    action: Action,
    wip_data: Dict[int, WIPData],
    job_data:  Dict[int, JobData],
    inter_times: Dict,
    machine_times: Dict,
) -> State:
    """
    S_{t+1} = S^M(S_t, x_t, W_{t+1})
    결정론 버전: W_{t+1}의 불확실 요소는 무시 (ω^ptime=0, ω^order=∅)
    """
    s = state.copy()
    crane = action.crane
    prod  = action.prod
    tau   = get_tau(crane, state, inter_times, machine_times)

    # ── (1) 설비 상태 전이 ────────────────────────────
    _update_machine(s, crane, prod, tau, wip_data, job_data)

    # ── (2) 야드 상태 전이 ────────────────────────────
    _update_yard(s, crane)

    # ── (3) 크레인 위치 갱신 ──────────────────────────
    s.crane_loc = _new_crane_loc(crane, state.crane_loc)

    # ── (4) 시간 전이 (Section 6.4) ───────────────────
    s.clock += tau
    # rem_shift, is_unm은 clock으로부터 즉시 계산 가능하므로 별도 저장 안 함

    s.step += 1
    return s


# ─────────────────────────────────────────────────────────────────
# 내부: 설비 상태 전이 (Section 6.3)
# ─────────────────────────────────────────────────────────────────

def _update_machine(
    s: State,
    crane: CraneAction,
    prod:  ProdAction,
    tau:   float,
    wip_data: Dict[int, WIPData],
    job_data:  Dict[int, JobData],
) -> None:
    """
    K_mach, j_mach, u_short, u_long, eta, phase, O_wait, Q_rem, Q_done 갱신
    """
    m = s   # 직접 수정

    # ── BUSY → 생산 완료 체크 ──────────
    if m.phase == MachinePhase.BUSY:
        new_eta = max(0.0, m.eta - tau)
        m.eta = new_eta
        if m.eta == 0.0:
            # 생산 완료
            q = m.j_mach
            job = job_data.get(q) if q is not None else None
            m.K_mach = frozenset()
            m.u_short = 0.0
            m.u_long  = 0.0
            # Q_rem에서 제거, Q_done에 추가 (BLOCKED 진입 시점에 job 소비 완료)
            if q in m.Q_rem:
                m.Q_rem  = m.Q_rem  - {q}
                m.Q_done = m.Q_done | {q}
            if job is not None and job.generates_output and job.output_wip_id is not None:
                m.phase = MachinePhase.BLOCKED
                m.O_wait = m.O_wait | {job.output_wip_id}
            else:
                m.phase = MachinePhase.EMPTY
                m.j_mach = None
        return   # BUSY 상태에서는 아래 로직 실행 안 함

    # ── PICKING 처리 ─────────────────
    if crane.type == CRANE_PICKING:
        k    = crane.wip_id
        q    = crane.job_id
        wip  = wip_data[k]

        if m.phase == MachinePhase.EMPTY:
            # case 1: 빈 설비에 첫 PICKING → LOADING으로 전환
            m.K_mach  = frozenset([k])
            m.j_mach  = q
            m.u_short = wip.short_side
            m.u_long  = wip.long_side
            m.phase   = MachinePhase.LOADING

        elif m.phase == MachinePhase.LOADING:
            # case 2: LOADING 중 추가 PICKING
            m.K_mach  = m.K_mach | {k}
            m.u_short = m.u_short + wip.short_side
            m.u_long  = max(m.u_long, wip.long_side)
            # j_mach 유지 (Section 7.5: q == j_mach 보장됨)

        return

    # ── START_PROCESS 처리 ────────────
    if prod.type == PROD_START and m.phase == MachinePhase.LOADING:
        q = prod.job_id
        job = job_data[q]
        m.phase = MachinePhase.BUSY
        ptime = job.process_time   # p_{q_t^mach}
        # 확률적 생산시간: ω_{t+1}^ptime ~ N(0, σ) (SDAM Section 3.2)
        if _stochastic_mode and SIGMA_PTIME > 0.0:
            noise = np.random.normal(0.0, SIGMA_PTIME)
            ptime = max(DELTA_MIN, ptime + noise)
        m.eta = ptime
        return

    # ── DIRECT_START 처리 (원자재 job: EMPTY → BUSY 직접 전이) ──
    # 크레인 PICKING 없이 바로 가공 시작. K_mach는 비워두고 cap 값으로 설정.
    if prod.type == PROD_DIRECT_START and m.phase == MachinePhase.EMPTY:
        q   = prod.job_id
        job = job_data[q]
        m.phase   = MachinePhase.BUSY
        m.j_mach  = q
        m.K_mach  = frozenset()        # 물리적 WIP 추적 없음 (원자재)
        m.u_short = job.cap_short      # 배치 용량 전체 사용으로 간주
        m.u_long  = job.cap_long
        ptime = job.process_time
        if _stochastic_mode and SIGMA_PTIME > 0.0:
            noise = np.random.normal(0.0, SIGMA_PTIME)
            ptime = max(DELTA_MIN, ptime + noise)
        m.eta = ptime
        return

    # ── STORE 처리 ──────────────────────────────
    if crane.type == CRANE_STORE and m.phase == MachinePhase.BLOCKED:
        k = crane.wip_id
        m.O_wait = m.O_wait - {k}
        if len(m.O_wait) == 0:
            # 모든 출력재 적재 완료 → EMPTY로 전환
            m.phase  = MachinePhase.EMPTY
            m.j_mach = None
        return   # BLOCKED 상태의 STORE는 여기서 끝

    # ── otherwise: 상태 유지 (LOADING 유지·BLOCKED 대기) ──────


# ─────────────────────────────────────────────────────────────────
# 내부: 야드 상태 전이
# ─────────────────────────────────────────────────────────────────

def _update_yard(s: State, crane: CraneAction) -> None:
    """
    stacks 딕셔너리 갱신
    PICKING: 스택에서 WIP 제거
    STORE: 스택에 WIP 추가 (출력재)
    MOVE, TEMP_MOVE, RESTORE
    """
    if crane.type == CRANE_PICKING:
        sid = crane.src_stack
        if sid is None:
            # Phase 2: 버퍼 WIP 직접 PICKING
            if crane.wip_id in s.buffer_wips:
                s.buffer_wips = s.buffer_wips - {crane.wip_id}
                s.buffer_cap += 1
        else:
            stk = s.stacks.get(sid, [])
            if stk and stk[-1] == crane.wip_id:
                s.stacks[sid] = stk[:-1]   # 최상단 제거

    elif crane.type == CRANE_STORE:
        # Phase 4: output_wip_id > 0인 실제 출력재는 해당 스택에 적재.
        # wid <= 0은 구버전 토큰 방식 (Phase 4에서는 미사용).
        wid = crane.wip_id
        if wid is not None and wid > 0:
            sid = crane.dst_stack
            stk = s.stacks.get(sid, [])
            s.stacks[sid] = stk + [wid]

    elif crane.type == CRANE_MOVE:
        src_stk = s.stacks.get(crane.src_stack, [])
        if src_stk and src_stk[-1] == crane.wip_id:
            s.stacks[crane.src_stack] = src_stk[:-1]
            dst_stk = s.stacks.get(crane.dst_stack, [])
            s.stacks[crane.dst_stack] = dst_stk + [crane.wip_id]

    elif crane.type == CRANE_TEMP_MOVE:
        sid = crane.src_stack
        stk = s.stacks.get(sid, [])
        if stk and stk[-1] == crane.wip_id:
            s.stacks[sid] = stk[:-1]
            s.buffer_wips = s.buffer_wips | {crane.wip_id}
            s.buffer_cap  = s.buffer_cap - 1

    elif crane.type == CRANE_RESTORE:
        sid = crane.dst_stack
        stk = s.stacks.get(sid, [])
        s.stacks[sid]  = stk + [crane.wip_id]
        s.buffer_wips  = s.buffer_wips - {crane.wip_id}
        s.buffer_cap   = s.buffer_cap + 1

    elif crane.type == CRANE_PRE_POSITION:
        # Phase 3: RESTORE와 동일한 물리적 효과 (버퍼 → 야드 스택 top)
        # 전략적 차이는 feasibility/greedy에서 선택 로직으로 처리
        sid = crane.dst_stack
        stk = s.stacks.get(sid, [])
        s.stacks[sid]  = stk + [crane.wip_id]
        s.buffer_wips  = s.buffer_wips - {crane.wip_id}
        s.buffer_cap   = s.buffer_cap + 1
