"""
C(S_t, x_t, W_{t+1}) =
    c^rel  · Δ_t^rel
  + c^temp · Δ_t^temp
  - r^fill · Δ_t^fill
  - r^unm  · Δ_t^unm

Terminal penalty:
  P_RUN     · |Q_rem|
+ P_BUFFER  · |buffer_wips|
+ P_MACH    · (|K_mach| + |O_wait|)
+ P_BLOCKER · Σ_{run ∈ Q_rem} blocker_count(input_wip)
"""

from typing import Dict, Optional

from data.params import (
    C_REL, C_TEMP, R_FILL, R_UNM,
    W_SHORT, W_LONG,
    P_RUN, P_BUFFER, P_MACH, P_BLOCKER,
    SHIFT_END, UNM_END,
)
from data.loader import RunData
from env.state import State, MachinePhase
from env.actions import Action, CRANE_MOVE, CRANE_TEMP_MOVE, PROD_START, PROD_DIRECT_START


def step_cost(
    state:    State,
    action:   Action,
    run_data: Dict[int, RunData],
    tau:      float,
) -> float:
    """
    단계별 비용 C(S_t, x_t, W_{t+1})
    양수 = 비용, 음수 = 보상
    최소화 목적이므로 보상 항목은 음수로 반환
    """
    crane = action.crane
    prod  = action.prod
    cost  = 0.0

    # ── Δ_t^rel: 영구 재배치 페널티 ───────────
    if crane.type == CRANE_MOVE:
        cost += C_REL

    # ── Δ_t^temp: 임시 이동 페널티 ────────────
    if crane.type == CRANE_TEMP_MOVE:
        cost += C_TEMP

    # ── Δ_t^fill: 설비 적재율 보상 ────────────
    if prod.type == PROD_START and state.q_mach is not None:
        q = state.q_mach
        run = run_data.get(q)
        if run is not None and run.cap_short > 0 and run.cap_long > 0:
            fill_util = (
                W_SHORT * state.u_short / run.cap_short
                + W_LONG  * state.u_long  / run.cap_long
            )
            cost -= R_FILL * fill_util   # 보상 → 음수

    # ── DIRECT_START fill 보상 (원자재 run은 cap 전체 사용으로 간주) ──
    if prod.type == PROD_DIRECT_START:
        run = run_data.get(prod.run_id)
        if run is not None:
            # 원자재는 배치 정원을 채운 것으로 처리 → fill_util = 1.0
            cost -= R_FILL * 1.0   # 보상 → 음수

    # ── Δ_t^unm: 무인가공 구간 overlap 보상 ───
    if state.phase == MachinePhase.BUSY:
        t_start = state.clock
        t_end   = state.clock + tau
        overlap = max(0.0, min(t_end, UNM_END) - max(t_start, SHIFT_END))
        cost -= R_UNM * overlap   # 보상 → 음수

    return cost


def terminal_cost(
    state: State,
    run_data: Optional[Dict[int, RunData]] = None,
) -> float:
    """
    에피소드 종료 시 terminal penalty C_T(S_T)
    - 미완료 run 수
    - 버퍼 미복원 WIP 수
    - 설비 위 미시작 WIP 수
    - 생산 완료 후 미적재 출력재 수
    - blocker WIP 수 (필요 WIP 위에 눌린 WIP, run_data 제공 시)
    """
    penalty = 0.0
    penalty += P_RUN    * len(state.Q_rem)
    penalty += P_BUFFER * len(state.buffer_wips)
    penalty += P_MACH   * len(state.K_mach)
    penalty += P_MACH   * len(state.O_wait)

    # P_BLOCKER: 미완료 run의 input_wip 위에 쌓인 blocker WIP 수
    if run_data is not None:
        for rid in state.Q_rem:
            if rid not in run_data:
                continue
            target_wip = run_data[rid].input_wip_id
            if target_wip <= 0:
                continue
            for stack in state.stacks.values():
                if target_wip not in stack:
                    continue
                pos = stack.index(target_wip)
                # target_wip 위에 있는 WIP 수 = blocker count
                blocker_count = len(stack) - pos - 1
                penalty += P_BLOCKER * blocker_count
                break

    return penalty


def episode_summary(log: list, run_data: Optional[Dict[int, RunData]] = None) -> dict:
    """시뮬레이션 로그로부터 에피소드 요약 통계 계산"""
    total_cost = sum(entry["cost"] for entry in log)
    total_cost += terminal_cost(log[-1]["state_after"], run_data=run_data) if log else 0.0

    n_pickings       = sum(1 for e in log if e["action"].crane.type == "PICKING")
    n_starts      = sum(1 for e in log if e["action"].prod.type  == "START_PROCESS")
    n_stores      = sum(1 for e in log if e["action"].crane.type == "STORE")
    n_moves       = sum(1 for e in log if e["action"].crane.type == "MOVE")
    n_temp_moves  = sum(1 for e in log if e["action"].crane.type == "TEMP_MOVE")
    n_restores    = sum(1 for e in log if e["action"].crane.type == "RESTORE")
    n_pre_pos     = sum(1 for e in log if e["action"].crane.type == "PRE_POSITION")
    # START_PROCESS는 crane=WAIT를 사용하지만, 운영상 "대기"라기보다
    # 생산 시작 이벤트이므로 WAIT 집계에서는 제외한다.
    n_waits  = sum(
        1
        for e in log
        if e["action"].crane.type == "WAIT"
        and e["action"].prod.type != PROD_START
    )

    final_state = log[-1]["state_after"] if log else None
    return {
        "total_cost":      total_cost,
        "n_steps":         len(log),
        "n_pickings":         n_pickings,
        "n_starts":        n_starts,
        "n_stores":        n_stores,
        "n_moves":         n_moves,
        "n_temp_moves":    n_temp_moves,
        "n_restores":      n_restores,
        "n_pre_positions": n_pre_pos,
        "n_waits":         n_waits,
        "runs_done":       len(final_state.Q_done) if final_state else 0,
        "runs_remain":     len(final_state.Q_rem)  if final_state else 0,
        "clock_end":       final_state.clock       if final_state else 0.0,
    }
