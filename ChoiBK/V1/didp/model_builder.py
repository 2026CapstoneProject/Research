"""
V5 Phase5 DIDPPy 모델 빌더
V5_SDAM.md Section 11 + V5_Coding_Plan.md Section 6 참조

Phase 5 변경사항:
  - 원자재 job (input_wip_id == 0): PICKING 전이 미생성 (DIRECT_START는 DyPDL 외 greedy 처리)
  - 출력재 WIP (is_output_wip == True): PICKING 전이 미생성 (후속 런 재사용 불가)
  - 남은 run이 모두 원자재이면 build_didp_model() = None → greedy fallback 유도
  - output_wip_id는 STORE 후 accessible에 추가되나 PICKING 대상에서 제외됨

Phase 2/3 변경사항 (Phase 1 대비):
  - buf_cap int_var 추가 (잔여 버퍼 용량)
  - MOVE(wi) 전이 추가 (BUSY 중 영구 재배치, cost = C_REL)
  - TEMP_MOVE(wi) 전이 추가 (BUSY 중 버퍼 임시이동, cost = C_TEMP, buf_cap 감소)
  - PICKING 전이: 버퍼 내 WIP도 accessible로 포함 (초기 accessible에 buffer_wips 합산)
  - same_spec 기반 PICKING 제약은 feasibility.py에서 처리, DyPDL 모델은 단순화 유지

DyPDL에서 MOVE/TEMP_MOVE의 핵심 효과:
  MOVE(wi)      : accessible.add(next_table[wi])   — wi 아래 WIP 노출
                  cost += C_REL
  TEMP_MOVE(wi) : accessible.add(next_table[wi])   — wi 아래 WIP 노출
                  buf_cap -= 1
                  cost += C_TEMP
  PRE_POS(wi)   : buffered에서 wi 제거, buf_cap += 1
                  cost -= C_PRE_BONUS  (RESTORE보다 우선 선택 유도)
  (wi 자체는 이동 후에도 accessible 유지 — 새 위치 또는 버퍼에서 여전히 접근 가능)
"""

from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Tuple
import math
import os
import sys

from data.loader import WIPData, JobData
from data.params import (
    DELTA_MIN, R_FILL, W_SHORT, W_LONG, P_RUN, P_MACH, P_BUFFER,
    STACK_TO_NODE, MACHINE_NODE, C_IDLE_WAIT, C_REL, C_TEMP, C_PRE_BONUS,
)
from env.state import State, MachinePhase
from env.actions import Action, CraneAction, ProdAction
from env.actions import (
    CRANE_PICKING, CRANE_STORE, CRANE_WAIT, CRANE_RESTORE, CRANE_PRE_POSITION,
    PROD_START, PROD_CONTINUE, PROD_NONE,
)

try:
    import didppy as dp
    DIDP_AVAILABLE = True
except ImportError:
    DIDP_AVAILABLE = False


# Phase 상수 (DyPDL integer encoding)
PHASE_EMPTY   = 0
PHASE_LOADING = 1
PHASE_BUSY    = 2
PHASE_BLOCKED = 3


@contextmanager
def _suppress_stderr():
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError):
        yield
        return

    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


def build_didp_model(
    state:         State,
    wip_data:      Dict[int, WIPData],
    job_data:      Dict[int, JobData],
    machine_times: Dict[str, float],
    horizon:       int = 8,
) -> Optional[object]:
    """
    현재 상태 S_t를 DIDPPy Model로 변환한다.

    Phase 4 추가 상태변수 (Phase 2/3에서 도입):
      buf_cap : int_var — 잔여 버퍼 용량

    Phase 4 추가 전이 (Phase 2/3에서 도입):
      MOVE_{wi}      : BUSY 중 wi를 영구 이동 → accessible에 next_table[wi] 추가
      TEMP_MOVE_{wi} : BUSY 중 wi를 버퍼 임시이동 → accessible에 next_table[wi] 추가, buf_cap 감소
    """
    if not DIDP_AVAILABLE:
        return None

    # ── 인덱스 매핑 ──────────────────────────────────────────
    active_job_ids: List[int] = sorted(state.Q_rem)
    n_runs = len(active_job_ids)
    if n_runs == 0:
        return None

    # Phase 5: 남은 run이 모두 원자재(has_external_input=True)이면
    # DyPDL 모델에 PICKING 전이가 하나도 생성되지 않아 DIRECT_START를 계획할 수 없다.
    # → None 반환으로 rolling_horizon이 greedy fallback을 사용하도록 유도한다.
    if all(job_data[jid].has_external_input for jid in active_job_ids):
        return None

    job_idx: Dict[int, int] = {jid: i for i, jid in enumerate(active_job_ids)}

    top_wip_ids = set(state.accessible_wips().values())

    # 버퍼 WIP도 accessible 초기값에 포함 (Phase 2에서 도입)
    acc_init_ids = top_wip_ids | set(state.buffer_wips)

    # lookahead에 필요한 WIP:
    # - 남은 run의 직접 입력재
    # - 그 입력재 위에 쌓인 blocker들
    # - 현재 top WIP
    # - 설비/버퍼에 이미 있는 WIP
    relevant_wips = compute_relevant_wip_ids(state, job_data, active_job_ids)

    active_wip_ids: List[int] = relevant_wips  # compute_relevant_wip_ids already returns sorted
    n_wips = len(active_wip_ids)
    if n_wips == 0:
        return None

    wip_idx: Dict[int, int] = {wid: i for i, wid in enumerate(active_wip_ids)}

    with _suppress_stderr():
        model = dp.Model(maximize=False, float_cost=True)

        # Object types (sentinel용 +1)
        wip_type = model.add_object_type(number=n_wips + 1)
        run_type = model.add_object_type(number=n_runs)

        # ── 상태변수 등록 ────────────────────────────────────────
        k_mach_init = [wip_idx[w] for w in state.K_mach if w in wip_idx]
        K_mach = model.add_set_var(object_type=wip_type, target=k_mach_init)

        buffered_init = [wip_idx[w] for w in state.buffer_wips if w in wip_idx]
        buffered = model.add_set_var(object_type=wip_type, target=buffered_init)

        q_rem_init = list(range(n_runs))
        Q_rem = model.add_set_var(object_type=run_type, target=q_rem_init)

        # accessible 초기값 = yard top + buffer_wips (Phase 2에서 도입)
        acc_init = [wip_idx[w] for w in acc_init_ids if w in wip_idx]
        accessible = model.add_set_var(object_type=wip_type, target=acc_init)

        phase = model.add_int_var(target=int(state.phase))

        j_mach_init = job_idx.get(state.j_mach, 0) if state.j_mach else 0
        j_mach = model.add_int_var(target=j_mach_init)

        u_short = model.add_float_var(target=float(state.u_short))
        u_long  = model.add_float_var(target=float(state.u_long))

        eta_discrete = model.add_int_var(
            target=max(0, int(math.ceil(state.eta / DELTA_MIN)))
        )

        steps_left = model.add_int_var(target=horizon)

        # 버퍼 잔여 용량 (Phase 2에서 도입)
        buf_cap = model.add_int_var(target=int(state.buffer_cap))

        empty_wips = model.create_set_const(object_type=wip_type, value=[])

        # ── 파라미터 테이블 ──────────────────────────────────────
        s_table = model.add_float_table(
            [wip_data[wid].short_side for wid in active_wip_ids]
        )
        l_table = model.add_float_table(
            [wip_data[wid].long_side for wid in active_wip_ids]
        )
        cap_s_table = model.add_float_table(
            [job_data[jid].cap_short for jid in active_job_ids]
        )
        cap_l_table = model.add_float_table(
            [job_data[jid].cap_long  for jid in active_job_ids]
        )
        ptime_table = model.add_int_table(
            [max(1, int(math.ceil(job_data[jid].process_time / DELTA_MIN)))
             for jid in active_job_ids]
        )

        # next accessible after removing wi
        next_acc = _compute_next_accessible(state, active_wip_ids, wip_idx)
        SENTINEL = n_wips
        next_table = model.add_element_table(
            [next_acc.get(wid, SENTINEL) for wid in active_wip_ids]
        )

        # ── Base case ─────────────────────────────────────────────
        model.add_base_case(
            [steps_left <= 0, phase == PHASE_EMPTY],
            cost=_terminal_expr(Q_rem, K_mach, buffered, phase, n_runs, n_wips)
        )
        model.add_base_case(
            [steps_left <= 0, phase == PHASE_LOADING],
            cost=_terminal_expr(Q_rem, K_mach, buffered, phase, n_runs, n_wips)
            + float(P_MACH)
        )
        model.add_base_case(
            [steps_left <= 0, phase == PHASE_BUSY],
            cost=_terminal_expr(Q_rem, K_mach, buffered, phase, n_runs, n_wips)
            + float(P_MACH)
        )
        model.add_base_case(
            [steps_left <= 0, phase == PHASE_BLOCKED],
            cost=_terminal_expr(Q_rem, K_mach, buffered, phase, n_runs, n_wips)
            + float(P_MACH)
        )
        model.add_base_case(
            [Q_rem.len() == 0, K_mach.len() == 0, buffered.len() == 0, phase == PHASE_EMPTY],
            cost=0.0
        )

        # ── 전이 등록 ──────────────────────────────────────────────
        # 1. WAIT
        _add_wait_transition(
            model, n_runs, phase, eta_discrete, steps_left,
            Q_rem, K_mach, j_mach, empty_wips, active_job_ids, job_data,
        )

        # 2. PICKING(wi, ri)
        for wi, wid in enumerate(active_wip_ids):
            for ri, jid in enumerate(active_job_ids):
                if not _didp_compatible_picking(wid, jid, wip_data, job_data):
                    continue
                _add_picking_transition(
                    model, wi, ri, n_wips, SENTINEL,
                    phase, K_mach, Q_rem, accessible, j_mach, buffered, buf_cap,
                    u_short, u_long, eta_discrete, steps_left,
                    s_table, l_table, cap_s_table, cap_l_table,
                    next_table, wip_type,
                )

        # 3. START_PROCESS(ri)
        for ri in range(n_runs):
            _add_start_transition(
                model, ri,
                phase, K_mach, Q_rem, j_mach,
                u_short, u_long, eta_discrete, steps_left,
                cap_s_table, cap_l_table, ptime_table,
                R_FILL, W_SHORT, W_LONG,
            )

        # 4. STORE (run별 output dependency 반영)
        output_wip_idx = {
            ri: wip_idx[job_data[jid].output_wip_id]
            for ri, jid in enumerate(active_job_ids)
            if job_data[jid].generates_output
            and job_data[jid].output_wip_id in wip_idx
        }
        _add_store_transitions(
            model, n_runs, phase, j_mach, steps_left,
            eta_discrete, empty_wips, u_short, u_long,
            accessible, output_wip_idx,
        )

        # 5. MOVE(wi) — BUSY 중 영구 재배치 (Phase 2에서 도입)
        for wi, wid in enumerate(active_wip_ids):
            _add_move_transition(
                model, wi, phase, K_mach, accessible,
                steps_left, next_table, buffered,
            )

        # 6. TEMP_MOVE(wi) — BUSY 중 버퍼 임시이동 (Phase 2에서 도입)
        for wi, wid in enumerate(active_wip_ids):
            _add_temp_move_transition(
                model, wi, phase, K_mach, accessible,
                steps_left, next_table, buf_cap, buffered,
            )

        # 7. RESTORE(wi) — 버퍼 WIP을 yard로 복원 (방어적, Phase 2에서 도입)
        for wi, wid in enumerate(active_wip_ids):
            _add_restore_transition(
                model, wi, phase, steps_left, buf_cap, buffered,
            )

        # 8. Phase 3: PRE_POSITION(wi) — 버퍼 needed_wip을 전략적 선배치
        # needed_wips_idx: active_wip_ids 중 어떤 run의 input_wip인 것의 인덱스 집합
        needed_wip_ids = {
            job_data[jid].input_wip_id
            for jid in state.Q_rem
            if jid in job_data and job_data[jid].input_wip_id > 0
        }
        for wi, wid in enumerate(active_wip_ids):
            if wid in needed_wip_ids:
                _add_pre_position_transition(
                    model, wi, phase, steps_left, buf_cap, buffered,
                )

    return model


# ─────────────────────────────────────────────────────────────────
# 전이 정의 헬퍼들
# ─────────────────────────────────────────────────────────────────

def _add_wait_transition(
    model, n_runs: int, phase, eta_discrete, steps_left,
    Q_rem, K_mach, j_mach, empty_wips, active_job_ids, job_data,
):
    """WAIT: Phase 1과 동일."""
    t_busy = dp.Transition(
        name="WAIT_BUSY_PROGRESS",
        cost=dp.FloatExpr.state_cost() + 0.0,
        preconditions=[phase == PHASE_BUSY, eta_discrete >= 2],
        effects=[
            (eta_discrete, eta_discrete - 1),
            (steps_left,   steps_left - 1),
        ],
    )
    model.add_transition(t_busy)

    for ri, jid in enumerate(active_job_ids):
        next_phase = (
            PHASE_BLOCKED
            if job_data[jid].generates_output and job_data[jid].output_wip_id is not None
            else PHASE_EMPTY
        )
        t_complete = dp.Transition(
            name=f"WAIT_BUSY_COMPLETE_{ri}",
            cost=dp.FloatExpr.state_cost() + 0.0,
            preconditions=[phase == PHASE_BUSY, eta_discrete == 1, j_mach == ri],
            effects=[
                (phase,        next_phase),
                (eta_discrete, 0),
                (Q_rem,        Q_rem.remove(ri)),
                (K_mach,       empty_wips),
                (steps_left,   steps_left - 1),
            ],
        )
        model.add_transition(t_complete)

    t_wait = dp.Transition(
        name="WAIT",
        cost=dp.FloatExpr.state_cost() + float(C_IDLE_WAIT),
        preconditions=[phase != PHASE_BUSY],
        effects=[(steps_left, steps_left - 1)],
    )
    model.add_transition(t_wait)


def _add_picking_transition(
    model, wi: int, ri: int, n_wips: int, SENTINEL: int,
    phase, K_mach, Q_rem, accessible, j_mach, buffered, buf_cap,
    u_short, u_long, eta_discrete, steps_left,
    s_table, l_table, cap_s_table, cap_l_table,
    next_table, wip_type,
):
    """PICKING(wi, ri): yard top WIP 또는 buffer WIP를 설비에 적재."""
    t_yard = dp.Transition(
        name=f"PICKING_YARD_{wi}_{ri}",
        cost=dp.FloatExpr.state_cost() + 0.0,
        preconditions=[
            (phase == PHASE_EMPTY) | (phase == PHASE_LOADING),
            accessible.contains(wi),
            ~K_mach.contains(wi),
            ~buffered.contains(wi),
            Q_rem.contains(ri),
            (phase == PHASE_EMPTY) | (j_mach == ri),
            u_short + s_table[wi] <= cap_s_table[ri],
            dp.max(u_long, l_table[wi]) <= cap_l_table[ri],
        ],
        effects=[
            (K_mach,     K_mach.add(wi)),
            (accessible, accessible.discard(wi).add(next_table[wi])),
            (u_short,    u_short + s_table[wi]),
            (u_long,     dp.max(u_long, l_table[wi])),
            (j_mach,     ri),
            (phase,      PHASE_LOADING),
            (steps_left, steps_left - 1),
        ],
    )
    model.add_transition(t_yard)

    t_buffer = dp.Transition(
        name=f"PICKING_BUF_{wi}_{ri}",
        cost=dp.FloatExpr.state_cost() + 0.0,
        preconditions=[
            (phase == PHASE_EMPTY) | (phase == PHASE_LOADING),
            accessible.contains(wi),
            ~K_mach.contains(wi),
            buffered.contains(wi),
            Q_rem.contains(ri),
            (phase == PHASE_EMPTY) | (j_mach == ri),
            u_short + s_table[wi] <= cap_s_table[ri],
            dp.max(u_long, l_table[wi]) <= cap_l_table[ri],
        ],
        effects=[
            (K_mach,     K_mach.add(wi)),
            (accessible, accessible.discard(wi).add(next_table[wi])),
            (buffered,   buffered.remove(wi)),
            (buf_cap,    buf_cap + 1),
            (u_short,    u_short + s_table[wi]),
            (u_long,     dp.max(u_long, l_table[wi])),
            (j_mach,     ri),
            (phase,      PHASE_LOADING),
            (steps_left, steps_left - 1),
        ],
    )
    model.add_transition(t_buffer)


def _add_start_transition(
    model, ri: int,
    phase, K_mach, Q_rem, j_mach,
    u_short, u_long, eta_discrete, steps_left,
    cap_s_table, cap_l_table, ptime_table,
    r_fill, w_short, w_long,
):
    """START_PROCESS(ri): Phase 1과 동일."""
    fill_reward = (
        w_short * (u_short / cap_s_table[ri])
        + w_long  * (u_long  / cap_l_table[ri])
    ) * r_fill

    t = dp.Transition(
        name=f"START_{ri}",
        cost=dp.FloatExpr.state_cost() - fill_reward,
        preconditions=[
            phase == PHASE_LOADING,
            K_mach.len() >= 1,
            j_mach == ri,
            Q_rem.contains(ri),
        ],
        effects=[
            (phase,        PHASE_BUSY),
            (eta_discrete, ptime_table[ri]),
            (steps_left,   steps_left - 1),
        ],
    )
    model.add_transition(t)


def _add_store_transitions(
    model, n_runs: int, phase, j_mach, steps_left,
    eta_discrete, empty_wips, u_short, u_long,
    accessible, output_wip_idx: Dict[int, int],
):
    """STORE: BLOCKED → EMPTY. output_wip가 있으면 accessible에 추가."""
    for ri in range(n_runs):
        effects = [
            (phase,        PHASE_EMPTY),
            (eta_discrete, 0),
            (u_short,      0.0),
            (u_long,       0.0),
            (steps_left,   steps_left - 1),
        ]
        if ri in output_wip_idx:
            effects.append((accessible, accessible.add(output_wip_idx[ri])))

        t = dp.Transition(
            name=f"STORE_{ri}",
            cost=dp.FloatExpr.state_cost() + 0.0,
            preconditions=[phase == PHASE_BLOCKED, j_mach == ri],
            effects=effects,
        )
        model.add_transition(t)


def _add_move_transition(
    model, wi: int,
    phase, K_mach, accessible,
    steps_left, next_table, buffered,
):
    """
    MOVE(wi): BUSY 중 wi를 다른 스택으로 영구 이동.

    효과:
      - next_table[wi] 가 accessible에 추가됨 (wi 아래 WIP 노출)
      - wi 자체는 accessible 유지 (새 위치에서 접근 가능)
      - cost += C_REL (영구 재배치 페널티)

    조건:
      - BUSY 상태
      - wi가 accessible (top WIP)
      - K_mach에 없음 (설비 위 WIP 이동 불가)
    """
    t = dp.Transition(
        name=f"MOVE_{wi}",
        cost=dp.FloatExpr.state_cost() + float(C_REL),
        preconditions=[
            phase == PHASE_BUSY,
            accessible.contains(wi),
            ~K_mach.contains(wi),
            ~buffered.contains(wi),
        ],
        effects=[
            # wi 아래 WIP을 accessible에 추가 (wi는 새 위치에서 여전히 accessible)
            (accessible, accessible.add(next_table[wi])),
            (steps_left, steps_left - 1),
        ],
    )
    model.add_transition(t)


def _add_temp_move_transition(
    model, wi: int,
    phase, K_mach, accessible,
    steps_left, next_table, buf_cap, buffered,
):
    """
    TEMP_MOVE(wi): BUSY 중 wi를 버퍼로 임시 이동.

    효과:
      - next_table[wi] 가 accessible에 추가됨 (wi 아래 WIP 노출)
      - wi 자체는 accessible 유지 (버퍼에서 접근 가능)
      - buf_cap -= 1 (버퍼 슬롯 소모)
      - cost += C_TEMP (임시 이동 페널티)

    조건:
      - BUSY 상태
      - wi가 accessible
      - K_mach에 없음
      - buf_cap >= 1 (버퍼 여유)
    """
    t = dp.Transition(
        name=f"TEMP_MOVE_{wi}",
        cost=dp.FloatExpr.state_cost() + float(C_TEMP),
        preconditions=[
            phase == PHASE_BUSY,
            accessible.contains(wi),
            ~K_mach.contains(wi),
            ~buffered.contains(wi),
            buf_cap >= 1,
        ],
        effects=[
            (accessible, accessible.add(next_table[wi])),
            (buffered,   buffered.add(wi)),
            (buf_cap,    buf_cap - 1),
            (steps_left, steps_left - 1),
        ],
    )
    model.add_transition(t)


def _add_restore_transition(
    model, wi: int,
    phase, steps_left, buf_cap, buffered,
):
    """
    RESTORE(wi): 버퍼의 WIP를 yard top으로 복원 (방어적 — 버퍼 공간 확보 목적).
    Phase 2에서는 복원 후에도 wi는 여전히 accessible로 남는 추상화다.
    """
    t = dp.Transition(
        name=f"RESTORE_{wi}",
        cost=dp.FloatExpr.state_cost() + 0.0,
        preconditions=[
            phase == PHASE_BUSY,
            buffered.contains(wi),
        ],
        effects=[
            (buffered,   buffered.remove(wi)),
            (buf_cap,    buf_cap + 1),
            (steps_left, steps_left - 1),
        ],
    )
    model.add_transition(t)


def _add_pre_position_transition(
    model, wi: int,
    phase, steps_left, buf_cap, buffered,
):
    """
    PRE_POSITION(wi): 버퍼의 needed_wip을 미래 PICKING 최적 위치로 선배치 (Phase 3 신규).

    RESTORE와 DyPDL 효과는 동일 (buffered 제거, buf_cap+1, accessible 추가).
    단, 전제조건이 다르다:
      - wi는 buffered에 있어야 함 (RESTORE와 동일)
      - Phase는 BUSY (RESTORE와 동일)
    전략적 선택(어느 스택에 놓을지)은 feasibility/greedy에서 결정하므로
    DyPDL 모델에서는 물리적 효과만 반영한다.

    DyPDL 솔버 관점: RESTORE와 동일한 상태 전이이지만 별도 이름을 부여함으로써
    extract_first_action()이 PRE_POSITION임을 식별하고 전략적 dst_stack을 선택.
    """
    t = dp.Transition(
        name=f"PRE_POS_{wi}",
        cost=dp.FloatExpr.state_cost() - float(C_PRE_BONUS),   # 보상 → RESTORE보다 우선
        preconditions=[
            phase == PHASE_BUSY,
            buffered.contains(wi),
        ],
        effects=[
            (buffered,   buffered.remove(wi)),
            (buf_cap,    buf_cap + 1),
            (steps_left, steps_left - 1),
        ],
    )
    model.add_transition(t)


# ─────────────────────────────────────────────────────────────────
# 보조 함수 (Phase 1과 동일)
# ─────────────────────────────────────────────────────────────────

def _terminal_expr(Q_rem, K_mach, buffered, phase, n_runs, n_wips):
    """
    Horizon 경계 Terminal penalty.
    P_MACH 제외 — procrastination 방지.
    대신 버퍼 WIP는 실제 simulator terminal과 맞추기 위해 패널티를 준다.
    """
    return float(P_RUN) * Q_rem.len() + float(P_BUFFER) * buffered.len()


def compute_relevant_wip_ids(
    state: State,
    job_data: Dict[int, JobData],
    active_job_ids: List[int],
) -> List[int]:
    """
    Phase 2 lookahead에서 추적해야 할 WIP 집합을 계산한다.

    포함 대상:
      - 각 남은 run의 input WIP
      - 그 input WIP 위에 쌓인 blocker들
      - 현재 각 스택의 top WIP
      - 설비/버퍼 내 WIP
    """
    relevant: Set[int] = set(state.K_mach) | set(state.buffer_wips)

    for stack in state.stacks.values():
        if stack:
            relevant.add(stack[-1])

    targets = {
        job_data[jid].input_wip_id
        for jid in active_job_ids
        if job_data[jid].input_wip_id > 0
    }
    for target in targets:
        relevant.add(target)
        for stack in state.stacks.values():
            if target not in stack:
                continue
            pos = stack.index(target)
            relevant.update(stack[pos:])  # target 자신 + 위 blocker들
            break

    return sorted(relevant)


def _didp_compatible_picking(
    wid: int,
    jid: int,
    wip_data: Dict[int, WIPData],
    job_data: Dict[int, JobData],
) -> bool:
    """
    DyPDL 모델에서 PICKING(wid, jid) 전이를 생성할지 결정한다.

    Phase 5:
      - 원자재 job (input_wip_id == 0): DIRECT_START만 허용, PICKING 전이 생성 안 함
      - 출력재 WIP (is_output_wip == True): 후속 런 입력 재사용 불가, PICKING 전이 생성 안 함
      - unique job (input_wip_id > 0): 정확히 해당 WIP ID만 허용
    """
    job = job_data[jid]
    wip = wip_data[wid]
    # 원자재 job: DyPDL PICKING 전이 없음 (DIRECT_START는 별도 처리)
    if job.input_wip_id == 0:
        return False
    # 출력재 WIP: 후속 런의 입력으로 재사용 불가
    if wip.is_output_wip:
        return False
    # unique job: 정확히 해당 WIP만 허용
    return job.input_wip_id == wid


def _matches_job_template(wip: WIPData, job: JobData) -> bool:
    return (
        wip.grade == job.grade
        and abs(wip.thickness - job.thickness) <= 0.1
        and abs(wip.short_side - job.short_side) <= 1.0
        and abs(wip.long_side - job.long_side) <= 1.0
    )


def _compute_next_accessible(
    state: State,
    active_wip_ids: List[int],
    wip_idx: Dict[int, int],
) -> Dict[int, int]:
    """각 WIP 제거 후 같은 스택에서 다음 accessible WIP 인덱스 계산."""
    wip_position: Dict[int, Tuple[int, int]] = {}
    for sid, stack in state.stacks.items():
        for pos, wid in enumerate(stack):
            wip_position[wid] = (sid, pos)

    SENTINEL = len(active_wip_ids)
    result: Dict[int, int] = {}

    for wid in active_wip_ids:
        if wid not in wip_position:
            result[wid] = SENTINEL
            continue
        sid, pos = wip_position[wid]
        stack = state.stacks[sid]
        if pos > 0:
            prev_wid = stack[pos - 1]
            if prev_wid in wip_idx:
                result[wid] = wip_idx[prev_wid]
                continue
        result[wid] = SENTINEL

    return result


def extract_first_action(
    solution,
    state: State,
    wip_data: Dict[int, WIPData],
    job_data:  Dict[int, JobData],
    active_wip_ids: List[int],
    active_job_ids: List[int],
) -> Optional[Action]:
    """DIDPPy solution에서 첫 번째 전이 → Action 변환."""
    if solution is None or len(solution) == 0:
        return None

    first_trans_name = solution[0].name
    return _parse_transition_name(
        first_trans_name, state, wip_data, job_data,
        active_wip_ids, active_job_ids,
    )


def _parse_transition_name(
    name: str,
    state: State,
    wip_data: Dict[int, WIPData],
    job_data:  Dict[int, JobData],
    active_wip_ids: List[int],
    active_job_ids: List[int],
) -> Optional[Action]:
    """전이 이름 문자열 → Action 변환 (Phase 2: MOVE/TEMP_MOVE 추가)"""
    from env.actions import CraneAction, ProdAction, Action, CRANE_MOVE, CRANE_TEMP_MOVE

    if name.startswith("PICKING_YARD_") or name.startswith("PICKING_BUF_"):
        parts = name.split("_")
        from_buffer = parts[1] == "BUF"
        wi, ri = int(parts[2]), int(parts[3])
        wid = active_wip_ids[wi]
        jid = active_job_ids[ri]
        src_stack = None
        if not from_buffer:
            for sid, stk in state.stacks.items():
                if stk and stk[-1] == wid:
                    src_stack = sid
                    break
        return Action(
            crane=CraneAction(CRANE_PICKING, wip_id=wid, src_stack=src_stack, job_id=jid),
            prod=ProdAction(PROD_NONE),
        )

    if name.startswith("START_"):
        ri = int(name.split("_")[1])
        jid = active_job_ids[ri]
        return Action(
            crane=CraneAction(CRANE_WAIT),
            prod=ProdAction(PROD_START, job_id=jid),
        )

    if name.startswith("STORE_"):
        dst = min(state.stacks.keys(), key=lambda s: len(state.stacks[s]))
        o_wait_id = next(iter(state.O_wait), None) if state.O_wait else None
        ri = int(name.split("_")[1])
        jid = active_job_ids[ri]
        return Action(
            crane=CraneAction(CRANE_STORE, wip_id=o_wait_id, dst_stack=dst,
                              job_id=jid),
            prod=ProdAction(PROD_NONE),
        )

    if name.startswith("MOVE_") and not name.startswith("MOVE_B"):
        # MOVE_{wi} — 영구 재배치: 가장 짧은 스택으로 이동
        wi = int(name.split("_")[1])
        wid = active_wip_ids[wi]
        src_stack = None
        for sid, stk in state.stacks.items():
            if stk and stk[-1] == wid:
                src_stack = sid
                break
        # dst: src와 다른 스택 중 가장 짧은 것
        dst = min(
            (sid for sid in state.stacks if sid != src_stack),
            key=lambda s: len(state.stacks[s]),
            default=None,
        )
        return Action(
            crane=CraneAction(CRANE_MOVE, wip_id=wid,
                              src_stack=src_stack, dst_stack=dst),
            prod=ProdAction(PROD_CONTINUE),
        )

    if name.startswith("TEMP_MOVE_"):
        wi = int(name.split("_")[2])
        wid = active_wip_ids[wi]
        src_stack = None
        for sid, stk in state.stacks.items():
            if stk and stk[-1] == wid:
                src_stack = sid
                break
        return Action(
            crane=CraneAction(CRANE_TEMP_MOVE, wip_id=wid, src_stack=src_stack),
            prod=ProdAction(PROD_CONTINUE),
        )

    if name.startswith("RESTORE_"):
        wi = int(name.split("_")[1])
        wid = active_wip_ids[wi]
        # 방어적 복원: 가장 빈 스택으로
        dst = min(state.stacks.keys(), key=lambda s: len(state.stacks[s]))
        return Action(
            crane=CraneAction(CRANE_RESTORE, wip_id=wid, dst_stack=dst),
            prod=ProdAction(PROD_CONTINUE),
        )

    if name.startswith("PRE_POS_"):
        # Phase 3: PRE_POSITION — needed_wip을 전략적 선배치
        wi = int(name.split("_")[2])
        wid = active_wip_ids[wi]
        # 전략 선택: 가장 빈 스택 (최상단 즉시 노출 보장)
        dst = min(state.stacks.keys(), key=lambda s: len(state.stacks[s]))
        return Action(
            crane=CraneAction(CRANE_PRE_POSITION, wip_id=wid, dst_stack=dst),
            prod=ProdAction(PROD_CONTINUE),
        )

    if name == "WAIT" or name.startswith("WAIT_BUSY_"):
        from env.actions import WAIT_NONE, WAIT_CONTINUE
        if state.phase == MachinePhase.BUSY:
            return WAIT_CONTINUE
        return WAIT_NONE

    return None
