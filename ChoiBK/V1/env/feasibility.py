"""
  - input_wip_id == 0 인 run은 모두 원자재(raw material) job.
    인벤토리 WIP을 LOAD하지 않고 DIRECT_START만 허용.
    동일 규격 원자재가 여러 장 존재할 수 있으므로 인벤토리 체크 없이 항상 실행 가능.
  - generates_output=True 런의 출력재(is_output_wip=True)는 PICKING 후보에서 영구 제외.
  - DIRECT_START는 PICKING 유무와 무관하게 EMPTY 상태에서 항상 후보로 추가.
  - idle/busy marshalling의 generic job 블로커 탐색 제거 (원자재 run은 야드 WIP 불필요).
  Section 7.7  PRE_POSITION 허용 — BUSY 중 버퍼 WIP을 미래 PICKING 최적 위치로 선배치
               조건: 해당 버퍼 WIP이 Q_rem의 어떤 unique run의 input_wip_id인 경우만 생성
               전략: WIP 수 최소 스택을 선택 (최상단 즉시 접근 보장)
               greedy 우선순위: PRE_POSITION > RESTORE
"""

from typing import Dict, List, Set
from data.params import SHIFT_END, STACK_TO_NODE
from data.loader import WIPData, JobData
from env.state import State, MachinePhase
from env.actions import (
    Action, CraneAction, ProdAction,
    CRANE_PICKING, CRANE_STORE, CRANE_MOVE, CRANE_TEMP_MOVE, CRANE_RESTORE,
    CRANE_PRE_POSITION, CRANE_WAIT,
    PROD_START, PROD_DIRECT_START, PROD_CONTINUE, PROD_NONE,
    WAIT_NONE, WAIT_CONTINUE,
)


# ─────────────────────────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────────────────────────

def get_feasible_actions(
    state: State,
    wip_data:  Dict[int, WIPData],
    job_data:  Dict[int, JobData],
) -> List[Action]:
    """
    현재 상태 S_t에서 실행 가능한 행동 목록을 반환한다.
    빈 리스트가 반환되는 일은 없다 — 항상 WAIT이 포함된다.
    행동 생성 규칙:
      - BUSY: MOVE / TEMP_MOVE / RESTORE 허용 (pre-marshalling)
      - EMPTY: PICKING 없을 때 블로커 MOVE / TEMP_MOVE 허용 (idle marshalling)
               단, 무인가공 시간대는 is_unm 가드로 막혀 크레인 정지 유지
      - EMPTY/LOADING: 버퍼 WIP도 PICKING 대상
      - EMPTY에서 generic job(input_wip_id == 0)의 첫 PICKING 후보를 탐색
      - same_spec + job template 체크 정밀화
    """
    phase = state.phase

    #무인가공 시간대 → crane=WAIT만 허용
    if state.is_unm(SHIFT_END):
        if phase == MachinePhase.BUSY:
            return [WAIT_CONTINUE]
        if phase == MachinePhase.LOADING:
            acts: List[Action] = []
            _add_start_process_actions(state, job_data, acts)
            acts.append(WAIT_NONE)
            return acts
        return [WAIT_NONE]

    actions: List[Action] = []

    # ── BLOCKED: STORE만 허용 ──────────────────
    if phase == MachinePhase.BLOCKED:
        _add_store_actions(state, actions)
        actions.append(WAIT_NONE)
        return actions

    # ── BUSY: 재배치 행동 허용 ──────────────────
    # Phase 2: MOVE / TEMP_MOVE / RESTORE 추가
    if phase == MachinePhase.BUSY:
        _add_marshalling_actions(state, wip_data, job_data, actions)
        actions.append(WAIT_CONTINUE)
        return actions

    # ── EMPTY / LOADING: PICKING 또는 START_PROCESS ─────────────
    if phase in (MachinePhase.EMPTY, MachinePhase.LOADING):
        _add_picking_actions(state, wip_data, job_data, actions)
        _add_start_process_actions(state, job_data, actions)
        if phase == MachinePhase.EMPTY:
            _add_cleanup_restore_actions(state, actions)
            # Phase 5: 원자재 job DIRECT_START는 PICKING 유무와 무관하게 항상 추가
            # (원자재는 인벤토리 WIP 없이도 항상 실행 가능, 동일 규격 여러 장 허용)
            _add_direct_start_actions(state, job_data, actions)
            # LOAD도 없고 DIRECT_START도 없으면 블로커 제거 (매몰 WIP 발굴)
            any_productive = any(
                a.crane.type == CRANE_PICKING or a.prod.type == PROD_DIRECT_START
                for a in actions
            )
            if not any_productive:
                _add_idle_marshalling_actions(state, wip_data, job_data, actions)

    # WAIT은 항상 추가
    if phase == MachinePhase.BUSY:
        actions.append(WAIT_CONTINUE)
    else:
        actions.append(WAIT_NONE)

    return actions if actions else [WAIT_NONE]


# ─────────────────────────────────────────────────────────────────
# 재배치 행동 (BUSY 중 pre-marshalling)
# ─────────────────────────────────────────────────────────────────

def _add_marshalling_actions(
    state:    State,
    wip_data: Dict[int, WIPData],
    job_data: Dict[int, JobData],
    out:      List[Action],
) -> None:
    """
    BUSY 상태에서의 재배치 행동 후보(Pre-marshalling): MOVE / TEMP_MOVE / PRE_POSITION / RESTORE

    MOVE(k, src, dst)        : WIP k를 src 스택에서 dst 스택으로 영구 이동 (C^rel 비용)
    TEMP_MOVE(k, src)        : WIP k를 버퍼로 임시 이동 (C^temp 비용, buffer_cap >= 1)
    PRE_POSITION(k, dst)     : 버퍼 WIP k를 미래 PICKING 최적 스택으로 선배치 (Phase 3 신규)
                               조건: k가 Q_rem의 어떤 run의 input_wip_id일 때만 허용
                               대상 스택: WIP 수 최소 스택 (최상단 즉시 노출 보장)
    RESTORE(k, dst)          : 버퍼 WIP k를 임의 스택으로 복원 (방어적, 비용 0)

    대상 WIP 조건 (MOVE/TEMP_MOVE):
      - yard stack의 최상단 (top_kt = 1) — 직접 접근 가능
      - K_mach에 없음 (설비 위 WIP은 이동 불가)
    """
    accessible = state.accessible_wips()  # {stack_id → top_wip_id}

    # Q_rem의 input_wip 집합 (PRE_POSITION 대상 판별용)
    needed_wips: Set[int] = {
        job_data[jid].input_wip_id
        for jid in state.Q_rem
        if jid in job_data and job_data[jid].input_wip_id > 0
    }

    # ── MOVE / TEMP_MOVE ──────────────────────────────────────
    for src_sid, wip_id in accessible.items():
        if wip_id in state.K_mach:
            continue

        # MOVE → 다른 스택으로 영구 이동
        for dst_sid in state.stacks.keys():
            if dst_sid == src_sid:
                continue
            out.append(Action(
                crane=CraneAction(
                    type=CRANE_MOVE,
                    wip_id=wip_id,
                    src_stack=src_sid,
                    dst_stack=dst_sid,
                ),
                prod=ProdAction(PROD_CONTINUE),
            ))

        # TEMP_MOVE → 버퍼로 임시 이동 (버퍼 여유 있을 때만)
        if state.buffer_cap >= 1:
            out.append(Action(
                crane=CraneAction(
                    type=CRANE_TEMP_MOVE,
                    wip_id=wip_id,
                    src_stack=src_sid,
                ),
                prod=ProdAction(PROD_CONTINUE),
            ))

    # ── PRE_POSITION ───────────────────────────
    # 버퍼 WIP 중 미래 run의 input_wip인 것만 전략적 선배치
    if needed_wips:
        # 전략 스택: WIP 수가 가장 적은 스택 (최상단 즉시 노출 보장)
        target_stacks = sorted(
            state.stacks.keys(),
            key=lambda sid: len(state.stacks[sid]),
        )
        for wip_id in state.buffer_wips:
            if wip_id not in needed_wips:
                continue
            # 최소 WIP 스택에만 PRE_POSITION 후보 생성 (상위 2개까지)
            for dst_sid in target_stacks[:2]:
                out.append(Action(
                    crane=CraneAction(
                        type=CRANE_PRE_POSITION,
                        wip_id=wip_id,
                        dst_stack=dst_sid,
                    ),
                    prod=ProdAction(PROD_CONTINUE),
                ))

    # ── RESTORE ────────────────────────────────────────────────
    # 버퍼 내 WIP을 yard 스택으로 복원 (방어적 — 버퍼 공간 확보)
    for wip_id in state.buffer_wips:
        for dst_sid in state.stacks.keys():
            out.append(Action(
                crane=CraneAction(
                    type=CRANE_RESTORE,
                    wip_id=wip_id,
                    dst_stack=dst_sid,
                ),
                prod=ProdAction(PROD_CONTINUE),
            ))


def _add_cleanup_restore_actions(
    state: State,
    out: List[Action],
) -> None:
    """
    EMPTY 상태에서 버퍼 잔류 WIP를 야드로 복원하는 cleanup 행동을 추가한다.
    """
    for wip_id in state.buffer_wips:
        for dst_sid in state.stacks.keys():
            out.append(Action(
                crane=CraneAction(
                    type=CRANE_RESTORE,
                    wip_id=wip_id,
                    dst_stack=dst_sid,
                ),
                prod=ProdAction(PROD_NONE),
            ))


def _add_idle_marshalling_actions(
    state:    State,
    wip_data: Dict[int, WIPData],
    job_data: Dict[int, JobData],
    out:      List[Action],
) -> None:
    """

    LOAD도 없고 DIRECT_START도 없을 때만 호출되며, 무인가공 시간대는 is_unm 가드로 차단됨.
    prod는 PROD_NONE (설비 미가동 상태).

    원자재 job (input_wip_id==0)은 야드 WIP을 사용하지 않으므로 블로커 탐색 제외.
    Unique job (input_wip_id > 0)의 needed_wip 위에 쌓인 WIP만 blocker로 간주한다.
    """
    # unique run의 needed_wip 집합
    needed_wips: Set[int] = {
        job_data[jid].input_wip_id
        for jid in state.Q_rem
        if jid in job_data and job_data[jid].input_wip_id > 0
    }
    blockers: Set[int] = set()
    for sid, stack in state.stacks.items():
        for pos in range(len(stack) - 1, -1, -1):
            wid = stack[pos]
            if wid in needed_wips:
                for above_pos in range(pos + 1, len(stack)):
                    blockers.add(stack[above_pos])
                break

    if not blockers:
        return

    accessible = state.accessible_wips()  # {stack_id → top_wip_id}

    for src_sid, wip_id in accessible.items():
        if wip_id not in blockers:
            continue  # blocker가 아닌 WIP은 건드리지 않음

        # TEMP_MOVE → 버퍼로 임시 이동 (버퍼 여유 있을 때만)
        if state.buffer_cap >= 1:
            out.append(Action(
                crane=CraneAction(
                    type=CRANE_TEMP_MOVE,
                    wip_id=wip_id,
                    src_stack=src_sid,
                ),
                prod=ProdAction(PROD_NONE),
            ))

        # MOVE → 다른 스택으로 영구 이동
        for dst_sid in state.stacks.keys():
            if dst_sid == src_sid:
                continue
            out.append(Action(
                crane=CraneAction(
                    type=CRANE_MOVE,
                    wip_id=wip_id,
                    src_stack=src_sid,
                    dst_stack=dst_sid,
                ),
                prod=ProdAction(PROD_NONE),
            ))


# ─────────────────────────────────────────────────────────────────
# 내부: PICKING 후보 생성 (버퍼 WIP 포함, same_spec 강화)
# ─────────────────────────────────────────────────────────────────

def _add_picking_actions(
    state: State,
    wip_data:  Dict[int, WIPData],
    job_data:  Dict[int, JobData],
    out:       List[Action],
) -> None:
    """
    PICKING(k, src_stack, job_id) 후보를 out에 추가한다.

    Phase 4 변경:
      - 야드 top WIP뿐 아니라 버퍼 내 WIP도 PICKING 대상에 포함
      - generic job(input_wip_id == 0)에 대해 첫 PICKING 후보를 탐색
      - same_spec 체크 정밀화 (_compat_p4 사용)
    """
    # ── (1) 야드 top WIP ──────────────────────────────────────
    accessible = state.accessible_wips()  # {stack_id → wip_id}

    for sid, wip_id in accessible.items():
        _try_add_picking(state, wip_data, job_data, out,
                      wip_id=wip_id, src_stack=sid)

    # ── (2) 버퍼 내 WIP ───────────────────
    # 버퍼 WIP은 src_stack=None으로 표시 (transition에서 buffer_wips 제거 처리)
    for wip_id in state.buffer_wips:
        if wip_id in state.K_mach:
            continue
        _try_add_picking(state, wip_data, job_data, out,
                      wip_id=wip_id, src_stack=None)


def _try_add_picking(
    state: State,
    wip_data: Dict[int, WIPData],
    job_data:  Dict[int, JobData],
    out:       List[Action],
    wip_id: int,
    src_stack,
) -> None:
    """단일 WIP에 대해 PICKING 가능한 job 후보를 검색하여 out에 추가한다."""
    wip = wip_data.get(wip_id)
    if wip is None:
        return
    if wip_id in state.K_mach:
        return
    # 출력재(is_output_wip=True)는 후속 런의 입력으로 재사용되지 않는다
    if wip.is_output_wip:
        return

    for job_id, job in job_data.items():
        if job_id not in state.Q_rem:
            continue

        # compat 체크 (first-load 탐색 + same_spec 강화)
        if not _compat_p4(wip_id, job_id, wip, job, state, wip_data):
            continue

        # job 일치 체크 (LOADING 중이면 j_mach와 같아야 함)
        if state.phase == MachinePhase.LOADING:
            if job_id != state.j_mach:
                continue

        # capa 체크 (원자재 run은 capa 체크 없이 항상 PICKING 가능)
        new_u_short = state.u_short + wip.short_side
        new_u_long  = max(state.u_long, wip.long_side)
        if new_u_short > job.cap_short:
            continue
        if new_u_long > job.cap_long:
            continue

        out.append(Action(
            crane=CraneAction(
                type=CRANE_PICKING,
                wip_id=wip_id,
                src_stack=src_stack,
                job_id=job_id,
            ),
            prod=ProdAction(PROD_NONE),
        ))


def _compat_p4(
    wip_id: int,
    job_id: int,
    wip:    WIPData,
    job:    JobData,
    state:  State,
    wip_data: Dict[int, WIPData],
) -> bool:
    """
    EMPTY 상태 (첫 번째 PICKING):
      - job.input_wip_id > 0 이면 해당 unique WIP만 허용
      - job.input_wip_id == 0 이면 job template과 맞는 어떤 WIP든 허용

    LOADING 상태 (추가 PICKING):
      - job 일치 (j_mach == job_id) 는 호출 측에서 보장
      - 기존 K_mach 내 WIP과 same_spec이어야 함
      - 동시에 job template과도 맞아야 함
    """
    if state.phase == MachinePhase.EMPTY:
        # 원자재 job (input_wip_id==0)은 DIRECT_START만 허용, PICKING 불가
        if job.input_wip_id == 0:
            return False
        # unique job: 정확히 해당 WIP ID만 허용
        return job.input_wip_id == wip_id

    # LOADING: 추가 PICKING — K_mach 내 모든 WIP과 same_spec 체크
    if state.phase == MachinePhase.LOADING:
        if not _matches_job_template(wip, job):
            return False
        for existing_wid in state.K_mach:
            existing_wip = wip_data.get(existing_wid)
            if existing_wip is None:
                continue
            if not _same_spec(wip, existing_wip):
                return False
        return True

    return False


def _same_spec(wip1: WIPData, wip2: WIPData) -> bool:
    """
    두 WIP의 규격 동일 여부 (Phase 5 기준)
    동일 grade + 두께(±0.1mm) 만 체크.
    가로/세로(단변·장변)는 설비 Capa 체크(_try_add_picking의 cap_short/cap_long)로 처리.
    """
    if wip1.grade != wip2.grade:
        return False
    if abs(wip1.thickness - wip2.thickness) > 0.1:
        return False
    return True


def _matches_job_template(wip: WIPData, job: JobData) -> bool:
    """
    WIP이 job이 요구하는 규격/재질 template과 일치하는지 검사한다.
    Phase 5: grade + 두께만 체크. 가로/세로는 설비 Capa 체크로 위임.
    """
    if wip.grade != job.grade:
        return False
    if abs(wip.thickness - job.thickness) > 0.1:
        return False
    return True


# ─────────────────────────────────────────────────────────────────
# 내부: START_PROCESS 후보 생성
# ─────────────────────────────────────────────────────────────────

def _add_start_process_actions(
    state:    State,
    job_data: Dict[int, JobData],
    out:      List[Action],
) -> None:
    """
    START_PROCESS(q) 후보를 out에 추가한다.
    조건 (Section 7.6):
      - m_t = LOADING
      - K_mach ≠ ∅
      - q = j_mach
    """
    if (state.phase == MachinePhase.LOADING
            and len(state.K_mach) > 0
            and state.j_mach is not None):
        q = state.j_mach
        out.append(Action(
            crane=CraneAction(CRANE_WAIT),
            prod=ProdAction(PROD_START, job_id=q),
        ))


# ─────────────────────────────────────────────────────────────────
# 내부: STORE 후보 생성
# ─────────────────────────────────────────────────────────────────

def _add_store_actions(
    state: State,
    out:   List[Action],
) -> None:
    """
    STORE(k, dst_stack, job_id) 후보를 out에 추가한다.
    조건:
      - m_t = BLOCKED
      - k ∈ O_wait
    """
    if state.phase != MachinePhase.BLOCKED:
        return

    available_stacks = sorted(
        state.stacks.keys(),
        key=lambda sid: len(state.stacks[sid]),
    )

    for k in state.O_wait:
        dst = available_stacks[0]
        out.append(Action(
            crane=CraneAction(
                type=CRANE_STORE,
                wip_id=k,
                dst_stack=dst,
                job_id=state.j_mach,
            ),
            prod=ProdAction(PROD_NONE),
        ))


# ─────────────────────────────────────────────────────────────────
# 내부: DIRECT_START 후보 생성 (원자재 job 전용)
# ─────────────────────────────────────────────────────────────────

def _add_direct_start_actions(
    state:    State,
    job_data: Dict[int, JobData],
    out:      List[Action],
) -> None:
    """
    EMPTY 상태에서 야드에 PICKING 가능한 WIP이 없을 때,
    has_external_input=True인 원자재 run에 대해 DIRECT_START 후보를 추가한다.

    DIRECT_START: crane=WAIT (이동 없음), prod=DIRECT_START(job_id)
    전이 효과: EMPTY → BUSY (K_mach=∅, u_short=cap_short, u_long=cap_long)
    """
    if state.phase != MachinePhase.EMPTY:
        return
    for job_id, job in job_data.items():
        if job_id not in state.Q_rem:
            continue
        if not job.has_external_input:
            continue
        out.append(Action(
            crane=CraneAction(CRANE_WAIT),
            prod=ProdAction(PROD_DIRECT_START, job_id=job_id),
        ))
