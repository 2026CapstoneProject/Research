"""
Greedy 정책 (DIDPPy 미설치 시 fallback)

우선순위:
  1. BLOCKED → 바로 STORE
  2. LOADING → batch가 꽉 찼으면 START_PROCESS
  3. 접근 가능한 WIP이 있고 남은 run에 맞으면 PICKING
  4. LOADING 상태이면 START_PROCESS (이미 최소 1개 로드됨)
  5. BUSY 상태:
     5a. 필요 WIP을 가로막는 blocker가 있으면 TEMP_MOVE (버퍼 여유 있을 때)
     5b. blocker가 있으면 MOVE (영구 재배치)
     5c. 버퍼에 다음 run input_wip이 있으면 PRE_POSITION (Phase 3 신규)
         → RESTORE보다 먼저, 전략적 스택에 선배치
     5d. 버퍼에 WIP이 있으면 RESTORE (방어적 버퍼 복원)
     5e. 그 외 → WAIT
  6. 그 외 → WAIT

핵심 아이디어:
  - 출력재가 후속 입력을 열어주지 않으므로 downstream unlock 보너스를 제거한다.
  - 대신 현재 run의 적재율(fill)과 접근성(accessibility)에 더 집중한다.
  - BUSY 중 버퍼에 다음 run의 input_wip이 있으면, RESTORE보다 PRE_POSITION을 먼저 실행한다.
"""

from typing import Dict, List, Optional, Set, Tuple

from data.loader import WIPData, RunData
from env.state import State, MachinePhase
from env.actions import Action, CRANE_PICKING, CRANE_STORE, PROD_START, PROD_DIRECT_START, CRANE_WAIT
from env.actions import CRANE_MOVE, CRANE_TEMP_MOVE, CRANE_RESTORE, CRANE_PRE_POSITION
from env.feasibility import get_feasible_actions


def greedy_policy(
    state:     State,
    wip_data:  Dict[int, WIPData],
    run_data:  Dict[int, RunData],
) -> Action:
    """
    Greedy 정책: feasible 행동 목록에서 우선순위에 따라 하나를 선택한다.
    """
    feasible = get_feasible_actions(state, wip_data, run_data)
    if not feasible:
        from env.actions import WAIT_NONE
        return WAIT_NONE

    phase = state.phase

    # ── 우선순위 1: BLOCKED → STORE ──────────────────────────
    if phase == MachinePhase.BLOCKED:
        stores = [a for a in feasible if a.crane.type == CRANE_STORE]
        if stores:
            return stores[0]

    # ── 우선순위 1.5: EMPTY + 모든 run 완료 + 버퍼 잔류 → cleanup RESTORE ───
    # 주의: Q_rem > 0이면 LOAD를 먼저 해야 하므로 이 조건이 필수.
    #       그렇지 않으면 PICKING 가능한 상황에서도 불필요한 RESTORE를 먼저 실행해 버림.
    if phase == MachinePhase.EMPTY and len(state.buffer_wips) > 0 and len(state.Q_rem) == 0:
        restore = _best_restore_action(state, run_data, feasible)
        if restore is not None:
            return restore

    # ── 우선순위 2: LOADING + batch 꽉 참 → START_PROCESS ────
    if phase == MachinePhase.LOADING and state.q_mach is not None:
        q = state.q_mach
        run = run_data.get(q)
        if run and len(state.K_mach) >= run.batch_count:
            starts = [a for a in feasible if a.prod.type == PROD_START]
            if starts:
                return starts[0]

    # ── 우선순위 3: PICKING 탐색 ────────────────────────────────
    # EMPTY에서도 "어떤 WIP + 어떤 run 조합으로 시작할지" 점수화한다.
    pickings = [a for a in feasible if a.crane.type == CRANE_PICKING]
    if pickings:
        def picking_score(a: Action) -> float:
            run = run_data.get(a.crane.run_id)
            wip = wip_data.get(a.crane.wip_id)
            if wip is None or run is None:
                return float("-inf")

            short_fill = min(1.0, wip.short_side / max(run.cap_short, 1.0))
            long_fill = min(1.0, wip.long_side / max(run.cap_long, 1.0))
            # Phase 5: PICKING 대상은 unique run (input_wip_id>0)만 해당.
            # 원자재 run (input_wip_id==0)은 DIRECT_START로 처리되므로 PICKING 후보에 없음.
            return 10.0 * short_fill + 6.0 * long_fill
        return max(pickings, key=picking_score)

    # ── 우선순위 4: START_PROCESS ──────────────────────────────
    if phase == MachinePhase.LOADING and len(state.K_mach) >= 1:
        starts = [a for a in feasible if a.prod.type == PROD_START]
        if starts:
            return starts[0]

    # ── 우선순위 4.5: EMPTY + PICKING 없음 → DIRECT_START 또는 idle marshalling ─
    if phase == MachinePhase.EMPTY and not pickings:
        # 4.5a: 원자재 run DIRECT_START (야드 조작 불필요)
        direct_starts = [a for a in feasible if a.prod.type == PROD_DIRECT_START]
        if direct_starts:
            # process_time이 짧은 run 우선
            def ds_score(a: Action) -> float:
                run = run_data.get(a.prod.run_id)
                return run.process_time if run else float("inf")
            return min(direct_starts, key=ds_score)

        # 4.5b: 블로커 제거해 다음 LOAD를 열어준다
        # (무인가공 시간대는 feasibility의 is_unm 가드로 이미 차단됨)
        idle_move = _best_idle_marshalling_action(state, wip_data, run_data, feasible)
        if idle_move is not None:
            return idle_move

    # ── 우선순위 5: BUSY 중 pre-marshalling ──────────────────
    if phase == MachinePhase.BUSY:
        move_action = _best_marshalling_action(state, wip_data, run_data, feasible)
        if move_action is not None:
            return move_action

    # ── 우선순위 6: WAIT ──────────────────────────────────────
    waits = [a for a in feasible if a.crane.type == CRANE_WAIT]
    return waits[0] if waits else feasible[0]


def _best_marshalling_action(
    state:    State,
    wip_data: Dict[int, WIPData],
    run_data: Dict[int, RunData],
    feasible: list,
) -> Optional[Action]:
    """
    pre-marshalling 행동 후보 및 우선순위

    우선순위:
      1. TEMP_MOVE — blocker를 버퍼로 임시 이동 (가장 빠른 차단 해소)
      2. MOVE      — blocker를 다른 스택으로 영구 이동
      3. PRE_POSITION (Phase 3 신규)
                   — 버퍼의 needed_wip을 최적 스택에 선배치
                     (RESTORE보다 먼저: 전략적 위치 선점)
      4. RESTORE   — 버퍼 WIP 범용 복원 (방어적)
    """
    # 다음에 필요한 WIP 집합
    needed_wips: Set[int] = set()
    for rid in state.Q_rem:
        run = run_data.get(rid)
        if run and run.input_wip_id > 0:
            needed_wips.add(run.input_wip_id)

    # ── 1. blocker 탐색 (TEMP_MOVE / MOVE) ───────────────────
    blockers_to_move: Set[int] = set()
    if needed_wips:
        for sid, stack in state.stacks.items():
            for pos in range(len(stack) - 1, -1, -1):
                wid = stack[pos]
                if wid in needed_wips:
                    for above_pos in range(pos + 1, len(stack)):
                        blockers_to_move.add(stack[above_pos])
                    break

    # TEMP_MOVE 우선 (버퍼 여유 있을 때)
    temp_moves = [
        a for a in feasible
        if a.crane.type == CRANE_TEMP_MOVE
        and a.crane.wip_id in blockers_to_move
    ]
    if temp_moves:
        return temp_moves[0]

    # MOVE (영구 재배치)
    moves = [
        a for a in feasible
        if a.crane.type == CRANE_MOVE
        and a.crane.wip_id in blockers_to_move
    ]
    if moves:
        return moves[0]

    # ── 2. PRE_POSITION — 버퍼의 needed_wip 전략 선배치 ──────
    pre_pos = [
        a for a in feasible
        if a.crane.type == CRANE_PRE_POSITION
        and a.crane.wip_id in needed_wips
    ]
    if pre_pos:
        # 가장 빈 스택(wip 수 최소)으로 이동하는 행동 우선
        def pre_score(a: Action) -> int:
            return len(state.stacks.get(a.crane.dst_stack, []))
        return min(pre_pos, key=pre_score)

    # ── 3. RESTORE — 방어적 버퍼 복원 ───────────────────────
    # 버퍼가 꽉 찼거나, 남은 run이 없을 때만 RESTORE를 적극 수행한다.
    # 그렇지 않으면 불필요한 복원으로 future blocker를 다시 만들 수 있어 WAIT이 낫다.
    if state.buffer_cap == 0 or len(state.Q_rem) == 0:
        restore = _best_restore_action(state, run_data, feasible)
        if restore is not None:
            return restore

    return None


def _best_restore_action(
    state: State,
    run_data: Dict[int, RunData],
    feasible: list,
) -> Optional[Action]:
    """
    RESTORE 목적지 선택.

    기본 원칙:
      - 아직 필요한 input WIP가 묻혀 있는 스택 위로는 되도록 복원하지 않는다.
      - 그 외에는 가장 빈 스택으로 보낸다.
    """
    restores = [a for a in feasible if a.crane.type == CRANE_RESTORE]
    if not restores:
        return None

    needed_wips: Set[int] = set()
    for rid in state.Q_rem:
        run = run_data.get(rid)
        if run and run.input_wip_id > 0:
            needed_wips.add(run.input_wip_id)

    blocked_target_stacks: Set[int] = set()
    if needed_wips:
        for sid, stack in state.stacks.items():
            if any(wid in needed_wips for wid in stack):
                blocked_target_stacks.add(sid)

    def restore_score(a: Action) -> Tuple[int, int]:
        dst_sid = a.crane.dst_stack
        penalized = 1 if dst_sid in blocked_target_stacks else 0
        return (penalized, len(state.stacks.get(dst_sid, [])))

    return min(restores, key=restore_score)


def _best_idle_marshalling_action(
    state:    State,
    wip_data: Dict[int, WIPData],
    run_data: Dict[int, RunData],
    feasible: List[Action],
) -> Optional[Action]:
    """
    EMPTY 상태에서 필요 WIP 블로커를 제거하는 최선의 행동을 선택한다.

    우선순위:
      1. TEMP_MOVE — 버퍼로 임시 이동 (버퍼 여유 있을 때, 나중에 RESTORE 가능)
      2. MOVE      — 다른 스택으로 영구 이동 (가장 짧은 스택으로)

    원자재 run (input_wip_id==0)은 야드 WIP을 사용하지 않으므로
    Unique run (input_wip_id > 0)의 needed_wip 위에 쌓인 WIP만 blocker로 탐색한다.

    블로커가 없거나 feasible에 해당 행동이 없으면 None 반환.
    """
    needed_wips: Set[int] = set()
    for rid in state.Q_rem:
        run = run_data.get(rid)
        if run and run.input_wip_id > 0:
            needed_wips.add(run.input_wip_id)

    blockers: Set[int] = set()
    for sid, stack in state.stacks.items():
        for pos in range(len(stack) - 1, -1, -1):
            wid = stack[pos]
            if wid in needed_wips:
                for above_pos in range(pos + 1, len(stack)):
                    blockers.add(stack[above_pos])
                break

    if not blockers:
        return None

    # TEMP_MOVE 우선 (버퍼로 임시 → 비용 C_TEMP, 나중에 복원 가능)
    temp_moves = [
        a for a in feasible
        if a.crane.type == CRANE_TEMP_MOVE and a.crane.wip_id in blockers
    ]
    if temp_moves:
        return temp_moves[0]

    # MOVE (가장 짧은 목적 스택으로)
    moves = [
        a for a in feasible
        if a.crane.type == CRANE_MOVE and a.crane.wip_id in blockers
    ]
    if moves:
        return min(moves, key=lambda a: len(state.stacks.get(a.crane.dst_stack, [])))
