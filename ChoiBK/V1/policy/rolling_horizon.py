"""
Rolling Horizon DLA 정책:
  - active_wip_ids에 buffer_wips 포함 (버퍼 WIP도 lookahead 대상)
  - generic job / unique output WIP이 추가된 DIDPPy 모델 사용

매 스텝마다:
  1. 현재 상태 S_t 관찰
  2. DIDPPy로 horizon H 이내 최적 행동 시퀀스 계산
  3. 첫 번째 행동만 실행 (FirstAction principle)
  4. DIDPPy 실패 시 greedy 정책으로 fallback
"""

from typing import Dict, Optional, Tuple

from data.loader import WIPData, JobData
from data.params import DEFAULT_HORIZON, DEFAULT_TIME_LIM, STACK_TO_NODE
from env.state import State, MachinePhase
from env.actions import Action, CRANE_WAIT
from env.feasibility import get_feasible_actions
from didp.model_builder import (
    build_didp_model, extract_first_action, compute_relevant_wip_ids,
    DIDP_AVAILABLE,
)
from didp.solver import solve, is_available
from policy.greedy import greedy_policy


def rolling_horizon_policy(
    state:         State,
    wip_data:      Dict[int, WIPData],
    job_data:      Dict[int, JobData],
    machine_times: Dict[str, float],
    horizon:       int   = DEFAULT_HORIZON,
    time_limit:    float = DEFAULT_TIME_LIM,
    solver_name:   str   = "CABS",
    verbose:       bool  = False,
) -> Tuple[Action, float]:
    """
    Rolling Horizon DLA 정책.

    Returns:
        (selected_action, estimated_lookahead_cost)
    """
    # DIDPPy가 없거나 Q_rem이 비어있으면 greedy fallback
    if not is_available() or len(state.Q_rem) == 0:
        action = greedy_policy(state, wip_data, job_data)
        return action, float("inf")

    #  DIDPPy 모델 빌드 
    active_job_ids = sorted(state.Q_rem)
    # extract_first_action()가 model_builder와 동일한 WIP 인덱스 순서를
    # 사용해야 LOAD_{wi}_{ri} → 실제 WIP ID 매핑이 어긋나지 않는다.
    active_wip_ids = compute_relevant_wip_ids(state, job_data, active_job_ids)

    model = build_didp_model(
        state, wip_data, job_data, machine_times, horizon
    )

    if model is None:
        if verbose:
            print("  [RH] DIDPPy 모델 빌드 실패 → greedy fallback")
        action = greedy_policy(state, wip_data, job_data)
        return action, float("inf")

    #  DIDPPy 풀기 
    result = solve(model, time_limit=time_limit, solver=solver_name)

    if not result.success or not result.transitions:
        if verbose:
            print("  [RH] 풀이 실패 → greedy fallback")
        action = greedy_policy(state, wip_data, job_data)
        return action, float("inf")

    #  첫 번째 전이 → Action 변환 
    active_job_ids_list  = sorted(state.Q_rem)
    active_wip_ids_list  = active_wip_ids

    action = extract_first_action(
        result.transitions,
        state, wip_data, job_data,
        active_wip_ids_list, active_job_ids_list,
    )

    if action is None:
        if verbose:
            print("  [RH] 전이 파싱 실패 → greedy fallback")
        action = greedy_policy(state, wip_data, job_data)
        return action, result.cost

    # 모델과 시뮬레이터의 제약이 미세하게 어긋날 수 있으므로,
    # 최종 실행 전에는 실제 feasible set으로 한 번 더 검증한다.
    feasible = get_feasible_actions(state, wip_data, job_data)
    if action not in feasible:
        if verbose:
            print(f"  [RH] 비실행 가능 행동 감지({action}) → greedy fallback")
        action = greedy_policy(state, wip_data, job_data)
        return action, result.cost

    # RH가 WAIT을 고르더라도, greedy가 더 진전되는 marshalling/load를 제안하면 그쪽을 사용한다.
    greedy_action = greedy_policy(state, wip_data, job_data)
    if action.crane.type == CRANE_WAIT:
        if state.phase == MachinePhase.BUSY and greedy_action.crane.type != CRANE_WAIT:
            if verbose:
                print(f"  [RH] WAIT 대신 진행성 있는 greedy 행동 사용 → {greedy_action}")
            return greedy_action, result.cost
        if state.phase == MachinePhase.EMPTY and greedy_action.crane.type != CRANE_WAIT:
            if verbose:
                print(f"  [RH] EMPTY WAIT 대신 greedy 행동 사용 → {greedy_action}")
            return greedy_action, result.cost

    if verbose:
        print(f"  [RH] {result.solver_name} cost={result.cost:.2f} "
              f"→ {action}")

    return action, result.cost
