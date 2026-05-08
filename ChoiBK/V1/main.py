"""
가정:
  - 출력 재공품은 다른 생산 run의 입력으로 다시 사용되지 않는다.
  - production_plan의 후속 연결(output -> input)은 제거된 input_data를 사용한다.
  - output WIP은 생성·적재되지만, downstream unlock 보너스는 더 이상 없다.

사용법:
  python main.py                         # greedy + 전체 production_plan 사용
  python main.py --policy rh             # Rolling Horizon DLA (DIDPPy 필요)
  python main.py --horizon 10            # horizon 크기 조정
  python main.py --job-ids 3 35 26       # 특정 job ID 지정
  python main.py --verbose               # 상세 로그
  python main.py --demo-filter           # accessible job 기반 데모 subset 사용
  python main.py --buffer-cap 3          # 버퍼 용량 설정 (기본: 3)
  python main.py --no-move               # MOVE/TEMP_MOVE 비활성화
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List

# 현재 디렉토리를 import 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import load_all, WIPData, JobData
from data.params import DEFAULT_HORIZON, DEFAULT_TIME_LIM, SIGMA_PTIME
from env.state import build_initial_state
from env.transition import set_stochastic
from policy.greedy import greedy_policy
from policy.rolling_horizon import rolling_horizon_policy
from simulation.simulator import run_episode, print_summary
from didp.solver import is_available


# input_data 경로
DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "input_data",
)


def select_accessible_runs(
    wip_data: Dict[int, WIPData],
    job_data:  Dict[int, JobData],
    depth: int = 5,
) -> Dict[int, JobData]:
    """
    Job 후보 선택

    규칙:
      - 각 스택의 상단 depth층 안에 있는 job input WIP는 모두 포함한다.
      - input_wip_id == 0 인 generic run은 top depth 내 WIP 중
        job template과 맞는 것이 하나라도 있으면 후보에 포함한다.
    Returns:
        상단 depth층 기준의 job_data 부분집합
    """
    # 스택별 WIP 리스트 구성 (level 오름차순 = index 0이 바닥)
    stacks: Dict[int, List[int]] = {1: [], 2: [], 3: [], 4: []}
    for wip in wip_data.values():
        if wip.stack_id not in stacks or wip.level <= 0:
            continue
        stacks[wip.stack_id].append((wip.level, wip.wip_id))
    for sid in stacks:
        stacks[sid].sort(key=lambda x: x[0])
        stacks[sid] = [wid for (_, wid) in stacks[sid]]

    selected: Dict[int, JobData] = {}
    wip_to_runs: Dict[int, List[int]] = {}
    generic_jobs: List[JobData] = []
    for jid, job in job_data.items():
        if job.input_wip_id > 0:
            wip_to_runs.setdefault(job.input_wip_id, []).append(jid)
        else:
            generic_jobs.append(job)

    top_depth_wips: List[int] = []

    for sid, stack in stacks.items():
        for i in range(1, depth + 1):
            if len(stack) < i:
                break
            wid = stack[-i]
            top_depth_wips.append(wid)
            for jid in wip_to_runs.get(wid, []):
                selected[jid] = job_data[jid]

    # generic job(input_wip_id == 0): top-depth WIP template별로 대표 run만 선택
    generic_candidates: Dict[tuple, List[JobData]] = {}
    for job in generic_jobs:
        generic_candidates.setdefault(
            (job.grade, job.thickness, job.short_side, job.long_side),
            [],
        ).append(job)

    matched_templates = set()
    for wid in top_depth_wips:
        wip = wip_data.get(wid)
        if wip is None:
            continue
        key = (wip.grade, wip.thickness, wip.short_side, wip.long_side)
        if key not in generic_candidates or key in matched_templates:
            continue
        # 같은 template의 generic run이 여러 개면 process_time이 가장 짧은 대표만 선택
        best_run = min(generic_candidates[key], key=lambda r: r.process_time)
        selected[best_run.job_id] = best_run
        matched_templates.add(key)

    return selected


def parse_args():
    parser = argparse.ArgumentParser(description="" \
    "시뮬레이션")
    parser.add_argument(
        "--policy", choices=["greedy", "rh"], default="greedy",
        help="정책 선택: greedy (기본) | rh (Rolling Horizon DLA)"
    )
    parser.add_argument(
        "--horizon", type=int, default=DEFAULT_HORIZON,
        help=f"Rolling Horizon 스텝 수 (기본: {DEFAULT_HORIZON})"
    )
    parser.add_argument(
        "--time-limit", type=float, default=DEFAULT_TIME_LIM,
        help=f"DIDPPy 솔버 제한 시간(초) (기본: {DEFAULT_TIME_LIM})"
    )
    parser.add_argument(
        "--job-ids", type=int, nargs="+", default=None,
        help="사용할 Job ID 직접 지정 (예: --job-ids 3 35 26)"
    )
    parser.add_argument(
        "--depth", type=int, default=3,
        help="accessible job 탐색 depth (기본: 3층)"
    )
    parser.add_argument(
        "--demo-filter", action="store_true",
        help="accessible job 기반 데모 subset만 사용 (기본값은 전체 production_plan)"
    )
    parser.add_argument(
        "--no-filter", action="store_true",
        help="Deprecated: 전체 production_plan 사용. 현재는 기본 동작과 동일"
    )
    parser.add_argument(
        "--solver", choices=["CABS", "CAASDy", "DFBB"], default="CAASDy",
        help="DIDPPy 솔버 알고리즘 (기본: CAASDy)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="상세 로그 출력"
    )
    parser.add_argument(
        "--buffer-cap", type=int, default=3,
        help="버퍼 용량 슬롯 수 (기본: 3)"
    )
    parser.add_argument(
        "--no-move", action="store_true",
        help="MOVE/TEMP_MOVE 비활성화 (동작 재현용)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=(
            "결과 저장 파일 경로 (.md 또는 .txt). "
            "기본: results/result_<타임스탬프>.md 자동 생성. "
            "예: --output results/run3.md"
        ),
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="결과 파일 저장 비활성화 (기본값: 자동 저장)"
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help=(
            f"확률적 생산시간 활성화: N(0, SIGMA_PTIME) 노이즈 추가 "
            f"(params.py SIGMA_PTIME={SIGMA_PTIME}). "
            "SIGMA_PTIME=0이면 효과 없음."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    #  데이터 로드
    print(f"데이터 로드 중: {os.path.abspath(DATA_DIR)}")
    wip_data, job_data, inter_times, machine_times = load_all(DATA_DIR)
    print(f"  WIP 수: {len(wip_data)} | Job 수 (전체): {len(job_data)}")

    #  Job 선택
    if args.job_ids is not None:
        # 직접 지정
        job_data = {jid: job_data[jid] for jid in args.job_ids
                    if jid in job_data}
        print(f"  지정 Job: {sorted(job_data.keys())}")

    elif args.demo_filter:
        # accessible WIP 기반 자동 선택
        job_data = select_accessible_runs(wip_data, job_data, depth=args.depth)
        print(f"  접근 가능 Job 자동 선택 (depth={args.depth}): "
              f"{sorted(job_data.keys())}")
    else:
        print("  전체 production_plan 사용")

    if not job_data:
        print("사용 가능한 run이 없습니다.")
        return

    #  Job 정보 출력
    print("\n사용할 Job 목록:")
    for jid, job in sorted(job_data.items()):
        wip = wip_data.get(job.input_wip_id)
        loc = f"stack={wip.stack_id} lv={wip.level}" if wip else "N/A"
        print(f"  Job {jid:3d}: WIP {job.input_wip_id:3d} "
              f"[{loc}] | {job.spec:20s} | "
              f"ptime={job.process_time:5.1f}분 | "
              f"C_s={job.cap_short:.0f} C_l={job.cap_long:.0f}")

    #  확률적 생산시간 활성화
    if getattr(args, 'stochastic', False):
        set_stochastic(True)
        print(f"\n확률적 생산시간 활성화 (SIGMA_PTIME={SIGMA_PTIME}분)")
    else:
        set_stochastic(False)

    #  Phase 2: MOVE/TEMP_MOVE 비활성화 옵션 처리
    if hasattr(args, 'no_move') and args.no_move:
        print("\n⚠️  --no-move: MOVE/TEMP_MOVE 비활성화")
        # feasibility 모듈을 monkey-patch하여 marshalling 비활성화
        import env.feasibility as _feas
        _feas._add_marshalling_actions = lambda *a, **k: None

    #  초기 상태 생성
    buf_cap = getattr(args, 'buffer_cap', 3)
    initial_state = build_initial_state(wip_data, job_data, buffer_cap=buf_cap)
    print(f"  버퍼 용량: {buf_cap} 슬롯")

    print(f"\n초기 야드 상태 (Job 관련 스택 top):")
    needed_stacks = set()
    for job in job_data.values():
        wip = wip_data.get(job.input_wip_id)
        if wip:
            needed_stacks.add(wip.stack_id)

    for sid in sorted(initial_state.stacks.keys()):
        stk = initial_state.stacks[sid]
        top = stk[-1] if stk else "empty"
        marker = "◀ job 있음" if sid in needed_stacks else ""
        print(f"  Stack {sid}: {len(stk)}개 WIP | top={top} {marker}")

    # 정책 설정
    if args.policy == "rh":
        if not is_available():
            print(f"\n⚠️  현재 인터프리터({sys.executable})에서 DIDPPy를 찾지 못했습니다.")
            print("   didppy가 설치된 python 환경으로 실행하거나, 현재 인터프리터에 설치해 주세요.")
            print("   greedy 정책으로 전환합니다.")
            args.policy = "greedy"
        else:
            print(f"\nRolling Horizon DLA 정책 "
                  f"(horizon={args.horizon}, solver={args.solver})")

    if args.policy == "greedy":
        print("\nGreedy 정책 사용")

        def policy_fn(state, wip_data, job_data, machine_times):
            return greedy_policy(state, wip_data, job_data), 0.0

    else:  # rh
        h, tl, sn = args.horizon, args.time_limit, args.solver

        def policy_fn(state, wip_data, job_data, machine_times):
            return rolling_horizon_policy(
                state, wip_data, job_data, machine_times,
                horizon=h, time_limit=tl, solver_name=sn,
                verbose=args.verbose,
            )

    #  출력 파일 경로 결정
    output_path = None
    if not getattr(args, "no_save", False):
        if getattr(args, "output", None):
            output_path = args.output
        else:
            # results/ 폴더에 타임스탬프 + 정책명으로 자동 생성
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
            os.makedirs(results_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            policy_tag = args.policy
            run_tag = (
                "jobs" + "_".join(str(r) for r in sorted(job_data.keys()))
                if len(job_data) <= 6
                else f"{len(job_data)}jobs"
            )
            output_path = os.path.join(results_dir, f"result_{ts}_{policy_tag}_{run_tag}.md")

    # 시뮬레이션 실행
    print("\n" + "" * 60)
    log = run_episode(
        initial_state = initial_state,
        wip_data      = wip_data,
        job_data      = job_data,
        inter_times   = inter_times,
        machine_times = machine_times,
        policy        = policy_fn,
        verbose       = True,
        output_path   = output_path,
    )

    # 결과 출력
    print_summary(log, job_data=job_data)


if __name__ == "__main__":
    main()
