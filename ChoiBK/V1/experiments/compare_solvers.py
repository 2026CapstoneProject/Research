"""
V5 Phase3 솔버 비교 실험
experiments/compare_solvers.py

동일한 시나리오에 대해 greedy / CABS / CAASDy 정책을 각각 실행하고
누적 비용, 완료 run 수, 실행시간, 주요 액션 카운트를 비교 테이블로 출력한다.

사용법:
  cd Algorithm/V5/Phase3
  python experiments/compare_solvers.py
  python experiments/compare_solvers.py --run-ids 3 35 26
  python experiments/compare_solvers.py --horizon 15 --time-limit 5.0
  python experiments/compare_solvers.py --output experiments/compare_result.md
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

# Phase3 루트를 import 경로에 추가
_PHASE3_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PHASE3_ROOT)

from data.loader import load_all, WIPData, RunData
from data.params import DEFAULT_HORIZON, DEFAULT_TIME_LIM
from env.state import build_initial_state
from env.cost import episode_summary
from env.transition import set_stochastic
from policy.greedy import greedy_policy
from policy.rolling_horizon import rolling_horizon_policy
from simulation.simulator import run_episode
from didp.solver import is_available


# ── 데이터 경로 ────────────────────────────────────────────────────
DATA_DIR = os.path.join(_PHASE3_ROOT, "..", "..", "V1", "input_data_v2")


def _select_accessible_runs(
    wip_data: Dict[int, WIPData],
    run_data: Dict[int, RunData],
    depth: int = 3,
) -> Dict[int, RunData]:
    """상단 depth층 안에 있는 run input WIP만 포함"""
    stacks: Dict[int, List] = {1: [], 2: [], 3: [], 4: []}
    for wip in wip_data.values():
        stacks[wip.stack_id].append((wip.level, wip.wip_id))
    for sid in stacks:
        stacks[sid].sort(key=lambda x: x[0])
        stacks[sid] = [wid for (_, wid) in stacks[sid]]

    wip_to_run = {r.input_wip_id: rid for rid, r in run_data.items()}
    selected = {}
    for stack in stacks.values():
        for i in range(1, depth + 1):
            if len(stack) < i:
                break
            wid = stack[-i]
            if wid in wip_to_run:
                rid = wip_to_run[wid]
                selected[rid] = run_data[rid]
    return selected


def run_single(
    policy_name: str,
    initial_state,
    wip_data:     Dict[int, WIPData],
    run_data:     Dict[int, RunData],
    inter_times:  Dict,
    machine_times: Dict,
    horizon:      int,
    time_limit:   float,
    solver_name:  str = "CAASDy",
) -> dict:
    """
    단일 정책 실행 → 결과 dict 반환
    """
    if policy_name == "greedy":
        def policy_fn(state, wip_data, run_data, machine_times):
            return greedy_policy(state, wip_data, run_data), 0.0
    elif policy_name in ("CABS", "CAASDy", "DFBB"):
        sn = policy_name
        def policy_fn(state, wip_data, run_data, machine_times):
            return rolling_horizon_policy(
                state, wip_data, run_data, machine_times,
                horizon=horizon, time_limit=time_limit,
                solver_name=sn, verbose=False,
            )
    else:
        raise ValueError(f"알 수 없는 정책: {policy_name}")

    t0 = time.perf_counter()
    log = run_episode(
        initial_state=initial_state,
        wip_data=wip_data,
        run_data=run_data,
        inter_times=inter_times,
        machine_times=machine_times,
        policy=policy_fn,
        verbose=False,
        output_path=None,
    )
    elapsed = time.perf_counter() - t0

    summary = episode_summary(log, run_data=run_data)
    return {
        "policy":          policy_name,
        "total_cost":      summary["total_cost"],
        "runs_done":       summary["runs_done"],
        "runs_remain":     summary["runs_remain"],
        "clock_end":       summary["clock_end"],
        "n_pickings":         summary["n_pickings"],
        "n_starts":        summary["n_starts"],
        "n_moves":         summary["n_moves"],
        "n_temp_moves":    summary["n_temp_moves"],
        "n_restores":      summary["n_restores"],
        "n_pre_positions": summary["n_pre_positions"],
        "n_waits":         summary["n_waits"],
        "n_steps":         summary["n_steps"],
        "elapsed_sec":     elapsed,
    }


def _fmt_table_md(rows: List[dict]) -> str:
    """결과 리스트 → Markdown 테이블 문자열"""
    cols = [
        ("정책",           "policy",          "<"),
        ("총 비용",         "total_cost",      ">"),
        ("완료 run",        "runs_done",       ">"),
        ("미완료 run",      "runs_remain",     ">"),
        ("종료 시각(분)",    "clock_end",       ">"),
        ("PICKING",           "n_pickings",         ">"),
        ("MOVE",           "n_moves",         ">"),
        ("TEMP_MOVE",      "n_temp_moves",    ">"),
        ("RESTORE",        "n_restores",      ">"),
        ("PRE_POS",        "n_pre_positions", ">"),
        ("WAIT",           "n_waits",         ">"),
        ("실행시간(초)",     "elapsed_sec",     ">"),
    ]

    headers  = [h for h, _, _ in cols]
    sep      = [("---" if a == "<" else "---:") for _, _, a in cols]

    def fmt_val(k, v):
        if isinstance(v, float):
            if k == "elapsed_sec":  return f"{v:.2f}"
            if k == "total_cost":   return f"{v:.1f}"
            if k == "clock_end":    return f"{v:.1f}"
        return str(v)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        cells = [fmt_val(key, row[key]) for _, key, _ in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _fmt_table_txt(rows: List[dict]) -> str:
    """결과 리스트 → 텍스트 테이블 문자열"""
    col_specs = [
        ("정책",       "policy",          15),
        ("총비용",     "total_cost",       9),
        ("완료",       "runs_done",        5),
        ("미완료",     "runs_remain",      5),
        ("종료(분)",   "clock_end",        9),
        ("PICKING",      "n_pickings",           5),
        ("MOVE",      "n_moves",           5),
        ("TMOV",      "n_temp_moves",      5),
        ("REST",      "n_restores",        5),
        ("PREP",      "n_pre_positions",   5),
        ("WAIT",      "n_waits",           5),
        ("시간(초)",   "elapsed_sec",       8),
    ]

    def fmt_val(k, v):
        if isinstance(v, float):
            if k == "elapsed_sec":  return f"{v:.2f}"
            if k == "total_cost":   return f"{v:.1f}"
            if k == "clock_end":    return f"{v:.1f}"
        return str(v)

    header = "  ".join(h.ljust(w) for h, _, w in col_specs)
    sep    = "  ".join("-" * w for _, _, w in col_specs)
    lines  = [header, sep]
    for row in rows:
        cells = [fmt_val(key, row[key]).ljust(w) for _, key, w in col_specs]
        lines.append("  ".join(cells))
    return "\n".join(lines)


def save_comparison(
    output_path: str,
    rows: List[dict],
    run_data: Dict[int, RunData],
    horizon: int,
    time_limit: float,
) -> None:
    """비교 결과를 .md 또는 .txt 파일로 저장"""
    is_md = output_path.endswith(".md")
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    if is_md:
        lines += [
            "# 솔버 비교 실험 결과",
            "",
            f"- 실행 시각: {now}",
            f"- Run 목록:  {sorted(run_data.keys())}",
            f"- Horizon:   {horizon}",
            f"- Time limit: {time_limit}초 (RH 정책)",
            "",
            "## 비교 테이블",
            "",
            _fmt_table_md(rows),
            "",
            "## 해석 가이드",
            "",
            "- **총 비용**: 낮을수록 우수 (step cost 합산 + terminal penalty)",
            "- **완료 run**: 많을수록 우수",
            "- **종료 시각**: 낮을수록 우수 (더 빠른 처리)",
            "- **실행시간**: greedy는 O(1), RH는 DIDPPy 솔버 시간 포함",
        ]
    else:
        lines += [
            "솔버 비교 실험 결과",
            "=" * 60,
            f"실행 시각: {now}",
            f"Run 목록:  {sorted(run_data.keys())}",
            f"Horizon={horizon}, Time limit={time_limit}초",
            "=" * 60,
            "",
            _fmt_table_txt(rows),
        ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="V5 Phase3 솔버 비교 실험")
    p.add_argument("--run-ids", type=int, nargs="+", default=None)
    p.add_argument("--depth",   type=int, default=3)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIM)
    p.add_argument(
        "--policies", nargs="+",
        default=["greedy", "CAASDy", "CABS"],
        help="비교할 정책 목록 (greedy / CAASDy / CABS / DFBB)"
    )
    p.add_argument("--stochastic", action="store_true")
    p.add_argument(
        "--output", type=str, default=None,
        help="결과 저장 경로 (.md / .txt). 기본: experiments/compare_<ts>.md"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── 데이터 로드 ──────────────────────────────────────────────
    print(f"데이터 로드 중: {os.path.abspath(DATA_DIR)}")
    wip_data, run_data, inter_times, machine_times = load_all(DATA_DIR)

    if args.run_ids:
        run_data = {rid: run_data[rid] for rid in args.run_ids if rid in run_data}
    else:
        run_data = _select_accessible_runs(wip_data, run_data, depth=args.depth)

    print(f"Run 목록: {sorted(run_data.keys())}")

    if not run_data:
        print("사용 가능한 run이 없습니다.")
        return

    # ── 확률적 모드 ──────────────────────────────────────────────
    if args.stochastic:
        set_stochastic(True)
        print("확률적 생산시간 모드 활성화")

    # ── DIDPPy 가용 여부 확인 ────────────────────────────────────
    didp_ok = is_available()
    if not didp_ok:
        print("⚠️  DIDPPy 미설치 — RH 정책 제외하고 greedy만 실행합니다.")

    policies = []
    for p in args.policies:
        if p != "greedy" and not didp_ok:
            print(f"  skip: {p} (DIDPPy 없음)")
            continue
        policies.append(p)

    # ── 실험 실행 ────────────────────────────────────────────────
    results = []
    for policy_name in policies:
        print(f"\n[{policy_name}] 실행 중...", end=" ", flush=True)
        initial_state = build_initial_state(wip_data, run_data, buffer_cap=3)
        try:
            result = run_single(
                policy_name, initial_state,
                wip_data, run_data, inter_times, machine_times,
                horizon=args.horizon, time_limit=args.time_limit,
            )
            results.append(result)
            print(
                f"완료 | cost={result['total_cost']:.1f} | "
                f"runs={result['runs_done']}/{result['runs_done']+result['runs_remain']} | "
                f"{result['elapsed_sec']:.2f}초"
            )
        except Exception as e:
            print(f"오류: {e}")

    if not results:
        print("실행된 정책이 없습니다.")
        return

    # ── 결과 출력 ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("솔버 비교 결과")
    print("=" * 70)
    print(_fmt_table_txt(results))
    print("=" * 70)

    # ── 최적 정책 추출 ────────────────────────────────────────────
    best = min(results, key=lambda r: r["total_cost"])
    print(f"\n최저 비용 정책: [{best['policy']}]  총비용={best['total_cost']:.1f}")

    # ── 파일 저장 ────────────────────────────────────────────────
    output_path = args.output
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(_PHASE3_ROOT, "experiments")
        output_path = os.path.join(exp_dir, f"compare_{ts}.md")

    save_comparison(output_path, results, run_data, args.horizon, args.time_limit)
    print(f"\n결과 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
