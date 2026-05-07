"""
V5 Phase5 상태 클래스
V5_SDAM.md Section 3 참조: S_t = (R_t, I_t, B_t)
"""

import copy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, FrozenSet, List, Optional


class MachinePhase(IntEnum):
    """
    설비 phase
    EMPTY   → LOADING (첫 PICKING 발생)
    LOADING → BUSY    (START_PROCESS 호출)
    BUSY    → BLOCKED (생산 완료, η_t ≤ τ)
    BLOCKED → EMPTY   (모든 출력재 STORE 완료)
    """
    EMPTY   = 0
    LOADING = 1
    BUSY    = 2
    BLOCKED = 3


@dataclass
class State:
    """
    전체 상태 S_t

    ─ R_t^yard ──────────────────────────────────────────
    stacks     : {stack_id → [wip_id_bottom … wip_id_top]}
                 level 1이 바닥, index -1이 최상단 (LIFO)
    crane_loc  : 크레인 현재 위치 노드명 (예: "A-1")

    ─ R_t^buffer ────────────────────────────────────────
    buffer_wips: 버퍼에 있는 WIP ID 집합
    buffer_cap : 잔여 버퍼 용량

    ─ R_t^machine ───────────────────────────────────────
    phase      : MachinePhase
    K_mach     : 현재 설비 위 WIP ID 집합 (FrozenSet)
    q_mach     : 현재 목표 run ID (None if EMPTY)
    u_short    : Σ s_k (파생 — K_mach에서 계산)
    u_long     : max l_k (파생 — K_mach에서 계산)
    eta        : 잔여 가공시간 (분)
    O_wait     : 완료 후 미적재 출력재 ID 집합

    ─ I_t^time ──────────────────────────────────────────
    clock      : wall-clock 시간 (분)

    ─ I_t^jobs ──────────────────────────────────────────
    Q_rem      : 미시작 run ID 집합
    Q_done     : 완료된 run ID 집합
    """

    # ── 야드 상태 ──────────────────────────────────────
    stacks:     Dict[int, List[int]]   # {stack_id → [wip_id, ...]}
    crane_loc:  str                    # 현재 크레인 위치 노드명

    # ── 버퍼 상태 ──────────────────────────────────────
    buffer_wips: FrozenSet[int]
    buffer_cap:  int

    # ── 설비 상태 ──────────────────────────────────────
    phase:    MachinePhase
    K_mach:   FrozenSet[int]
    q_mach:   Optional[int]
    u_short:  float
    u_long:   float
    eta:      float
    O_wait:   FrozenSet[int]

    # ── 시간 상태 ──────────────────────────────────────
    clock:    float                    # 누적 시간 (분)

    # ── job 상태 ───────────────────────────────────────
    Q_rem:    FrozenSet[int]
    Q_done:   FrozenSet[int]

    # ── 에피소드 카운터 ─────────────────────────────────
    step:     int = 0

    # ─────────────────────────────────────────────────
    # 편의 메서드
    # ─────────────────────────────────────────────────

    def top_wip_of(self, stack_id: int) -> Optional[int]:
        """stack_id 스택의 최상단 WIP ID를 반환 (없으면 None)"""
        stk = self.stacks.get(stack_id, [])
        return stk[-1] if stk else None

    def accessible_wips(self) -> Dict[int, int]:
        """
        현재 접근 가능한 WIP: {stack_id → top_wip_id}
        (버퍼 내 WIP 제외 — Phase 1에서는 yard top만 고려)
        """
        result = {}
        for sid, stk in self.stacks.items():
            if stk:
                result[sid] = stk[-1]
        return result

    def is_unm(self, shift_end: float) -> bool:
        """무인가공 시간대 여부 (반복 교대 사이클 지원).

        교대 사이클 = UNM_END (= SHIFT_END + UNM_DURATION = 600분).
        clock을 사이클로 나눈 나머지가 shift_end 이상이면 무인가공 구간.
        예) clock=3149.5 → 3149.5 % 600 = 149.5 < 480 → 유인 근무 시간.
        """
        from data.params import UNM_END
        return (self.clock % UNM_END) >= shift_end

    def rem_shift(self, shift_end: float) -> float:
        """현재 shift 종료까지 남은 시간 (분). 사이클 내 경과시간 기준."""
        from data.params import UNM_END
        clock_in_cycle = self.clock % UNM_END
        return max(0.0, shift_end - clock_in_cycle)

    def is_terminal(self, max_steps: int = 0) -> bool:
        """에피소드 종료 조건: 모든 run 완료 + 미적재 출력재/버퍼 WIP 없음

        max_steps: 0이면 step 상한 없음 (simulator가 range(MAX_SIM_STEPS)로 제어).
                   양수이면 s.step >= max_steps 시 강제 종료.
        """
        all_done = (
            len(self.Q_rem) == 0
            and len(self.O_wait) == 0
            and len(self.buffer_wips) == 0
        )
        if max_steps > 0:
            return all_done or (self.step >= max_steps)
        return all_done

    def copy(self) -> "State":
        return copy.deepcopy(self)

    def summary(self) -> str:
        """현재 상태 간략 요약 문자열"""
        lines = [
            f"Step {self.step:3d} | clock={self.clock:6.1f}min | "
            f"phase={self.phase.name}",
            f"  Q_rem={sorted(self.Q_rem)} | Q_done={sorted(self.Q_done)}",
            f"  K_mach={sorted(self.K_mach)} q={self.q_mach} "
            f"u_s={self.u_short:.0f} u_l={self.u_long:.0f} η={self.eta:.1f}",
            f"  O_wait={sorted(self.O_wait)}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# 초기 상태 생성 헬퍼
# ─────────────────────────────────────────────────────────────────

def build_initial_state(
    wip_data: dict,
    run_data:  dict,
    buffer_cap: int = 3,
    initial_crane_loc: str = "A-1",
) -> State:
    """
    wip_data, run_data로부터 초기 State S_0 를 생성한다.

    - stacks: inventory 위치 정보를 stack별 LIFO 리스트로 변환
      (level이 낮은 순서 = 바닥부터 쌓임)
    - 설비 EMPTY, 버퍼 비어있음
    """
    from data.loader import WIPData

    # stack 구성: level 오름차순 정렬 → index 0=바닥, -1=최상단
    stacks: Dict[int, List[int]] = {1: [], 2: [], 3: [], 4: []}
    for wip in wip_data.values():
        if wip.stack_id not in stacks or wip.level <= 0:
            continue
        stacks[wip.stack_id].append((wip.level, wip.wip_id))

    for sid in stacks:
        stacks[sid].sort(key=lambda x: x[0])      # level 오름차순 정렬
        stacks[sid] = [wid for (_, wid) in stacks[sid]]  # wip_id만 남김

    return State(
        stacks      = stacks,
        crane_loc   = initial_crane_loc,
        buffer_wips = frozenset(),
        buffer_cap  = buffer_cap,
        phase       = MachinePhase.EMPTY,
        K_mach      = frozenset(),
        q_mach      = None,
        u_short     = 0.0,
        u_long      = 0.0,
        eta         = 0.0,
        O_wait      = frozenset(),
        clock       = 0.0,
        Q_rem       = frozenset(run_data.keys()),
        Q_done      = frozenset(),
        step        = 0,
    )
