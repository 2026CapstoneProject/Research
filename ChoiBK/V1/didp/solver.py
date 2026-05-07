"""
V5 Phase3 DIDPPy 솔버 래퍼
V5_Coding_Plan.md Section 7 참조

CABS (Cost-Algebraic Beam Search): Anytime 알고리즘, 시간 제한 내 최선 해
CAASDy: 빠른 근사 해, 실시간 응답이 필요할 때

DIDPPy 미설치 시 graceful fallback 제공.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import os
import sys
from typing import Any, List, Optional

try:
    import didppy as dp
    DIDP_AVAILABLE = True
except ImportError:
    DIDP_AVAILABLE = False
    dp = None


@dataclass
class SolverResult:
    """DIDPPy 풀이 결과"""
    transitions: List[Any]   # dp.Transition 객체 리스트 (첫 행동만 사용)
    cost:        float
    solver_name: str
    success:     bool = True

    @property
    def first_transition_name(self) -> Optional[str]:
        if self.transitions:
            return self.transitions[0].name
        return None


@contextmanager
def _suppress_stderr():
    """
    DIDPPy 내부 grounding 경고가 stderr로 과도하게 출력되는 것을 억제한다.
    """
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


def solve(
    model: Any,
    time_limit: float = 3.0,
    solver: str = "CABS",
    beam_size: int = 1000,
) -> SolverResult:
    """
    DIDPPy 모델을 풀어 최적 행동 시퀀스를 반환한다.

    Args:
        model:      dp.Model 객체
        time_limit: 솔버 제한 시간 (초)
        solver:     "CABS" | "CAASDy" | "DFBB" | "DBDFS"
        beam_size:  CABS initial_beam_size (DIDPPy 0.10+ 파라미터)

    Returns:
        SolverResult
    """
    if not DIDP_AVAILABLE or model is None:
        return SolverResult(transitions=[], cost=float("inf"),
                            solver_name="NONE", success=False)

    try:
        with _suppress_stderr():
            if solver == "CABS":
                # DIDPPy 0.10+: beam_size → initial_beam_size, quiet=True로 경고 억제
                s = dp.CABS(model, time_limit=time_limit,
                            initial_beam_size=beam_size, quiet=True)
            elif solver == "CAASDy":
                s = dp.CAASDy(model, time_limit=time_limit, quiet=True)
            elif solver == "DFBB":
                s = dp.DFBB(model, time_limit=time_limit, quiet=True)
            elif solver == "DBDFS":
                s = dp.DBDFS(model, time_limit=time_limit, quiet=True)
            else:
                s = dp.CABS(model, time_limit=time_limit,
                            initial_beam_size=beam_size, quiet=True)

            # DIDPPy 0.10+: search()가 Solution 객체 반환 (튜플 언팩 X)
            sol = s.search()

        if sol.cost is None or sol.is_infeasible:
            return SolverResult(transitions=[], cost=float("inf"),
                                solver_name=solver, success=False)

        return SolverResult(
            transitions=sol.transitions,
            cost=float(sol.cost),
            solver_name=solver,
            success=True,
        )

    except Exception as e:
        print(f"[DIDPPy solver error] {e}")
        return SolverResult(transitions=[], cost=float("inf"),
                            solver_name=solver, success=False)


def is_available() -> bool:
    """DIDPPy 사용 가능 여부"""
    return DIDP_AVAILABLE
