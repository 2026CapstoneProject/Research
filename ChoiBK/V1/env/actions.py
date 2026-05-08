"""

크레인 행동 (x_t^crane):
  PICKING(k, src_stack, job_id)        — WIP k를 스택에서 설비로 올림
  STORE(k, dst_stack, job_id)       — 출력재 k를 설비에서 스택으로 내림
  MOVE(k, src_stack, dst_stack)     — WIP k를 영구 재배치
  TEMP_MOVE(k, src_stack)           — WIP k를 버퍼로 임시 이동
  RESTORE(k, dst_stack)             — 버퍼 WIP k 범용 복원 (방어적)
  PRE_POSITION(k, dst_stack)        — 버퍼 WIP k를 미래 PICKING 최적 위치로 선배치 (공격적, 신규)
                                        조건: k가 Q_rem 중 어떤 run의 input_wip일 때만 허용
                                        전략: WIP 수가 가장 적은 스택(= 최상단 노출 보장) 선택
  WAIT                              — 이번 스텝 크레인 대기

생산 행동 (x_t^prod):
  START_PROCESS(job_id) — 현재 K_mach로 job 시작
  CONTINUE — 기존 생산 유지
  NONE — 생산 관련 이벤트 없음

PRE_POSITION vs RESTORE 차이:
  RESTORE      : 버퍼 → 아무 야드 스택 (버퍼 공간 확보가 주목적)
  PRE_POSITION : 버퍼 → 전략적 스택 (다음 job LOAD를 빠르게 하는 것이 주목적)
                 대상 WIP이 Q_rem의 input_wip인 경우만 생성
                 greedy 우선순위: PRE_POSITION > RESTORE
"""

from dataclasses import dataclass
from typing import Optional


CRANE_PICKING         = "PICKING"
CRANE_STORE        = "STORE"
CRANE_MOVE         = "MOVE"
CRANE_TEMP_MOVE    = "TEMP_MOVE"
CRANE_RESTORE      = "RESTORE"
CRANE_PRE_POSITION = "PRE_POSITION"
CRANE_WAIT         = "WAIT"

PROD_START        = "START_PROCESS"
PROD_DIRECT_START = "DIRECT_START"   # 원자재 job: 야드에 적재 X. 바로 생산 시작한다고 가정
PROD_CONTINUE     = "CONTINUE"
PROD_NONE         = "NONE"


@dataclass(frozen=True)
class CraneAction:
    type:       str
    wip_id:     Optional[int] = None   # 대상 WIP
    src_stack:  Optional[int] = None   # 출발 스택 (PICKING, MOVE, TEMP_MOVE)
    dst_stack:  Optional[int] = None   # 목적 스택 (STORE, MOVE, RESTORE)
    job_id:     Optional[int] = None   # 연결 job (PICKING, STORE)

    def __repr__(self):
        args = []
        if self.wip_id   is not None: args.append(f"wip={self.wip_id}")
        if self.src_stack is not None: args.append(f"src={self.src_stack}")
        if self.dst_stack is not None: args.append(f"dst={self.dst_stack}")
        if self.job_id   is not None: args.append(f"job={self.job_id}")
        return f"{self.type}({', '.join(args)})"


@dataclass(frozen=True)
class ProdAction:
    type:   str
    job_id: Optional[int] = None

    def __repr__(self):
        if self.job_id is not None:
            return f"{self.type}(job={self.job_id})"
        return self.type


@dataclass(frozen=True)
class Action:
    """
    한 에포크의 완전한 행동 = (크레인 행동, 생산 행동)
    x_t = (x_t^prod, x_t^crane)
    """
    crane: CraneAction
    prod:  ProdAction

    def __repr__(self):
        return f"[{self.crane} | {self.prod}]"


WAIT_NONE     = Action(CraneAction(CRANE_WAIT), ProdAction(PROD_NONE))
WAIT_CONTINUE = Action(CraneAction(CRANE_WAIT), ProdAction(PROD_CONTINUE))
