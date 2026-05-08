"""
V5 Phase5 파라미터 정의
V5_SDAM.md Section 0.2 / Section 8 참조
"""

# ── 시간 단위 ──────────────────────────────────────────
DELTA_MIN = 1.0          # 최소 시간 단위 (분). τ(WAIT) = DELTA_MIN > 0 (Section 6.3.5)
SHIFT_END = 480.0        # 유인 근무 종료 시각 (분, 8시간)
UNM_DURATION = 120.0     # 무인가공 구간 길이 (분)
UNM_END = SHIFT_END + UNM_DURATION  # = 600.0 분

# ── 설비 용량 (기본값: 데이터에서 run별로 오버라이드됨) ──
DEFAULT_CAP_SHORT = 2500.0   # mm
DEFAULT_CAP_LONG  = 7000.0   # mm

# ── 버퍼 ──────────────────────────────────────────────
BUFFER_CAP = 3           # 버퍼 슬롯 수 (B-1, B-2, B-3)

# ── 비용 가중치 (V5_SDAM Section 8) ──────────────────
C_REL  = 5.0             # 영구 재배치 페널티  (c^rel)
C_TEMP = 2.0             # 임시 이동 페널티    (c^temp)
R_FILL = 10.0            # 적재율 보상 승수    (r^fill)
R_UNM  = 0.05            # 무인가공 보상/분    (r^unm)

W_SHORT = 0.5            # short-side 가중치 (ω^short)
W_LONG  = 0.5            # long-side  가중치 (ω^long)

# ── Terminal penalty ──────────────────────────────────
P_RUN    = 100.0         # 미완료 run당 패널티
P_BUFFER = 5.0           # 버퍼 미복원 WIP당 패널티
P_MACH   = 20.0          # 설비 위 미시작 WIP당 패널티
P_BLOCKER = 10.0         # 필요 WIP 위에 눌린 blocker WIP당 terminal 패널티
C_IDLE_WAIT = 0.1        # RH lookahead에서 유휴 WAIT procrastination 방지용 미세 패널티

# ── Phase 3/4 파라미터 ────────────────────────────────
C_PRE_BONUS = 0.5        # PRE_POSITION DyPDL 보상 (RESTORE 대비 우선 선택 유도)
SIGMA_PTIME = 0.0        # 생산시간 표준편차 (0.0=결정론적, >0=확률적 노이즈)

# ── 노드 이름 매핑 ─────────────────────────────────────
# LocationId(1~4) → 크레인 시간표 노드명
STACK_TO_NODE = {1: "A-1", 2: "A-2", 3: "A-3", 4: "A-4"}
NODE_TO_STACK = {"A-1": 1, "A-2": 2, "A-3": 3, "A-4": 4}

# 버퍼 슬롯 이름
BUFFER_NODES = ["B-1", "B-2", "B-3"]

# 설비 노드 이름 (크레인 시간표의 컬럼명)
MACHINE_NODE = "레이저설비"

# ── Rolling Horizon 설정 ──────────────────────────────
DEFAULT_HORIZON   = 20   # 기본 lookahead 스텝 수 (Phase 1의 긴 job 완료까지 보이도록 확장)
DEFAULT_TIME_LIM  = 3.0  # 솔버 제한 시간 (초)
MAX_SIM_STEPS     = 3000  # 시뮬레이션 최대 스텝 수 (full production_plan 대응)
