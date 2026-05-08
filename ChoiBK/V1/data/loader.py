"""
데이터 로더

핵심 가정:
  1. input_wip_id == 0 인 run은 모두 원자재(raw material) run이다.
     원자재는 동일 규격으로 여러 장이 있을 수 있으며, 인벤토리 WIP을 크레인으로 LOAD하지 않고
     DIRECT_START로 즉시 가공을 시작한다. has_external_input = True (항상).
  2. output_wip_id는 unique WIP로 생성·적재되지만, 다른 run의 input으로 재사용되지 않는다.
     생성된 출력재는 is_output_wip = True로 마킹되어 PICKING 후보에서 제외된다.
  3. input_wip_id > 0 인 unique job: 초기 inventory에 없는 WIP은 외부 원자재로 본다.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd


@dataclass
class WIPData:
    """인벤토리 또는 생산 후 생성될 WIP 한 장의 정보"""
    wip_id: int
    stack_id: int
    level: int
    short_side: float
    long_side: float
    thickness: float
    grade: str
    spec: str
    is_output_wip: bool = False  # Phase 5: generates_output run의 출력재 → PICKING 후보 제외

    def __repr__(self):
        return (
            f"WIP(id={self.wip_id}, stack={self.stack_id}, lv={self.level}, "
            f"spec={self.spec})"
        )


@dataclass
class JobData:
    """생산 job 한 건의 정보"""
    job_id: int
    input_wip_id: int
    grade: str
    spec: str
    batch_count: int
    process_time: float
    cap_short: float
    cap_long: float
    thickness: float
    short_side: float
    long_side: float
    generates_output: bool
    output_wip_id: Optional[int]
    has_external_input: bool = False  # True = 원자재 job (크레인 PICKING 불필요)

    def __repr__(self):
        return (
            f"Job(id={self.job_id}, in={self.input_wip_id}, out={self.output_wip_id}, "
            f"batch={self.batch_count}, spec={self.spec})"
        )


def _parse_spec(spec_str: str) -> Tuple[float, float, float]:
    parts = str(spec_str).strip().split("*")
    t = float(parts[0])
    w = float(parts[1])
    l = float(parts[2])
    return t, min(w, l), max(w, l)


def _load_inventory(inv_path: str) -> pd.DataFrame:
    df = pd.read_csv(inv_path)
    df.columns = ["wip_id", "zone", "stack_id", "level"]
    return df


def _load_production_plan(plan_path: str) -> pd.DataFrame:
    df = pd.read_csv(plan_path, encoding="cp949")
    df.columns = [
        "job_id", "date", "use_machine", "batch_count",
        "input_wip_id", "grade", "spec", "quantity",
        "est_hours", "process_time_min",
        "generates_output", "output_wip_id",
    ]
    return df


def _load_crane_times(crane_path: str) -> Tuple[Dict, Dict]:
    df = pd.read_csv(crane_path, encoding="cp949", index_col=0)

    inter = {}
    for row_node in df.index:
        for col_node in df.columns:
            val = df.loc[row_node, col_node]
            if val == "-" or pd.isna(val):
                continue
            try:
                inter[(row_node, col_node)] = float(val)
            except ValueError:
                pass

    machine_col = df.columns[-1]
    machine_times = {}
    for row_node in df.index:
        val = df.loc[row_node, machine_col]
        if val != "-" and not pd.isna(val):
            try:
                machine_times[row_node] = float(val)
            except ValueError:
                pass

    return inter, machine_times


def load_all(data_dir: str) -> Tuple[
    Dict[int, WIPData],
    Dict[int, JobData],
    Dict[Tuple[str, str], float],
    Dict[str, float],
]:
    inv_path = os.path.join(data_dir, "01_inventory_data.csv")
    plan_path = os.path.join(data_dir, "02_production_plan.csv")
    crane_path = os.path.join(data_dir, "node_arc_crane_time.csv")

    inv_df = _load_inventory(inv_path)
    plan_df = _load_production_plan(plan_path)
    inter_times, machine_times = _load_crane_times(crane_path)

    # 모든 positive input_wip_id에서 메타데이터 수집
    # Phase 5에서는 output이 다시 input으로 재사용되지 않으므로,
    # output_wip_id 메타데이터는 현재 row의 spec fallback만 있으면 충분하다.
    meta_from_input: Dict[int, Tuple[float, float, float, str, str]] = {}
    meta_from_output_fallback: Dict[int, Tuple[float, float, float, str, str]] = {}

    for _, row in plan_df.iterrows():
        t, s, l = _parse_spec(row["spec"])
        grade = str(row["grade"])
        spec = str(row["spec"])

        in_wid = int(row["input_wip_id"])
        if in_wid > 0:
            meta_from_input[in_wid] = (t, s, l, grade, spec)

        gen_flag = int(row["generates_output"]) if not pd.isna(row["generates_output"]) else 0
        out_val = row["output_wip_id"]
        if gen_flag == 1 and not pd.isna(out_val):
            out_wid = int(out_val)
            meta_from_output_fallback[out_wid] = (t, s, l, grade, spec)

    default_meta = (12.0, 2438.0, 6096.0, "SM355A", "12*2438*6096")

    # 초기 inventory WIP 생성
    wip_data: Dict[int, WIPData] = {}
    for _, row in inv_df.iterrows():
        wid = int(row["wip_id"])
        t, s, l, grade, spec = meta_from_input.get(wid, default_meta)
        wip_data[wid] = WIPData(
            wip_id=wid,
            stack_id=int(row["stack_id"]),
            level=int(row["level"]),
            short_side=s,
            long_side=l,
            thickness=t,
            grade=grade,
            spec=spec,
        )

    # 생산 후 생성될 WIP / 외부 unique input WIP placeholder 추가
    all_known_wids = set(meta_from_input.keys()) | set(meta_from_output_fallback.keys())
    for wid in sorted(all_known_wids):
        if wid in wip_data:
            continue
        t, s, l, grade, spec = meta_from_input.get(
            wid,
            meta_from_output_fallback.get(wid, default_meta),
        )
        wip_data[wid] = WIPData(
            wip_id=wid,
            stack_id=0,
            level=0,
            short_side=s,
            long_side=l,
            thickness=t,
            grade=grade,
            spec=spec,
        )

    job_data: Dict[int, JobData] = {}
    for _, row in plan_df.iterrows():
        if int(row["use_machine"]) == 0:
            continue

        wid = int(row["input_wip_id"])
        t, s, l = _parse_spec(row["spec"])
        batch = int(row["batch_count"])
        gen_flag = int(row["generates_output"]) if not pd.isna(row["generates_output"]) else 0
        out_wid = None if pd.isna(row["output_wip_id"]) else int(row["output_wip_id"])

        job_data[int(row["job_id"])] = JobData(
            job_id=int(row["job_id"]),
            input_wip_id=wid,
            grade=str(row["grade"]),
            spec=str(row["spec"]),
            batch_count=batch,
            process_time=float(row["process_time_min"]),
            cap_short=batch * s,
            cap_long=l,
            thickness=t,
            short_side=s,
            long_side=l,
            generates_output=(gen_flag == 1 and out_wid is not None),
            output_wip_id=out_wid,
        )

    # 가정:
    #   - 원자재 job (input_wip_id=0): 항상 외부 공급 → has_external_input = True
    #     (동일 규격 원자재가 여러 장 존재할 수 있으므로 인벤토리 체크 불필요)
    #   - unique job (input_wip_id>0): 초기 인벤토리에 없으면 외부 원자재
    initial_wids = {
        wid for wid, w in wip_data.items()
        if w.stack_id > 0 and w.level > 0
    }

    for job in job_data.values():
        if job.input_wip_id == 0:
            job.has_external_input = True   # 원자재 job: 항상 DIRECT_START
        else:
            job.has_external_input = job.input_wip_id not in initial_wids

    # Phase 5: generates_output=True 런의 출력재는 후속 런의 입력으로 재사용되지 않는다.
    # 해당 WIP를 PICKING 후보에서 제외하기 위해 플래그를 설정한다.
    for job in job_data.values():
        if job.generates_output and job.output_wip_id is not None:
            wip = wip_data.get(job.output_wip_id)
            if wip is not None:
                wip.is_output_wip = True

    return wip_data, job_data, inter_times, machine_times


def get_crane_time(
    src: str,
    dst: str,
    inter_times: Dict[Tuple[str, str], float],
    machine_times: Dict[str, float],
    machine_node: str = "레이저설비",
) -> float:
    if dst == machine_node:
        return machine_times.get(src, 5.0)
    if src == machine_node:
        return machine_times.get(dst, 5.0)
    return inter_times.get((src, dst), inter_times.get((dst, src), 5.0))
