#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ------------------------------------------------------------
# 이 스크립트는 Kraken2 리포트(= taxon별 minimizer 요약)를
# kraken2-inspect 결과(= 각 taxon의 DB 내 minimizer 총수)와 합쳐서
# species/strain 커버리지를 계산한 뒤,
# 주어진 임계값(coverage, uniq_minimizers)에 따라
# "진양성(true-positive) species"만 남기도록 리포트를 잘라내는 도구입니다.
#
# 핵심 로직 요약:
#   1) Kraken report와 kraken2-inspect 파일을 taxid로 조인
#   2) coverage = uniq_minimizers / minimizers_taxa (단, minimizers_taxa < min_db_min 이면 NaN)
#   3) strain(랭크가 S<number>)에서 coverage >= cov_thr & uniq_minimizers >= min_thr 인 행을 찾음
#   4) 위 strain이 속한 species를 "보존 대상"으로 선정
#   5) 원본 리포트에서 species(S) 및 그 산하 strain(S<number>)은 보존 대상 species에 속할 때만 출력
#      (상위 계급은 --drop-upper 옵션이 없으면 그대로 통과)
# ------------------------------------------------------------

import argparse            # 커맨드라인 인자 파싱(옵션 읽기)
import logging             # 진행 로그 출력
import re                  # 랭크가 'S123' 형태인지 판별할 정규식
from pathlib import Path   # 경로 객체 다루기
from typing import Dict, Tuple, Set, List, Optional  # 타입 힌트

import pandas as pd        # 테이블 데이터 처리
import numpy as np         # 수치 연산 / NaN 처리

# taxid를 문자열 별칭으로 정의(가독성 목적)
taxid = str

# ------------------------------------------------------------
# 1) 커맨드라인 인자 정의
#    사용자는 여기서 리포트/DB/인스펙트 파일과 임계값 등을 지정합니다.
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Make a trimmed Kraken2 report that contains only true-positive species/strains."
    )
    # --kraken_report : 단일 리포트 파일 또는 리포트 파일들이 들어있는 디렉토리
    p.add_argument("--kraken_report", type=Path, required=True,
                   help="Path to a Kraken2 report OR a directory of reports.")
    # --kraken_db : Kraken2 DB 경로(여기서 taxonomy/nodes.dmp를 사용)
    p.add_argument("--kraken_db", type=Path, required=True,
                   help="Kraken2 DB dir (expects taxonomy/nodes.dmp).")
    # --inspect : kraken2-inspect 결과 파일 경로(UHGV_inspect.out 등)
    p.add_argument("--inspect", type=Path, required=True,
                   help="kraken2-inspect output (e.g., UHGV_inspect.out).")
    # --cov_thr : coverage 임계값(기본 0.10 = 10%)
    p.add_argument("--cov_thr", type=float, default=0.10,
                   help="Coverage threshold (default: 0.10).")
    # --min_thr : uniq_minimizers(해당 택손에 고유하게 매칭된 minimizer 개수) 임계값
    p.add_argument("--min_thr", type=int, default=0,
                   help="Minimum distinct minimizers in reads (default: 0).")
    # --min_db_min : DB 내 minimizers_taxa(택손의 총 minimizer 수)가 이 값 미만이면 coverage 계산 제외
    p.add_argument("--min_db_min", type=int, default=5,
                   help="Ignore DB totals < this when computing coverage (default: 5).")
    # --out_dir : 출력 디렉토리(미지정 시 리포트 폴더 하위 truepositives_YYYYMMDD 생성)
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Output directory (default: <report_dir>/truepositives_YYYYMMDD).")
    # --drop-upper : 설정 시 상위 계급(예: genus, family 등) 라인은 출력에서 제외하고 S/S*만 남김
    p.add_argument("--drop-upper", action="store_true",
                   help="If set, drop non-S ranks (keep only S/S* lines).")
    return p.parse_args()

# ------------------------------------------------------------
# 2) taxonomy/nodes.dmp 읽기
#    - 각 taxid의 부모(parent)와 계급(rank)을 딕셔너리로 보관
#    - species를 찾기 위해 계통수 거슬러 올라갈 때 활용
# ------------------------------------------------------------
def load_taxonomy(nodes_file: Path) -> Tuple[Dict[taxid, taxid], Dict[taxid, str]]:
    child_to_parent: Dict[taxid, taxid] = {}  # 자식 -> 부모 매핑
    taxid_to_rank: Dict[taxid, str] = {}      # taxid -> rank 이름 매핑
    with nodes_file.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")  # Kraken taxonomy는 탭 구분
            if len(parts) < 5:                # 방어적 코딩: 컬럼 부족 시 스킵
                continue
            child, parent, rank = parts[0], parts[2], parts[4]
            child_to_parent[child] = parent
            taxid_to_rank[child] = rank
    return child_to_parent, taxid_to_rank

# ------------------------------------------------------------
# 3) 특정 taxid에서 시작해 위로 올라가며 원하는 계급(desired_rank)을 찾음
#    - species 조상 taxid를 얻는 용도로 주로 사용
# ------------------------------------------------------------
def find_lineage_taxid(tid: taxid, desired_rank: str,
                       child_parent: Dict[taxid, taxid], taxid_rank: Dict[taxid, str]) -> taxid:
    node = tid
    while node and node != "1":                          # '1'은 루트(= unclassified의 상위)로 취급
        if taxid_rank.get(node) == desired_rank:         # 원하는 랭크를 찾으면 반환
            return node
        node = child_parent.get(node, "1")               # 한 단계 위로
    return "unclassified"                                # 찾지 못하면 'unclassified' 반환

# ------------------------------------------------------------
# 4) Kraken report + kraken2-inspect 병합 및 coverage 계산
#    - report: 샘플별 minimizer 관측치(uniq_minimizers 포함)
#    - inspect: DB 내 해당 taxon의 minimizer 총수(minimizers_taxa)
#    - coverage = uniq_minimizers / minimizers_taxa
#    - species/strain 행만 따로 분리
# ------------------------------------------------------------
def load_and_compute(report_path: Path, inspect_path: Path, nodes,
                     min_db_min: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    child_parent, taxid_rank = nodes

    # Kraken report 읽기: 표준 8컬럼(문서/버전마다 이름은 다르지만 여기선 고정)
    df_report = pd.read_csv(
        report_path, sep="\t",
        names=["fraction","fragments","assigned","minimizers","uniq_minimizers","rank","taxid","name"],
        dtype={"taxid": str}, engine="python",
    )

    # kraken2-inspect 읽기: 해당 taxon의 DB 정보(여기서 minimizers_taxa 사용)
    df_inspect = pd.read_csv(
        inspect_path, sep="\t",
        names=["frac_i","minimizers_clade","minimizers_taxa","rank_i","taxid_i","sci_name_i"],
        dtype={"taxid_i": str}, engine="python",
    )

    # taxid 기준 내부 조인(inner) → 리포트에 존재하고 DB에도 존재하는 taxon만 취급
    df = df_report.merge(df_inspect, left_on="taxid", right_on="taxid_i", how="inner")

    # 수치 컬럼 안전 변환
    df["minimizers_taxa"] = pd.to_numeric(df["minimizers_taxa"], errors="coerce")
    df["uniq_minimizers"] = pd.to_numeric(df["uniq_minimizers"], errors="coerce")

    # coverage 계산:
    #   - DB에서 해당 taxon의 minimizers_taxa(총수)가 충분히 커야 안정적
    #   - 작으면(예: 5 미만) coverage를 NaN으로 두어 필터링에서 제외되게 함
    df["coverage"] = np.where(
        df["minimizers_taxa"] >= min_db_min,
        df["uniq_minimizers"] / df["minimizers_taxa"],
        np.nan,
    )

    # species 행만 분리(랭크가 정확히 'S')
    df_species = df[df["rank"] == "S"].copy()
    # strain 행만 분리(랭크가 'S123'처럼 'S' 뒤에 숫자가 붙은 형태)
    df_strain  = df[df["rank"].str.match(r"^S[0-9]+$", na=False)].copy()

    # 두 데이터프레임 모두 각 행이 속한 species 조상 taxid를 계산해 추가
    for sub in (df_species, df_strain):
        if not sub.empty:
            sub["species_taxid"] = sub["taxid"].apply(
                lambda x: find_lineage_taxid(x, "species", child_parent, taxid_rank)
            )

    # species 전용 / strain 전용 프레임 반환
    return df_species, df_strain

# ------------------------------------------------------------
# 5) 필터 기준에 따라 "보존해야 할 species/strain" 집합 계산
#    - 본 구현은 "strain 조건으로 species 채택" 전략:
#      strain 중 (coverage >= cov_thr) & (uniq_minimizers >= min_thr)인 것이 하나라도 있으면
#      그 strain의 species를 보존 대상에 추가합니다.
# ------------------------------------------------------------
def call_tp(df_species: pd.DataFrame, df_strain: pd.DataFrame,
            cov_thr: float, min_thr: int) -> Tuple[Set[taxid], Set[taxid]]:
    # strain에서 임계값을 만족하는 행만 추출
    tp_strains_df = df_strain.loc[
        (df_strain["coverage"] >= cov_thr) & (df_strain["uniq_minimizers"] >= min_thr)
    ]
    # 해당 strain이 속한 species taxid 집합
    keep_species: Set[taxid] = set(tp_strains_df["species_taxid"].dropna().astype(str))
    # (옵션) strain 자체 id도 집합으로 저장 — 현재 필터링 로직에서는 직접 사용하진 않음
    keep_strains: Set[taxid] = set(tp_strains_df["taxid"].astype(str))
    return keep_species, keep_strains

# ------------------------------------------------------------
# 6) 원본 리포트를 한 줄씩 읽으면서 보존 대상만 출력
#    - species(S) 라인은: 그 species가 보존 대상(keep_species)이면 출력
#    - strain(S123) 라인은: "부모 species"가 보존 대상이면 출력 (strain 본인 조건은 다시 보지 않음)
#    - 그 외 상위 랭크 라인은: --drop-upper 미지정 시 그대로 통과(맥락 보존용)
# ------------------------------------------------------------
def write_trimmed_report(original_report: Path, out_report: Path,
                         keep_species: Set[taxid], keep_strains: Set[taxid],
                         nodes, drop_upper: bool) -> None:
    child_parent, taxid_rank = nodes
    # 원본 읽기 / 출력 쓰기 동시 오픈
    with original_report.open(encoding="utf-8") as infile, out_report.open("w", encoding="utf-8") as outfile:
        for line in infile:
            cols = line.rstrip("\n").split("\t")  # 탭 분리
            if len(cols) < 7:                     # 최소 컬럼수 확인(방어적)
                continue
            rank = cols[5]                        # 컬럼 6: 랭크 코드('S' or 'S123' 등)
            tx   = cols[6]                        # 컬럼 7: taxid(문자열로 처리)

            if rank == "S":
                # species 라인이면, 그 species 자신이 보존 대상인지 검사
                sp = find_lineage_taxid(tx, "species", child_parent, taxid_rank)
                if sp in keep_species:
                    outfile.write(line)

            elif re.match(r"^S[0-9]+$", rank):
                # strain 라인이면, strain 본인의 TP 여부를 묻지 않고 "부모 species"가 보존 대상이면 출력
                sp = find_lineage_taxid(tx, "species", child_parent, taxid_rank)
                if sp in keep_species:
                    outfile.write(line)

            else:
                # 상위 랭크(속/과/목...)는 옵션에 따라 유지/제거
                if not drop_upper:
                    outfile.write(line)

# ------------------------------------------------------------
# 7) 입력 경로가 파일인지/디렉토리인지에 따라 리포트 목록 만들기
#    - 디렉토리면 흔한 이름 패턴으로 스캔
# ------------------------------------------------------------
def find_reports(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    files: List[Path] = []
    for patt in ("*.report", "*report*.txt", "*_report.txt"):
        files.extend(path.glob(patt))
    return sorted(files)

# ------------------------------------------------------------
# 8) 출력 디렉토리 결정 및 생성
#    - 명시(out_dir) 없으면 <리포트_폴더>/truepositives_YYYYMMDD 생성
# ------------------------------------------------------------
def ensure_outdir(base: Path, explicit: Optional[Path]) -> Path:
    out = explicit if explicit is not None else base / f"truepositives_{pd.Timestamp.now().strftime('%Y%m%d')}"
    out.mkdir(parents=True, exist_ok=True)
    return out

# ------------------------------------------------------------
# 9) 메인 함수: 전체 파이프라인 실행
# ------------------------------------------------------------
def main():
    args = parse_args()  # 인자 읽기
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")  # 로그 포맷/레벨

    # Kraken DB에서 taxonomy/nodes.dmp 경로 확인
    nodes_path = args.kraken_db / "taxonomy" / "nodes.dmp"
    if not nodes_path.exists():
        raise FileNotFoundError(f"Cannot find taxonomy nodes at: {nodes_path}")

    # taxonomy 로드(부모/랭크 테이블 두 개 반환)
    nodes = load_taxonomy(nodes_path)

    # 리포트 파일(또는 디렉토리 내 여러 파일) 수집
    reports = find_reports(args.kraken_report)
    if not reports:
        raise FileNotFoundError(f"No report files found under: {args.kraken_report}")

    # 출력 디렉토리 준비
    base_dir = args.kraken_report.parent if args.kraken_report.is_file() else args.kraken_report
    out_dir  = ensure_outdir(base_dir, args.out_dir)
    logging.info(f"Output directory: {out_dir}")

    # 각 리포트 파일에 대해 반복 처리
    for rep in reports:
        sample = rep.stem                     # 파일명(확장자 제외)을 샘플명처럼 사용
        logging.info(f"Processing: {rep.name}")

        # coverage 계산용 테이블 생성(species/strain 분리)
        df_sp, df_st = load_and_compute(rep, args.inspect, nodes, args.min_db_min)

        # 임계값에 따라 "보존할 species/strain 집합" 결정
        keep_species, keep_strains = call_tp(df_sp, df_st, args.cov_thr, args.min_thr)

        # 잘라낸(필터링된) 리포트 파일명: <원본>.report.truepositives.txt
        trimmed_path = out_dir / f"{sample}.report.truepositives.txt"

        # 실제 라인 필터링/쓰기
        write_trimmed_report(rep, trimmed_path, keep_species, keep_strains, nodes, args.drop_upper)

        logging.info(f"✓ Wrote: {trimmed_path.name}")

        # 보존 species가 하나도 없으면 경고(임계값이 너무 높을 수 있음)
        if not keep_species:
            logging.warning("No true-positive species found for this sample with current thresholds.")

    logging.info("Done.")  # 전체 완료 로그

# 파이썬 스크립트로 직접 실행될 때만 main() 호출
if __name__ == "__main__":
    main()
