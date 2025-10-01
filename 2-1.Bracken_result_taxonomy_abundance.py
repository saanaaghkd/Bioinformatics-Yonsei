#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bracken 결과 → Phanta 최종 형식(final_merged_outputs/) 생성기
- final_merged_outputs/counts.txt
- final_merged_outputs/relative_read_abundance.txt
- final_merged_outputs/relative_taxonomic_abundance.txt  (길이 보정 + 샘플 내 합=1)
입력:
  (1) Bracken species 레벨 결과(단일 파일 또는 디렉토리)
  (2) species 길이 테이블(이미 median이면 그대로 사용)
매칭:
  --join_on name  : Bracken의 name 과 길이파일 ID를 맞춤 (UHGV vOTU 권장)
  --join_on taxid : Bracken taxonomy_id 와 길이파일 ID를 맞춤
길이파일 컬럼명 지정:
  --length_id_col, --length_len_col
"""

import argparse
from pathlib import Path
import sys, re
from typing import List, Tuple

import pandas as pd
import numpy as np

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Make Phanta-style final_merged_outputs from Bracken + length table")
    p.add_argument("--bracken", type=Path, required=True,
                   help="Bracken 결과: 파일 또는 디렉토리(*.bracken / *.tsv / *bracken*.txt)")
    p.add_argument("--lengths", type=Path, required=True,
                   help="species 길이 TSV (median이면 그대로 사용).")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="출력 루트(여기에 final_merged_outputs/ 생성)")
    p.add_argument("--join_on", choices=["name","taxid"], default="name",
                   help="길이 테이블과 매칭 키 (UHGV vOTU는 보통 name 권장)")
    p.add_argument("--length_id_col", type=str, default=None,
                   help="길이파일의 ID 컬럼명 (예: uhgv_votu / species_taxid)")
    p.add_argument("--length_len_col", type=str, default=None,
                   help="길이파일의 길이 컬럼명 (예: median_genome_length / genome_length)")
    p.add_argument("--count_col", type=str, default=None,
                   help="Bracken count 컬럼명 (기본 자동: new_est_reads / estimated_num_reads ...)")
    p.add_argument("--sample_regex", type=str, default=None,
                   help="파일명에서 샘플명 추출 정규식(캡처 그룹 1 사용). 미지정 시 stem 사용")
    return p.parse_args()

# -------------- helpers --------------
def find_bracken_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    files: List[Path] = []
    for patt in ("*.bracken", "*.tsv", "*bracken*.txt"):
        files.extend(path.glob(patt))
    return sorted(set(files))

def infer_sample_name(path: Path, sample_regex: str|None) -> str:
    if sample_regex:
        m = re.search(sample_regex, path.name)
        if m:
            return m.group(1)
    return path.stem

def autodetect_length_cols(df: pd.DataFrame) -> Tuple[str,str]:
    id_candidates  = ["species_taxid","taxid","ncbi_taxid","taxonomy_id","taxon_id","uhgv_votu","id","name"]
    len_candidates = ["genome_length","median_genome_length","median_length","length","size","genome_len"]
    id_col  = next((c for c in id_candidates  if c in df.columns), None)
    len_col = next((c for c in len_candidates if c in df.columns), None)
    if id_col is None or len_col is None:
        raise SystemExit(f"[ERROR] Cannot find ID/length columns in length table. Have: {list(df.columns)}")
    return id_col, len_col

def load_lengths(lengths_path: Path, id_col: str|None, len_col: str|None) -> pd.Series:
    df = pd.read_csv(lengths_path, sep="\t", header=0)
    if id_col is None or len_col is None:
        id_col, len_col = autodetect_length_cols(df)
    if id_col not in df.columns or len_col not in df.columns:
        raise SystemExit(f"[ERROR] Lengths must have columns {id_col}, {len_col}.")
    df = df[[id_col, len_col]].copy()
    df[len_col] = pd.to_numeric(df[len_col], errors="coerce")
    df = df.dropna(subset=[len_col])
    df = df[df[len_col] > 0]
    if df.empty:
        raise SystemExit("[ERROR] No positive genome lengths in the table.")
    # 여러 행이 있어도 median으로 수렴 (이미 median이어도 안전)
    med = df.groupby(id_col)[len_col].median()
    med.index = med.index.astype(str)
    return med  # index=ID(str), value=median length

def read_bracken_species(fp: Path, count_col: str|None, join_on: str) -> pd.DataFrame:
    """
    반환: DataFrame(index = join_key(name or taxid), columns=['count'])
    - species(S)만 사용
    - 같은 key가 여러 줄이면 합산
    """
    df = pd.read_csv(fp, sep="\t", header=0, dtype=str, engine="python")
    if "taxonomy_lvl" in df.columns:
        df = df[df["taxonomy_lvl"] == "S"].copy()

    # count 컬럼 자동 선택
    candidates = ["new_est_reads","estimated_num_reads","est_reads","new_est_reads_mapped"]
    use_col = count_col if (count_col and count_col in df.columns) else next((c for c in candidates if c in df.columns), None)
    if use_col is None:
        raise SystemExit(f"[ERROR] {fp.name}: cannot find count column; specify --count_col")

    if "name" not in df.columns:
        df["name"] = df["taxonomy_id"] if "taxonomy_id" in df.columns else df.iloc[:,0]

    if join_on == "taxid":
        taxid_col = None
        for c in ("taxonomy_id","species_taxid","ncbi_taxid","taxid","taxon_id"):
            if c in df.columns:
                taxid_col = c; break
        if taxid_col is None:
            raise SystemExit(f"[ERROR] {fp.name}: cannot find any taxid column for --join_on taxid")
        key = df[taxid_col].astype(str)
    else:
        key = df["name"].astype(str)

    count = pd.to_numeric(df[use_col], errors="coerce").fillna(0.0)
    out = pd.DataFrame({"key": key, "count": count})
    out = out.groupby("key", as_index=True)["count"].sum().to_frame()
    return out  # index=key, col=count

# ---------------- main ----------------
def main():
    a = parse_args()

    # 경로/출력 준비
    if not a.bracken.exists():
        raise SystemExit(f"[ERROR] --bracken not found: {a.bracken}")
    if not a.lengths.exists():
        raise SystemExit(f"[ERROR] --lengths not found: {a.lengths}")
    out_root = a.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    out_final = out_root / "final_merged_outputs"
    out_final.mkdir(parents=True, exist_ok=True)

    # 길이 로드 (median)
    med_len = load_lengths(a.lengths, a.length_id_col, a.length_len_col)

    # Bracken 파일 모으기
    files = find_bracken_files(a.bracken)
    if not files:
        raise SystemExit("[ERROR] No Bracken files found.")

    # counts 매트릭스 만들기 (행=taxon key, 열=sample, 값=count)
    counts_mx = pd.DataFrame()
    for f in files:
        sample = infer_sample_name(f, a.sample_regex)
        s = read_bracken_species(f, a.count_col, a.join_on)  # index=key, col=count
        s = s.rename(columns={"count": sample})
        counts_mx = s if counts_mx.empty else counts_mx.join(s, how="outer")

    counts_mx = counts_mx.fillna(0.0)
    counts_mx.index.name = "taxon"  # Phanta 스타일: 첫 열이 taxon ID/이름

    # ---- final_merged_outputs/counts.txt ----
    counts_path = out_final / "counts.txt"
    counts_mx.to_csv(counts_path, sep="\t", float_format="%.8g")

    # ---- final_merged_outputs/relative_read_abundance.txt ----
    # 샘플 내 총 count로 나눠 합이 1이 되게 (Phanta의 relative_read_abundance 의미)
    col_sums = counts_mx.sum(axis=0)
    rra = counts_mx.div(col_sums.replace(0, np.nan), axis=1).fillna(0.0)
    rra_path = out_final / "relative_read_abundance.txt"
    rra.to_csv(rra_path, sep="\t", float_format="%.8g")

    # ---- final_merged_outputs/relative_taxonomic_abundance.txt ----
    # 길이 보정 후(= counts / length) 샘플 내 합이 1이 되게
    # 길이 없는 taxon은 제외
    common = counts_mx.index.astype(str).intersection(med_len.index.astype(str))
    dropped = sorted(set(counts_mx.index.astype(str)) - set(common))
    if dropped:
        pd.DataFrame({"taxon": dropped}).to_csv(out_root / "dropped_taxa_no_length.tsv", sep="\t", index=False)
        print(f"[WARN] {len(dropped)} taxa without length → dropped (see dropped_taxa_no_length.tsv)")

    lc = counts_mx.loc[common].div(med_len.loc[common], axis=0)
    col_sums_lc = lc.sum(axis=0)
    rta = lc.div(col_sums_lc.replace(0, np.nan), axis=1).fillna(0.0)
    rta.index.name = "taxon"
    rta_path = out_final / "relative_taxonomic_abundance.txt"
    rta.to_csv(rta_path, sep="\t", float_format="%.8g")

    # 요약 출력
    print("[OK] Wrote Phanta-style final_merged_outputs:")
    print(f"  - {counts_path}")
    print(f"  - {rra_path}")
    print(f"  - {rta_path}")
    # 간단 체크(앞 3개 샘플)
    for sample in list(counts_mx.columns)[:3]:
        print(f"[CHECK] {sample}: counts sum={counts_mx[sample].sum():.0f}, "
              f"RRA sum={rra[sample].sum():.3f}, RTA sum={rta[sample].sum():.3f}")

if __name__ == "__main__":
    main()
