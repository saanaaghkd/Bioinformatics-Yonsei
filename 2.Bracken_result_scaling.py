#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bracken → Genome length normalization (연구실 기준)
- species 레벨 count / species genome length × 1e6 (RPMB)
- 길이 파일 컬럼명 직접 지정 가능(--length_id_col, --length_len_col)
- 매칭 키 선택(--join_on taxid|name|auto), 기본 auto: taxid로 먼저 시도 후, 이름 겹침이 더 크면 name로 전환
- 드롭된 종 목록 저장(dropped_species_*.tsv)
- 출력: rpmb_matrix.tsv, 샘플별 <sample>.rpmb.tsv
"""

import argparse
from pathlib import Path
import sys, re
from typing import List, Tuple

import pandas as pd
import numpy as np

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Bracken → length-normalized RPMB (reads per megabase)")
    p.add_argument("--bracken", type=Path, required=True,
                   help="Bracken 결과 파일 또는 폴더(*.bracken / *.tsv)")
    p.add_argument("--lengths", type=Path, required=True,
                   help="genome length TSV (ID/length 컬럼 지정 또는 자동)")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="결과 저장 폴더")
    p.add_argument("--count_col", type=str, default=None,
                   help="Bracken count 컬럼명(기본 자동: new_est_reads/estimated_num_reads/...)")
    p.add_argument("--sample_regex", type=str, default=None,
                   help="파일명에서 샘플명 추출 정규식(캡처 그룹 1 사용, 선택)")
    p.add_argument("--length_id_col", type=str, default=None,
                   help="길이 파일의 ID 컬럼명 (예: uhgv_votu, species_taxid)")
    p.add_argument("--length_len_col", type=str, default=None,
                   help="길이 파일의 길이 컬럼명 (예: median_genome_length, genome_length)")
    p.add_argument("--join_on", type=str, choices=["auto","taxid","name"], default="auto",
                   help="길이 테이블과 매칭 키: taxid(Bracken taxonomy_id) 또는 name(Bracken name)")
    return p.parse_args()

# ---------- helpers ----------
def find_bracken_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    files: List[Path] = []
    for patt in ("*.bracken", "*.tsv", "*bracken*.txt"):
        files.extend(path.glob(patt))
    return sorted(set(files))

def infer_sample_name(path: Path, sample_regex: str = None) -> str:
    if sample_regex:
        m = re.search(sample_regex, path.name)
        if m:
            return m.group(1)
    return path.stem

def autodetect_cols(df: pd.DataFrame) -> Tuple[str, str]:
    id_candidates  = ["species_taxid","taxid","ncbi_taxid","ncbi_taxon_id","taxonomy_id","taxon_id","uhgv_votu"]
    len_candidates = ["genome_length","median_length","median_size","length","len","size","genome_len","median_genome_length"]
    id_col  = next((c for c in id_candidates  if c in df.columns), None)
    len_col = next((c for c in len_candidates if c in df.columns), None)
    if id_col is None or len_col is None:
        raise SystemExit(f"[ERROR] Cannot find ID/length columns. Have: {list(df.columns)}")
    return id_col, len_col

def load_lengths(lengths_path: Path, id_col: str=None, len_col: str=None) -> pd.Series:
    df = pd.read_csv(lengths_path, sep="\t", header=0)
    if id_col is None or len_col is None:
        id_col, len_col = autodetect_cols(df)
    if id_col not in df.columns or len_col not in df.columns:
        raise SystemExit(f"[ERROR] lengths must have columns {id_col}, {len_col}. Have: {list(df.columns)}")
    df = df[[id_col, len_col]].copy()
    df[len_col] = pd.to_numeric(df[len_col], errors="coerce")
    df = df.dropna(subset=[len_col])
    df = df[df[len_col] > 0]
    if df.empty:
        raise SystemExit("[ERROR] No valid positive genome_length in lengths file.")
    med = df.groupby(id_col)[len_col].median()
    med.index = med.index.astype(str)
    return med  # index: ID(str), value: length

def read_bracken_species_counts(bracken_path: Path, count_col: str=None) -> pd.DataFrame:
    df = pd.read_csv(bracken_path, sep="\t", header=0, dtype=str, engine="python")
    if "taxonomy_lvl" in df.columns:
        df = df[df["taxonomy_lvl"] == "S"].copy()
    # count 컬럼 탐색
    candidates = ["new_est_reads","estimated_num_reads","est_reads","new_est_reads_mapped"]
    use_col = count_col if (count_col and count_col in df.columns) else next((c for c in candidates if c in df.columns), None)
    if use_col is None:
        raise SystemExit(f"[ERROR] {bracken_path.name}: cannot find count column; specify --count_col")

    # taxid/name 정리
    taxid_col = None
    for c in ("taxonomy_id","species_taxid","ncbi_taxid","taxid","taxon_id"):
        if c in df.columns:
            taxid_col = c; break
    if taxid_col is None:
        raise SystemExit(f"[ERROR] {bracken_path.name}: cannot find a taxid column")

    if "name" not in df.columns:
        df["name"] = df[taxid_col]

    out = pd.DataFrame({
        "taxid": df[taxid_col].astype(str),
        "name": df["name"].astype(str),
        "count": pd.to_numeric(df[use_col], errors="coerce").fillna(0.0)
    })
    # 동일 ID/name 중복 합산
    out_taxid = out.groupby(["taxid"], as_index=False)["count"].sum().set_index("taxid")
    out_name  = out.groupby(["name"],  as_index=False)["count"].sum().set_index("name")
    # 이름 사전(마지막 본 이름)
    name_map = dict(zip(out["taxid"], out["name"]))
    return out_taxid, out_name, name_map  # 두 키 모두 반환

# ---------- main ----------
def main():
    a = parse_args()
    if not a.bracken.exists():
        raise SystemExit(f"[ERROR] --bracken not found: {a.bracken}")
    if not a.lengths.exists():
        raise SystemExit(f"[ERROR] --lengths not found: {a.lengths}")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    # 길이 테이블 로드
    med_len = load_lengths(a.lengths, a.length_id_col, a.length_len_col)

    # Bracken 파일 수집
    files = find_bracken_files(a.bracken)
    if not files:
        raise SystemExit("[ERROR] No Bracken files found.")

    # 두 방식의 매트릭스를 모두 구성해보고, 겹침이 큰 쪽을 선택(auto일 때)
    mx_taxid = pd.DataFrame()
    mx_name  = pd.DataFrame()
    name_map_all = {}

    for f in files:
        sample = infer_sample_name(f, a.sample_regex)
        df_taxid, df_name, name_map = read_bracken_species_counts(f, a.count_col)
        name_map_all.update(name_map)
        mx_taxid = df_taxid.rename(columns={"count": sample}) if mx_taxid.empty else mx_taxid.join(df_taxid.rename(columns={"count": sample}), how="outer")
        mx_name  = df_name.rename(columns={"count": sample})  if mx_name.empty  else mx_name.join(df_name.rename(columns={"count": sample}),  how="outer")

    mx_taxid = mx_taxid.fillna(0.0)
    mx_name  = mx_name.fillna(0.0)

    # 겹침 계산
    common_taxid = set(mx_taxid.index.astype(str)) & set(med_len.index.astype(str))
    common_name  = set(mx_name.index.astype(str))  & set(med_len.index.astype(str))

    # 어떤 키로 조인할지 결정
    if a.join_on == "taxid":
        key = "taxid"
    elif a.join_on == "name":
        key = "name"
    else:
        # auto: 겹침이 더 큰 쪽 선택
        key = "taxid" if len(common_taxid) >= len(common_name) else "name"

    if key == "taxid":
        if len(common_taxid) == 0:
            print("[WARN] taxid로는 겹침이 0 → name으로 시도합니다.", file=sys.stderr)
            key = "name"  # fallback
        use_mx = mx_taxid
        common = sorted(common_taxid)
        dropped = sorted(set(use_mx.index.astype(str)) - set(common))
    else:
        if len(common_name) == 0:
            raise SystemExit("[ERROR] name으로도 겹침이 0입니다. 길이 파일 ID가 Bracken name/taxid와 맞는지 확인하세요.")
        use_mx = mx_name
        common = sorted(common_name)
        dropped = sorted(set(use_mx.index.astype(str)) - set(common))

    if dropped:
        pd.DataFrame({("name" if key=="name" else "taxid"): dropped}).to_csv(
            a.out_dir / f"dropped_species_{key}.tsv", sep="\t", index=False)
        print(f"[WARN] {len(dropped)} entries have no length → dropped (see dropped_species_{key}.tsv)")

    use_mx = use_mx.loc[common].copy()
    med_use = med_len.loc[common].copy()
    med_use.index = med_use.index.astype(str)

    # 길이 보정 + ×1e6
    length_corrected = use_mx.div(med_use, axis=0)
    rpmb_mx = length_corrected * 1_000_000.0

    # 합본 저장
    rpmb_mx.to_csv(a.out_dir / "rpmb_matrix.tsv", sep="\t", float_format="%.8g")

    # per-sample 파일
    for sample in rpmb_mx.columns:
        sub = rpmb_mx[[sample]].reset_index().rename(columns={"index": ("name" if key=="name" else "species_taxid"),
                                                              sample: "rpmB"})
        # name 열 보강(가능하면)
        if key == "taxid":
            sub["name"] = sub["species_taxid"].map(name_map_all).fillna(sub["species_taxid"])
            sub = sub[[ "species_taxid","name","rpmB"]]
        else:
            sub = sub.rename(columns={"name":"species_name"})
            sub = sub[["species_name","rpmB"]]
        sub = sub.sort_values("rpmB", ascending=False)
        sub.to_csv(a.out_dir / f"{sample}.rpmb.tsv", sep="\t", index=False, float_format="%.8g")

    print(f"[OK] Files processed: {len(files)}")
    print(f"[OK] Matrix:  {a.out_dir / 'rpmb_matrix.tsv'}")
    print(f"[OK] Per-sample: {a.out_dir}/*.rpmb.tsv")
    for sample in rpmb_mx.columns[:5]:
        top3 = rpmb_mx[sample].sort_values(ascending=False).head(3)
        print(f"[TOP3] {sample}:")
        for k, v in top3.items():
            print(f"   {k}: {v:.3f} RPMB")

if __name__ == "__main__":
    main()
