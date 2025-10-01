#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alpha diversity: Wilcoxon rank-sum (Mann-Whitney U) + colored box-plots
- abundance: taxon x sample TSV (Phanta style). 열 이름에 '.k2.S' 등이 붙어도 ERR######만 자동 추출.
- metadata : TSV (예: Sample, Responder[R/NR])
- metrics : shannon, richness, simpson, pielou
- outputs : alpha_values.tsv, wilcoxon_results.tsv, alpha_<metric>_boxplot.png
"""

import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# ---------- 유틸 ----------
def normalize_sample_name(c: str) -> str:
    """열 이름에서 ERR숫자만 남기기 (없으면 원래 이름 유지)"""
    m = re.search(r"(ERR\\d+)", str(c))
    return m.group(1) if m else str(c)

def load_abundance(ab_path: Path) -> pd.DataFrame:
    """TSV → index=taxon, columns=samples; 숫자화/결측0/음수0."""
    df = pd.read_csv(ab_path, sep="\t", dtype=str)
    if df.shape[1] < 2:
        raise SystemExit("[ERROR] Abundance 파일 컬럼이 2개 미만입니다.")
    df = df.rename(columns={df.columns[0]: "taxon"}).set_index("taxon")
    df.columns = [normalize_sample_name(c) for c in df.columns]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(0.0)
    df[df < 0] = 0.0
    return df

def to_relative(df_taxa_by_sample: pd.DataFrame) -> pd.DataFrame:
    """열 합이 1이 아니면 각 열을 합으로 나눠 상대값화."""
    sums = df_taxa_by_sample.sum(axis=0)
    df_rel = df_taxa_by_sample.copy()
    nz = sums > 0
    df_rel.loc[:, nz] = df_taxa_by_sample.loc[:, nz] / sums[nz]
    return df_rel

def load_metadata(md_path: Path, sample_col: str, group_col: str) -> pd.DataFrame:
    md = pd.read_csv(md_path, sep="\t", dtype=str)
    if sample_col not in md.columns or group_col not in md.columns:
        raise SystemExit(f"[ERROR] 메타데이터에 지정한 컬럼이 없습니다. 현재 컬럼: {md.columns.tolist()}")
    md = md[[sample_col, group_col]].copy()
    md.columns = ["sample", "group"]
    md["sample"] = md["sample"].astype(str).str.strip()
    md["group"]  = md["group"].astype(str).str.strip()
    md = md.drop_duplicates(subset=["sample"])
    return md

# ---------- Alpha 지표 ----------
def alpha_metrics(df_rel: pd.DataFrame) -> pd.DataFrame:
    """
    입력: df_rel (index=taxon, columns=sample), 각 열 합=1 또는 0
    반환: DataFrame(index=sample, columns=[shannon, richness, simpson, pielou])
    """
    samples = df_rel.columns
    vals = {"shannon": [], "richness": [], "simpson": [], "pielou": []}
    for s in samples:
        p = df_rel[s].values
        # Richness: p>0 인 택손 수
        S = int((p > 0).sum())
        # Shannon: -Σ p ln p (p>0만)
        mask = p > 0
        sh = float(-(p[mask] * np.log(p[mask])).sum()) if mask.any() else 0.0
        # Simpson (Gini-Simpson): 1 - Σ p^2
        sim = float(1.0 - (p ** 2).sum())
        # Pielou: H / ln(S) (S>1일 때만 정의, 아니면 0)
        pi = float(sh / np.log(S)) if S > 1 else 0.0

        vals["shannon"].append(sh)
        vals["richness"].append(S)
        vals["simpson"].append(sim)
        vals["pielou"].append(pi)

    out = pd.DataFrame(vals, index=samples)
    out.index.name = "sample"
    return out

# ---------- 통계 + 그림 ----------
def wilcoxon_and_boxplot(alpha_df: pd.DataFrame, md_use: pd.DataFrame, out_dir: Path,
                         metrics=("shannon","richness","simpson","pielou")):
    """
    alpha_df: index=sample, columns=metrics
    md_use  : index=sample, column 'group' (예: R/NR)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # 결과 표 저장용
    res_rows = []
    # 색상 매핑(베타와 동일)
    color_map = {"R": "tab:orange", "NR": "tab:blue"}

    # 그룹 체크 (정확히 2개 필요)
    groups = sorted(md_use["group"].unique().tolist())
    if len(groups) != 2:
        raise SystemExit(f"[ERROR] 그룹이 2개가 아닙니다: {groups} (예상: ['NR','R'])")

    g1, g2 = groups[0], groups[1]
    for m in metrics:
        if m not in alpha_df.columns:
            continue
        # 데이터 벡터
        x1 = alpha_df.loc[md_use[md_use["group"] == g1].index, m].values
        x2 = alpha_df.loc[md_use[md_use["group"] == g2].index, m].values

        # Wilcoxon rank-sum ≈ Mann-Whitney U (양측)
        U, p = mannwhitneyu(x1, x2, alternative="two-sided")
        res_rows.append({"metric": m, "group1": g1, "group2": g2, "U": U, "p_value": p,
                         "n_group1": len(x1), "n_group2": len(x2)})

        # Box-plot (색깔 채움 + 점 찍기)
        fig, ax = plt.subplots(figsize=(4,5))
        bp = ax.boxplot([x1, x2], labels=[g1, g2], patch_artist=True, widths=0.6)
        # 색 입히기
        for patch, g in zip(bp["boxes"], [g1, g2]):
            patch.set_facecolor(color_map.get(g, "tab:gray"))
            patch.set_alpha(0.6)
            patch.set_edgecolor("black")
        # 중앙선/수염 색
        for elem in ["medians", "whiskers", "caps"]:
            for art in bp[elem]:
                art.set_color("black")

        # 점(분포) 살짝 흩뿌리기
        rng = np.random.default_rng(42)
        jitter1 = rng.normal(loc=1.0, scale=0.03, size=len(x1))
        jitter2 = rng.normal(loc=2.0, scale=0.03, size=len(x2))
        ax.scatter(jitter1, x1, s=14, alpha=0.8, c=color_map.get(g1, "tab:gray"), edgecolors="none")
        ax.scatter(jitter2, x2, s=14, alpha=0.8, c=color_map.get(g2, "tab:gray"), edgecolors="none")

        ax.set_title(f"{m} (Wilcoxon p={p:.3g})")
        ax.set_xlabel("Group")
        ax.set_ylabel(m.capitalize())
        fig.tight_layout()
        fig.savefig(out_dir / f"alpha_{m}_boxplot.png", dpi=150)
        plt.close(fig)

    # 결과 표 저장
    pd.DataFrame(res_rows).to_csv(out_dir / "wilcoxon_results.tsv", sep="\t", index=False)

def main():
    ap = argparse.ArgumentParser(description="Alpha diversity Wilcoxon + colored box-plots (Standalone)")
    ap.add_argument("--abundance", required=True, type=Path, help="taxon x sample TSV (Phanta 스타일)")
    ap.add_argument("--metadata",  required=True, type=Path, help="메타데이터 TSV")
    ap.add_argument("--out_dir",   required=True, type=Path, help="결과 폴더")
    ap.add_argument("--sample_col", default="Sample", help="메타데이터 샘플 컬럼명 (기본: Sample)")
    ap.add_argument("--group_col",  default="Responder", help="메타데이터 그룹 컬럼명 (기본: Responder)")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 로드
    df_taxa = load_abundance(args.abundance)             # index=taxon, columns=samples
    md = load_metadata(args.metadata, args.sample_col, args.group_col)

    # 2) 교집합
    samples = sorted(set(df_taxa.columns) & set(md["sample"]))
    if len(samples) < 2:
        raise SystemExit(f"[ERROR] 메타데이터/풍부도 교집합 샘플이 2개 미만입니다. 교집합={len(samples)}")
    df_taxa = df_taxa[samples]
    md_use = md.set_index("sample").loc[samples][["group"]]

    # 3) 상대화 (이미 상대값이라도 안전하게 한 번 더)
    df_rel = to_relative(df_taxa)

    # 4) Alpha 계산 후 저장
    alpha_df = alpha_metrics(df_rel)
    alpha_df.to_csv(out_dir / "alpha_values.tsv", sep="\t")

    # 5) Wilcoxon + Box-plot (색상: R=tab:orange, NR=tab:blue)
    wilcoxon_and_boxplot(alpha_df, md_use, out_dir)

    print(f"[OK] Alpha 완료 → {out_dir}")

if __name__ == "__main__":
    main()
