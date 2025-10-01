#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bray–Curtis PCoA + PERMANOVA + 95% 부트스트랩 타원 (라벨 없음, 점선 테두리 + 40% 채움)

입력:
  - abundance: 행=taxon, 열=sample (Phanta 스타일 TSV). 열 이름에 '.k2.S' 등이 붙어 있어도 자동 정리.
  - metadata : 탭 구분 TSV, 샘플/그룹 컬럼 지정(예: Sample, Responder)

출력:
  - braycurtis_pcoa_plot.png      (그룹별 점 + 95% 부트스트랩 타원: 얇은 점선 + 40% 채움)
  - braycurtis_pcoa_scores.tsv    (샘플 PC 좌표)
  - permanova_braycurtis.txt      (PERMANOVA 결과)
  - dispersion_check.txt          (그룹 내 평균 쌍거리)
"""

import argparse, sys, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 서버 환경에서도 그림 저장만
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from numpy.linalg import inv, LinAlgError
from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa
from skbio.stats.distance import DistanceMatrix, permanova

# -------------------- 유틸 --------------------
def normalize_sample_name(c: str) -> str:
    """열 이름에서 ERR숫자만 남기기 (없으면 원래 이름 유지)"""
    m = re.search(r"(ERR\d+)", str(c))
    return m.group(1) if m else str(c)

def load_abundance(ab_path: Path) -> pd.DataFrame:
    """TSV 읽어 taxon index, 열=sample. 숫자화/결측0/음수→0."""
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
    """열(샘플) 합으로 나눠 상대값화. 합=0인 샘플은 그대로."""
    sums = df_taxa_by_sample.sum(axis=0)
    df_rel = df_taxa_by_sample.copy()
    nz = sums > 0
    df_rel.loc[:, nz] = df_taxa_by_sample.loc[:, nz] / sums[nz]
    return df_rel

def load_metadata(md_path: Path, sample_col: str, group_col: str) -> pd.DataFrame:
    """메타데이터 준비: 필요한 두 컬럼만, 문자열/공백, 중복 제거."""
    md = pd.read_csv(md_path, sep="\t", dtype=str)
    if sample_col not in md.columns or group_col not in md.columns:
        raise SystemExit(f"[ERROR] 메타데이터 컬럼이 없습니다. 현재 컬럼: {md.columns.tolist()}")
    md = md[[sample_col, group_col]].copy()
    md.columns = ["sample", "group"]
    md["sample"] = md["sample"].astype(str).str.strip()
    md["group"]  = md["group"].astype(str).str.strip()
    md = md.drop_duplicates(subset=["sample"])
    return md

def mean_within_group(dist: pd.DataFrame, samples: list[str]) -> float:
    """그룹 내 평균 쌍거리 (상삼각 평균)"""
    if len(samples) < 2:
        return float("nan")
    sub = dist.loc[samples, samples].values
    iu = np.triu_indices_from(sub, 1)
    return float(sub[iu].mean()) if iu[0].size > 0 else float("nan")

# --------- 부트스트랩 타원 스케일(q95) 추정 ----------
def mahalanobis_sq(X: np.ndarray, mu: np.ndarray, Sigma_inv: np.ndarray) -> np.ndarray:
    """각 행에 대해 (x-mu)^T Sigma_inv (x-mu) (제곱 마할라노비스)"""
    D = X - mu
    return np.einsum('ij,ji->i', D @ Sigma_inv, D.T)

def bootstrap_q95_mah2(X: np.ndarray, n_boot: int = 1000, eps: float = 1e-8) -> float:
    """
    부트스트랩(리샘플링)으로 마할라노비스 제곱거리의 95% 분위수를 추정.
    - 각 부트스트랩에서 (mu_b, Sigma_b) 추정 후, 리샘플된 점들의 d^2 분포에서 95% 분위 계산
    - q95들을 모아 중앙값(robust) 사용
    - 공분산이 특이하면 대각에 작은 ridge 추가
    """
    n = X.shape[0]
    if n < 2:
        return chi2.ppf(0.95, df=2)  # 최소한의 fallback
    qs = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)  # with replacement
        Xb = X[idx, :]
        mu_b = Xb.mean(axis=0)
        # 공분산 (ddof=1), 특이시 ridge
        Sigma_b = np.cov(Xb.T, ddof=1)
        if not np.isfinite(Sigma_b).all():
            continue
        # 2x2 보장, ridge
        tr = np.trace(Sigma_b)
        Sigma_b = Sigma_b + (eps + 1e-12 * max(tr, 1.0)) * np.eye(2)
        try:
            Sigma_inv_b = inv(Sigma_b)
        except LinAlgError:
            continue
        d2 = mahalanobis_sq(Xb, mu_b, Sigma_inv_b)
        if np.isfinite(d2).any():
            qs.append(np.quantile(d2[np.isfinite(d2)], 0.95))
    if len(qs) == 0:
        return chi2.ppf(0.95, df=2)
    return float(np.median(qs))

# -------------------- PCoA 그림 --------------------
def plot_pcoa_boot(scores_df: pd.DataFrame, explained: pd.Series, md_use: pd.DataFrame,
                   out_png: Path, n_boot: int = 1000):
    """
    라벨 없이 산점도 + '부트스트랩 q95'를 사용한 95% 타원 (얇은 점선 + 40% 채움)
    - scores_df: index=sample, columns=['PC1','PC2',...]
    - explained: 각 PC 설명력 (0~1), pandas Series
    - md_use: index=sample, column 'group'
    """
    fig, ax = plt.subplots(figsize=(6,5))

    # 색상: R=주황, NR=파랑, 그 외 회색
    color_map = {"R": "tab:orange", "NR": "tab:blue"}
    groups = sorted(md_use["group"].unique().tolist())

    merged = scores_df.join(md_use, how="inner")  # index=sample
    for g in groups:
        sub = merged[merged["group"] == g]
        color = color_map.get(g, "tab:gray")

        # 점
        ax.scatter(sub["PC1"], sub["PC2"], s=20, c=color, alpha=0.9,
                   edgecolors="none", label=g)

        # 부트스트랩 타원 (샘플 ≥ 2)
        if len(sub) >= 2:
            X = sub[["PC1","PC2"]].values
            # 1) 샘플 공분산/고유분해 (모양)
            mu_hat = X.mean(axis=0)
            Sigma_hat = np.cov(X.T, ddof=1)
            # 수치 안정화
            tr = np.trace(Sigma_hat)
            Sigma_hat = Sigma_hat + (1e-8 + 1e-12 * max(tr, 1.0)) * np.eye(2)
            vals, vecs = np.linalg.eigh(Sigma_hat)
            vals = np.clip(vals, 0, None)

            # 2) 부트스트랩으로 q95 추정 (마할라노비스 제곱거리의 95% 분위)
            q95 = bootstrap_q95_mah2(X, n_boot=n_boot)

            # 3) 타원 반축 길이: 2 * sqrt(q95 * 고유값)
            width, height = 2 * np.sqrt(q95 * vals)
            angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))

            ell = Ellipse(xy=mu_hat, width=width, height=height, angle=angle,
                          facecolor=color, edgecolor=color, linestyle=":", lw=1.0, alpha=0.40)
            ax.add_patch(ell)

    pc1_var = explained.iloc[0]*100
    pc2_var = explained.iloc[1]*100
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)")
    ax.set_title("PCoA (Bray–Curtis) — 95% Bootstrap Ellipses")
    ax.legend(title="Group")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

# -------------------- 메인 --------------------
def main():
    ap = argparse.ArgumentParser(description="Bray–Curtis PCoA + PERMANOVA + 95% Bootstrap Ellipses")
    ap.add_argument("--abundance", required=True, type=Path, help="taxon x sample TSV (Phanta 스타일)")
    ap.add_argument("--metadata",  required=True, type=Path, help="메타데이터 TSV")
    ap.add_argument("--out_dir",   required=True, type=Path, help="결과 폴더")
    ap.add_argument("--sample_col", default="Sample", help="메타데이터 샘플 컬럼명 (기본: Sample)")
    ap.add_argument("--group_col",  default="Responder", help="메타데이터 그룹 컬럼명 (기본: Responder)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="부트스트랩 반복 수 (기본: 1000)")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 데이터 로드
    df_taxa = load_abundance(args.abundance)             # index=taxon, columns=samples
    md = load_metadata(args.metadata, args.sample_col, args.group_col)

    # 2) 교집합 샘플
    samples = sorted(set(df_taxa.columns) & set(md["sample"]))
    if len(samples) < 2:
        raise SystemExit(f"[ERROR] 메타데이터/풍부도 교집합 샘플이 2개 미만입니다. 교집합={len(samples)}")
    df_taxa = df_taxa[samples]
    md_use = md.set_index("sample").loc[samples][["group"]]

    # 3) 상대화
    df_rel = to_relative(df_taxa)

    # 4) Bray–Curtis 거리행렬 (행=샘플, 열=택손 → 전치)
    X = df_rel.T.values
    dm = beta_diversity(metric="braycurtis", counts=X, ids=samples)

    # 5) PCoA
    ordn = pcoa(dm)
    scores = ordn.samples[["PC1","PC2"]].copy()
    scores.index.name = "sample"
    scores.to_csv(out_dir / "braycurtis_pcoa_scores.tsv", sep="\t")

    # 6) PERMANOVA (index=sample 유지)
    res = permanova(dm, md_use, column="group", permutations=999)
    with (out_dir / "permanova_braycurtis.txt").open("w") as f:
        f.write(str(res) + "\n")

    # 7) 그룹 내 분산 체크
    dm_df = pd.DataFrame(dm.data, index=samples, columns=samples)
    with (out_dir / "dispersion_check.txt").open("w") as f:
        f.write("# Rough within-group mean pairwise distance\n")
        for g, sub in md_use.groupby("group"):
            ss = list(sub.index)
            mw = mean_within_group(dm_df, ss)
            f.write(f"{g}\tN={len(ss)}\tmean_within={mw:.6f}\n")

    # 8) 부트스트랩 타원 플롯
    plot_pcoa_boot(scores, ordn.proportion_explained, md_use, out_dir / "braycurtis_pcoa_plot.png",
                   n_boot=args.bootstrap)

    print(f"[OK] Beta 완료 → {out_dir}")

if __name__ == "__main__":
    main()
