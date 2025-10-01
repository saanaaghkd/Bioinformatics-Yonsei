
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Tuple, Set, List, Optional

import pandas as pd
import numpy as np

taxid = str

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Make a trimmed Kraken2 report that contains only true-positive species/strains."
    )
    p.add_argument("--kraken_report", type=Path, required=True,
                   help="Path to a Kraken2 report OR a directory of reports.")
    p.add_argument("--kraken_db", type=Path, required=True,
                   help="Kraken2 DB dir (expects taxonomy/nodes.dmp).")
    p.add_argument("--inspect", type=Path, required=True,
                   help="kraken2-inspect output (e.g., UHGV_inspect.out).")
    p.add_argument("--cov_thr", type=float, default=0.10,
                   help="Coverage threshold (default: 0.10).")
    p.add_argument("--min_thr", type=int, default=0,
                   help="Minimum distinct minimizers in reads (default: 0).")
    p.add_argument("--min_db_min", type=int, default=5,
                   help="Ignore DB totals < this when computing coverage (default: 5).")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Output directory (default: <report_dir>/truepositives_YYYYMMDD).")
    p.add_argument("--drop-upper", action="store_true",
                   help="If set, drop non-S ranks (keep only S/S* lines).")
    return p.parse_args()

def load_taxonomy(nodes_file: Path) -> Tuple[Dict[taxid, taxid], Dict[taxid, str]]:
    child_to_parent: Dict[taxid, taxid] = {}
    taxid_to_rank: Dict[taxid, str] = {}
    with nodes_file.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            child, parent, rank = parts[0], parts[2], parts[4]
            child_to_parent[child] = parent
            taxid_to_rank[child] = rank
    return child_to_parent, taxid_to_rank

def find_lineage_taxid(tid: taxid, desired_rank: str,
                       child_parent: Dict[taxid, taxid], taxid_rank: Dict[taxid, str]) -> taxid:
    node = tid
    while node and node != "1":
        if taxid_rank.get(node) == desired_rank:
            return node
        node = child_parent.get(node, "1")
    return "unclassified"

def load_and_compute(report_path: Path, inspect_path: Path, nodes,
                     min_db_min: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    child_parent, taxid_rank = nodes

    df_report = pd.read_csv(
        report_path, sep="\t",
        names=["fraction","fragments","assigned","minimizers","uniq_minimizers","rank","taxid","name"],
        dtype={"taxid": str}, engine="python",
    )
    df_inspect = pd.read_csv(
        inspect_path, sep="\t",
        names=["frac_i","minimizers_clade","minimizers_taxa","rank_i","taxid_i","sci_name_i"],
        dtype={"taxid_i": str}, engine="python",
    )

    df = df_report.merge(df_inspect, left_on="taxid", right_on="taxid_i", how="inner")
    df["minimizers_taxa"]   = pd.to_numeric(df["minimizers_taxa"], errors="coerce")
    df["uniq_minimizers"]   = pd.to_numeric(df["uniq_minimizers"], errors="coerce")
    df["coverage"] = np.where(
        df["minimizers_taxa"] >= min_db_min,
        df["uniq_minimizers"] / df["minimizers_taxa"],
        np.nan,
    )

    df_species = df[df["rank"] == "S"].copy()
    df_strain  = df[df["rank"].str.match(r"^S[0-9]+$", na=False)].copy()

    for sub in (df_species, df_strain):
        if not sub.empty:
            sub["species_taxid"] = sub["taxid"].apply(
                lambda x: find_lineage_taxid(x, "species", child_parent, taxid_rank)
            )

    return df_species, df_strain

def call_tp(df_species: pd.DataFrame, df_strain: pd.DataFrame,
            cov_thr: float, min_thr: int) -> Tuple[Set[taxid], Set[taxid]]:
    tp_strains_df = df_strain.loc[
        (df_strain["coverage"] >= cov_thr) & (df_strain["uniq_minimizers"] >= min_thr)
    ]
    keep_species: Set[taxid] = set(tp_strains_df["species_taxid"].dropna().astype(str))
    keep_strains: Set[taxid] = set(tp_strains_df["taxid"].astype(str))  # (필터링엔 사용하지 않지만 보관)
    return keep_species, keep_strains

def write_trimmed_report(original_report: Path, out_report: Path,
                         keep_species: Set[taxid], keep_strains: Set[taxid],
                         nodes, drop_upper: bool) -> None:
    child_parent, taxid_rank = nodes
    with original_report.open(encoding="utf-8") as infile, out_report.open("w", encoding="utf-8") as outfile:
        for line in infile:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 7:
                continue
            rank = cols[5]; tx = cols[6]
            if rank == "S":
                sp = find_lineage_taxid(tx, "species", child_parent, taxid_rank)
                if sp in keep_species:
                    outfile.write(line)
            elif re.match(r"^S[0-9]+$", rank):
                # ✅ 변경: strain 자체가 TP인지 보지 않고, 부모 species가 TP면 포함
                sp = find_lineage_taxid(tx, "species", child_parent, taxid_rank)
                if sp in keep_species:
                    outfile.write(line)
            else:
                if not drop_upper:
                    outfile.write(line)

def find_reports(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    files: List[Path] = []
    for patt in ("*.report", "*report*.txt", "*_report.txt"):
        files.extend(path.glob(patt))
    return sorted(files)

def ensure_outdir(base: Path, explicit: Optional[Path]) -> Path:
    out = explicit if explicit is not None else base / f"truepositives_{pd.Timestamp.now().strftime('%Y%m%d')}"
    out.mkdir(parents=True, exist_ok=True)
    return out

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    nodes_path = args.kraken_db / "taxonomy" / "nodes.dmp"
    if not nodes_path.exists():
        raise FileNotFoundError(f"Cannot find taxonomy nodes at: {nodes_path}")
    nodes = load_taxonomy(nodes_path)

    reports = find_reports(args.kraken_report)
    if not reports:
        raise FileNotFoundError(f"No report files found under: {args.kraken_report}")

    base_dir = args.kraken_report.parent if args.kraken_report.is_file() else args.kraken_report
    out_dir  = ensure_outdir(base_dir, args.out_dir)
    logging.info(f"Output directory: {out_dir}")

    for rep in reports:
        sample = rep.stem
        logging.info(f"Processing: {rep.name}")

        df_sp, df_st = load_and_compute(rep, args.inspect, nodes, args.min_db_min)
        keep_species, keep_strains = call_tp(df_sp, df_st, args.cov_thr, args.min_thr)

        trimmed_path = out_dir / f"{sample}.report.truepositives.txt"
        write_trimmed_report(rep, trimmed_path, keep_species, keep_strains, nodes, args.drop_upper)

        logging.info(f"✓ Wrote: {trimmed_path.name}")
        if not keep_species:
            logging.warning("No true-positive species found for this sample with current thresholds.")

    logging.info("Done.")

if __name__ == "__main__":
    main()

