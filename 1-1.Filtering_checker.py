#!/usr/bin/env python3
import argparse, re
import pandas as pd
from pathlib import Path

def read_report(path: Path):
    # Kraken2 --report-minimizer-data
    cols=["fraction","fragments","assigned","minimizers","uniq_minimizers","rank","taxid","name"]
    df = pd.read_csv(path, sep="\t", names=cols, dtype={"taxid":str}, engine="python")
    # keep only S / S* lines
    df = df[df["rank"].str.match(r"^S[0-9]*$", na=False)].copy()
    for c in ["minimizers","uniq_minimizers"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def read_inspect(path: Path):
    cols=["frac","minimizers_clade","minimizers_taxa","rank_i","taxid","sci_name"]
    df = pd.read_csv(path, sep="\t", names=cols, dtype={"taxid":str}, engine="python")
    df["minimizers_taxa"] = pd.to_numeric(df["minimizers_taxa"], errors="coerce")
    return df[["taxid","minimizers_taxa","sci_name"]]

def load_nodes(nodes: Path):
    c2p, t2r = {}, {}
    with nodes.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5: continue
            child, parent, rank = parts[0], parts[2], parts[4]
            c2p[child]=parent; t2r[child]=rank
    return c2p, t2r

def species_of(tid, c2p, t2r):
    node = tid
    while node and node != "1":
        if t2r.get(node) == "species":
            return node
        node = c2p.get(node, "1")
    return None

def compute_tp_species(orig_rep, inspect_df, nodes_path, cov_thr, min_thr, min_db_min):
    c2p, t2r = load_nodes(nodes_path)
    df = orig_rep.merge(inspect_df, on="taxid", how="left")
    df["coverage"] = (df["uniq_minimizers"] / df["minimizers_taxa"]).where(df["minimizers_taxa"]>=min_db_min)
    df["species_taxid"] = df["taxid"].apply(lambda x: species_of(x, c2p, t2r))
    strains = df[df["rank"].str.match(r"^S[0-9]+$", na=False)].copy()
    # strain that passes thresholds
    pass_mask = (strains["coverage"]>=cov_thr) & (strains["uniq_minimizers"]>=min_thr)
    tp_species = set(strains.loc[pass_mask,"species_taxid"].dropna().astype(str))
    # all strains per species in original
    strains_by_species = (strains.groupby("species_taxid")["taxid"]
                          .apply(lambda s: set(s.astype(str)))).to_dict()
    return tp_species, strains_by_species

def species_in_file(rep_df):
    # species lines only
    return set(rep_df.loc[rep_df["rank"]=="S","taxid"].astype(str))

def strains_by_species_in_file(rep_df, nodes_path):
    c2p, t2r = load_nodes(nodes_path)
    strains = rep_df[rep_df["rank"].str.match(r"^S[0-9]+$", na=False)].copy()
    strains["species_taxid"] = strains["taxid"].apply(lambda x: species_of(x, c2p, t2r))
    d = (strains.groupby("species_taxid")["taxid"]
         .apply(lambda s: set(s.astype(str)))).to_dict()
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_report", required=True, type=Path)
    ap.add_argument("--filtered_report", required=True, type=Path)
    ap.add_argument("--inspect", required=True, type=Path)
    ap.add_argument("--kraken_db", required=True, type=Path)
    ap.add_argument("--cov_thr", type=float, default=0.10)
    ap.add_argument("--min_thr", type=int, default=0)
    ap.add_argument("--min_db_min", type=int, default=5)
    args = ap.parse_args()

    nodes_path = args.kraken_db / "taxonomy" / "nodes.dmp"

    orig = read_report(args.original_report)
    filt = read_report(args.filtered_report)
    insp = read_inspect(args.inspect)

    tp_species_expected, strains_by_species_orig = compute_tp_species(
        orig, insp, nodes_path, args.cov_thr, args.min_thr, args.min_db_min
    )
    species_in_filtered = species_in_file(filt)
    strains_by_species_filtered = strains_by_species_in_file(filt, nodes_path)

    missing_species = sorted(tp_species_expected - species_in_filtered)
    extra_species   = sorted(species_in_filtered - tp_species_expected)

    print(f"[SUMMARY]")
    print(f"  TP species expected (from original): {len(tp_species_expected)}")
    print(f"  species present in filtered        : {len(species_in_filtered)}")
    print(f"  missing species in filtered        : {len(missing_species)}")
    print(f"  extra species in filtered          : {len(extra_species)}")

    if missing_species:
        print("\n[MISSING SPECIES]")
        print("\n".join(missing_species[:50]))
    if extra_species:
        print("\n[EXTRA SPECIES]")
        print("\n".join(extra_species[:50]))

    # For each TP species, all strains should be kept
    bad_species = []
    for sp in tp_species_expected:
        orig_set = strains_by_species_orig.get(sp, set())
        filt_set = strains_by_species_filtered.get(sp, set())
        if orig_set != filt_set:
            bad_species.append((sp, len(orig_set), len(filt_set)))
    print(f"\n[ALL-STRAINS CHECK] species with mismatched strain sets: {len(bad_species)}")
    if bad_species:
        print("  species_taxid\t#orig_strains\t#filtered_strains")
        for sp, no, nf in bad_species[:50]:
            print(f"  {sp}\t{no}\t{nf}")

if __name__ == "__main__":
    main()
