#!/usr/bin/env python3
"""Generate the small, simulated demo dataset (demo/data/vep_df.csv).

Deterministic (seeded), pure standard library — no numpy/pandas needed, so it
runs instantly anywhere. Output is a long-format table of (haplotype, site, VEP)
with planted additive wild-type-variant effects, consumed by run_demo.py.
"""
import csv
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    rng = random.Random(42)
    prot = "ENSP00000269305"
    # 8 background variants so a few dozen distinct haplotypes are possible
    wt_pool = ["12A>V", "48R>Q", "72P>R", "99S>N", "155T>I", "175H>Y", "213R>C", "248Q>R"]
    sites = [f"{prot}:175R>H", f"{prot}:248R>Q", f"{prot}:273R>C"]
    wt_eff = {w: rng.gauss(0, 0.25) for w in wt_pool}
    site_base = {s: rng.uniform(0.4, 0.6) for s in sites}

    # Draw a bounded number of random haplotypes (subsets of WT variants) and
    # de-duplicate. Bounded iteration guarantees termination regardless of how
    # many distinct haplotypes are combinatorially possible.
    seen = set()
    for _ in range(300):
        k = rng.randint(0, 3)
        chosen = tuple(sorted(rng.sample(wt_pool, k))) if k else ()
        hap = prot + ":" + (",".join(chosen) if chosen else "REF")
        seen.add(hap)

    rows = []
    for hap in sorted(seen):
        chosen = [] if hap.endswith(":REF") else hap.split(":", 1)[1].split(",")
        for s in sites:
            vep = site_base[s] + sum(wt_eff[w] for w in chosen) + rng.gauss(0, 0.03)
            vep = min(1.0, max(0.0, vep))
            rows.append({"haplotype": hap, "site": s, "VEP": round(vep, 6)})

    out_dir = os.path.join(HERE, "data")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "vep_df.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["haplotype", "site", "VEP"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}: {len(rows)} rows, {len(seen)} haplotypes, {len(sites)} sites")


if __name__ == "__main__":
    main()
