"""Per-cell audit driver: build_table → attribute → markdown.

Usage:
  python -m diagnostics.failure_audit.run_audit
"""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

from .build_table import build_cell_table
from .attribute  import attribute_table


REPO = Path(__file__).resolve().parents[2]

# Corrected 2026-05-14: prior coverage_recon was wrong (loader inverted
# obj_id/frame in cache schema). All 3 cells have 100% iKUN coverage —
# attribution must run on all of them.
CELLS = [
    ("turning-cars",                  "0011"),
    ("turning-vehicles",              "0011"),
    ("pedestrian-who-are-walking",    "0011"),
]


def summarize(df: pd.DataFrame) -> dict:
    counts = df["failure_class"].value_counts().to_dict()
    n = int(len(df))
    return {
        "n_rows": n,
        "class_counts": counts,
        "class_pct": {k: round(100.0 * v / n, 1) for k, v in counts.items()},
    }


def main():
    out_dir = REPO / "diagnostics" / "results" / "failure_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for expr, seq in CELLS:
        df = build_cell_table(REPO, seq, expr)
        df = attribute_table(df)
        df.to_csv(out_dir / f"audit_{expr}_{seq}.csv", index=False)
        s = summarize(df)
        s["expr"], s["seq"] = expr, seq
        summaries.append(s)

    md = ["# Failure-Mode Audit — Per-row Attribution",
          "",
          "Cells included: only those with non-zero iKUN coverage (recon §coverage).",
          ""]
    for s in summaries:
        md += [f"## {s['expr']} × {s['seq']}",
               "",
               f"- n_gt_rows = {s['n_rows']}",
               ""]
        md.append("| class | n | pct |")
        md.append("|---|---:|---:|")
        for cls in ["FN_ikun_coverage","FN_detector","FN_tracker",
                    "FN_aligner","FN_fusion","TP","FP","TN"]:
            n = s["class_counts"].get(cls, 0)
            p = s["class_pct"].get(cls, 0.0)
            md.append(f"| {cls} | {n} | {p}% |")
        md.append("")
    (out_dir / "attribution.md").write_text("\n".join(md))
    (out_dir / "attribution.json").write_text(json.dumps(summaries, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
