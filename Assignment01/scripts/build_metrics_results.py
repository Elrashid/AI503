"""
Build the flat metrics_results dataset from merged RP*_extraction.json files.

Reads:
  extractions/RP*_extraction.json  (must already have `metrics_results` array
                                    after running merge_metrics_into_extractions.py)

Writes:
  metrics_results.csv          — single flat dataset, all papers (long format)
  metrics_results.csv.gz       — gzip-compressed for fast download
  metrics_results/RP{NN}.csv   — per-paper shards, each renderable as a sortable
                                 spreadsheet on github.com (under 512 KB)

Columns:
  rp_id, model, method, bit_width, group_size, dataset, metric, value,
  baseline_label, baseline_value, delta, delta_unit, headline, source_kind,
  page, section, source_table, quote
"""

import csv
import gzip
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # v4/
EXTR = HERE / "extractions"
OUT_FLAT = HERE / "metrics_results.csv"
OUT_GZ = HERE / "metrics_results.csv.gz"
OUT_SHARDS = HERE / "metrics_results"

VALUE_COLS = [
    "rp_id", "model", "method", "bit_width", "group_size",
    "dataset", "metric", "value",
    "baseline_label", "baseline_value", "delta", "delta_unit",
    "headline", "source_kind",
]
EV_COLS = ["page", "section", "source_table", "quote"]
HEADER = VALUE_COLS + EV_COLS


def row_for(rp_id: str, entry: dict) -> list:
    v = entry.get("value", {})
    ev = entry.get("ev", {})
    quote = (ev.get("quote") or "").replace("\n", " ").replace("\r", " ").strip()
    return [
        rp_id,
        v.get("model"),
        v.get("method"),
        v.get("bit_width"),
        v.get("group_size"),
        v.get("dataset"),
        v.get("metric"),
        v.get("value"),
        v.get("baseline_label"),
        v.get("baseline_value"),
        v.get("delta"),
        v.get("delta_unit"),
        v.get("headline"),
        v.get("source_kind"),
        ev.get("page"),
        ev.get("section"),
        ev.get("source_table"),
        quote,
    ]


def main():
    OUT_SHARDS.mkdir(exist_ok=True)

    files = sorted(EXTR.glob("RP*_extraction.json"))
    flat_rows = []
    paper_count = 0
    paper_with_metrics = 0

    for fp in files:
        ext = json.loads(fp.read_text(encoding="utf-8"))
        rp_id = ext.get("rp_id") or fp.stem.replace("_extraction", "")
        paper_count += 1

        entries = ext.get("metrics_results") or []
        if not entries:
            continue
        paper_with_metrics += 1

        # Per-paper shard
        shard_path = OUT_SHARDS / f"{rp_id}.csv"
        with shard_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            w.writerow(HEADER)
            for e in entries:
                row = row_for(rp_id, e)
                w.writerow(row)
                flat_rows.append(row)
        print(f"  {rp_id}: {len(entries)} entries -> metrics_results/{rp_id}.csv")

    # Flat CSV
    with OUT_FLAT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(HEADER)
        w.writerows(flat_rows)

    # Gzip-compressed flat CSV
    with OUT_FLAT.open("rb") as src, gzip.open(OUT_GZ, "wb", compresslevel=9) as dst:
        dst.write(src.read())

    flat_kb = OUT_FLAT.stat().st_size / 1024
    gz_kb = OUT_GZ.stat().st_size / 1024
    print(f"\nDone.")
    print(f"  Papers scanned: {paper_count}, with metrics_results: {paper_with_metrics}")
    print(f"  Total rows: {len(flat_rows)}")
    print(f"  metrics_results.csv: {flat_kb:.1f} KB")
    print(f"  metrics_results.csv.gz: {gz_kb:.1f} KB")
    print(f"  Per-paper shards: {paper_with_metrics} files in metrics_results/")


if __name__ == "__main__":
    main()
