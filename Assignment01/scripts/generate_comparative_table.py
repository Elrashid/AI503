"""
Derive the assignment's required Comparative Analysis Table from extraction JSONs.

Reads:
  extractions/RP*_extraction.json

Writes:
  comparative_analysis_table.csv  (overwrites the existing one)

Columns (per A1 §3 requirements):
  RP_ID, Year, Author, Title, Venue, DOI, Contribution, Dataset, Data Size,
  Models Used, Evaluation Metrics, Key Results

Each cell strips evidence wrappers and joins multi-valued fields with "; ".
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # project root (parent of scripts/)
EXTR = HERE / "extractions"
OUT = HERE / "comparative_analysis_table.csv"


def v(node):
    """Unwrap {value, ev} -> value. Pass through scalars."""
    if isinstance(node, dict) and "value" in node and "ev" in node:
        return node["value"]
    return node


def join_values(arr, sep="; ", limit=None):
    """Join an array of {value, ev} objects (or scalars) into a single string."""
    if not arr:
        return ""
    items = []
    for x in arr:
        val = v(x)
        if val is None:
            continue
        items.append(str(val).strip())
    if limit:
        items = items[:limit]
    return sep.join(items)


def row_for(ext):
    rp = ext.get("rp_id", "")
    year = v(ext.get("year")) or ""
    first_author = v(ext.get("first_author")) or ""
    title = v(ext.get("title")) or ""
    venue = v(ext.get("venue")) or ""
    doi = v(ext.get("doi")) or ""
    contribution = v(ext.get("contribution")) or ""
    datasets = join_values(ext.get("datasets") or [])
    data_size = v(ext.get("data_size")) or ""
    models = join_values(ext.get("models_used") or [])
    metrics = join_values(ext.get("eval_metrics") or [])
    key_results = join_values(ext.get("key_results") or [], sep=" | ")
    return [rp, year, first_author, title, venue, doi, contribution,
            datasets, data_size, models, metrics, key_results]


def main():
    files = sorted(EXTR.glob("RP*_extraction.json"))
    rows = []
    for fp in files:
        try:
            ext = json.loads(fp.read_text(encoding="utf-8"))
            rows.append(row_for(ext))
        except Exception as e:
            print(f"WARN {fp.name}: {e}")

    header = ["RP_ID", "Year", "Author", "Title", "Venue", "DOI", "Contribution",
              "Dataset", "Data Size", "Models Used", "Evaluation Metrics", "Key Results"]
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Wrote {OUT.name}: {len(rows)} rows · {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
