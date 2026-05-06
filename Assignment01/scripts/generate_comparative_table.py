"""
Derive the assignment's required Comparative Analysis Table from extraction JSONs.

Reads:
  extractions/RP*_extraction.json

Writes:
  comparative_analysis_table.csv  (overwrites the existing one)

Columns (per A1 §3 requirements + Dr Manar verbal additions):
  RP_ID, Year, Author, Title, Venue, DOI, Contribution, Dataset, Data Size,
  Models Used, Evaluation Metrics, Key Results, Strengths, Limitations

Dr Manar (verbal, 2026): the table must support Section 4 critical analysis,
so Strengths and Limitations are explicit columns and Key Results carries the
actual reported numerical values, not just metric names. Limitations must be
*author-reported* (taken from each paper's own acknowledgments), not our
gap-analysis framing — we therefore source the Limitations column from the
extraction's `weaknesses` array (author-acknowledged with direct quotes), not
the `limitations` array (our scope/coverage framing used elsewhere for gap
analysis).

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


def headline_metrics_inline(ext, max_count=4):
    """Build a compact 'Metric=value (model bit-width dataset)' string from
    metrics_results entries marked headline=true. Caps at max_count entries.
    Falls back to plain eval_metrics names if no headline values are available."""
    entries = ext.get("metrics_results") or []
    headlines = [e for e in entries
                 if isinstance(e.get("value"), dict)
                 and e["value"].get("headline") is True]
    if not headlines:
        return join_values(ext.get("eval_metrics") or [])
    parts = []
    for e in headlines[:max_count]:
        v_obj = e["value"]
        # "Metric=value (Model bit-width? Dataset?)"
        ctx_bits = []
        if v_obj.get("model"): ctx_bits.append(str(v_obj["model"]))
        if v_obj.get("bit_width"): ctx_bits.append(str(v_obj["bit_width"]))
        if v_obj.get("dataset"): ctx_bits.append(str(v_obj["dataset"]))
        ctx = " ".join(ctx_bits)
        metric = v_obj.get("metric") or "?"
        val = v_obj.get("value")
        s = f"{metric}={val}"
        if ctx:
            s += f" ({ctx})"
        parts.append(s)
    out = "; ".join(parts)
    # Append metric names that aren't already covered, so the column still lists scope.
    names_in_headlines = {h["value"].get("metric", "").lower() for h in headlines[:max_count]}
    other_names = [str(v(m)) for m in (ext.get("eval_metrics") or [])
                   if str(v(m)).lower() not in names_in_headlines]
    if other_names:
        out += " | other metrics: " + "; ".join(other_names[:5])
    return out


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
    # Evaluation Metrics column now embeds the paper's headline values inline,
    # per Dr Manar's "actual reported values" feedback. Falls back to plain
    # metric names if metrics_results is absent (legacy data).
    metrics = headline_metrics_inline(ext, max_count=4)
    key_results = join_values(ext.get("key_results") or [], sep=" | ")
    strengths = join_values(ext.get("strengths") or [], sep=" | ")
    # Use weaknesses (author-acknowledged) for the Limitations column;
    # the limitations array contains our gap-analysis framing.
    limitations = join_values(ext.get("weaknesses") or [], sep=" | ")
    return [rp, year, first_author, title, venue, doi, contribution,
            datasets, data_size, models, metrics, key_results,
            strengths, limitations]


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
              "Dataset", "Data Size", "Models Used", "Evaluation Metrics", "Key Results",
              "Strengths", "Limitations"]
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Wrote {OUT.name}: {len(rows)} rows · {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
