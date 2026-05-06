"""
Merge per-paper RP{NN}_metrics.json into RP{NN}_extraction.json.

Reads:
  extractions/RP{NN}_metrics.json   (transient, written by extract subagents)

Writes (in place):
  extractions/RP{NN}_extraction.json  (gains a top-level `metrics_results` array)

Filtering policy (CRITICAL):
- Only entries with source_kind == "measured" are merged into the extraction.
- "cited" entries (numbers the paper imported from prior work, e.g. AdaRound
  baselines on ImageNet in the GPTQ paper) are dropped during merge.
- Rationale: the published comparative dataset should reflect each paper's
  own contribution, not values it inherited from elsewhere. Cited values
  remain in the transient RP{NN}_metrics.json for audit purposes but are not
  part of the canonical extraction.

Other behaviour:
- Adds a `_metrics_meta` block recording schema_version, kept count, and the
  count of dropped (cited) entries for audit.
- Idempotent: rerunning replaces the existing metrics_results array.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # v4/
EXTR = HERE / "extractions"

SCHEMA_VERSION = "a1.v3"


def merge_one(rp_id: str) -> tuple[bool, str]:
    metrics_path = EXTR / f"{rp_id}_metrics.json"
    extraction_path = EXTR / f"{rp_id}_extraction.json"

    if not metrics_path.exists():
        return False, f"  SKIP {rp_id}: no metrics file"
    if not extraction_path.exists():
        return False, f"  SKIP {rp_id}: no extraction file"

    metrics_doc = json.loads(metrics_path.read_text(encoding="utf-8"))
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))

    all_entries = metrics_doc.get("metrics_results", [])

    # Filter: only source_kind == "measured" survives the merge.
    measured = [e for e in all_entries
                if isinstance(e.get("value"), dict)
                and e["value"].get("source_kind") == "measured"]
    cited_dropped = [e for e in all_entries
                     if isinstance(e.get("value"), dict)
                     and e["value"].get("source_kind") == "cited"]
    other_dropped = [e for e in all_entries
                     if not isinstance(e.get("value"), dict)
                     or e["value"].get("source_kind") not in ("measured", "cited")]

    extraction["metrics_results"] = measured
    extraction["_metrics_meta"] = {
        "schema_version": SCHEMA_VERSION,
        "merge_filter": "source_kind == 'measured'",
        "kept_count": len(measured),
        "dropped_cited_count": len(cited_dropped),
        "dropped_other_count": len(other_dropped),
        "headline_count": sum(1 for e in measured
                              if isinstance(e.get("value"), dict)
                              and e["value"].get("headline") is True),
    }

    extraction_path.write_text(
        json.dumps(extraction, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    meta = extraction["_metrics_meta"]
    return True, (f"  OK   {rp_id}: kept {meta['kept_count']} measured "
                  f"(dropped {meta['dropped_cited_count']} cited, "
                  f"{meta['dropped_other_count']} other; "
                  f"headline={meta['headline_count']})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rp", action="append", default=None,
                        help="Specific RP IDs to merge (e.g. --rp RP09 --rp RP15). "
                             "If omitted, merge every RP*_metrics.json found.")
    args = parser.parse_args()

    if args.rp:
        rp_ids = args.rp
    else:
        rp_ids = sorted(p.stem.replace("_metrics", "")
                        for p in EXTR.glob("RP*_metrics.json"))

    if not rp_ids:
        print("No metrics files to merge.")
        sys.exit(0)

    print(f"Merging {len(rp_ids)} paper(s) — keeping only source_kind='measured':")
    ok = 0
    total_kept = 0
    total_dropped = 0
    for rp in rp_ids:
        success, msg = merge_one(rp)
        print(msg)
        if success:
            ok += 1
            # Re-read to tally
            ext = json.loads((EXTR / f"{rp}_extraction.json").read_text(encoding="utf-8"))
            meta = ext.get("_metrics_meta", {})
            total_kept += meta.get("kept_count", 0)
            total_dropped += meta.get("dropped_cited_count", 0)

    print(f"\nDone. {ok}/{len(rp_ids)} merged. "
          f"Total kept (measured): {total_kept}. Total dropped (cited): {total_dropped}.")


if __name__ == "__main__":
    main()
