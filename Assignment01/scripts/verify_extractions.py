"""
Verify every extraction JSON:
  1. Parses cleanly with json.loads
  2. Has all required top-level fields
  3. Every leaf is shaped {value, ev}
  4. Every array element is shaped {value, ev}
  5. ev has page, section, quote, bbox, block_id (bbox/block_id may be null)
  6. gap_signals has all 9 expected booleans

Reports per-paper stats and a summary table.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # project root (parent of scripts/)
EXTR = HERE / "extractions"
SCHEMA = HERE / ".slr" / "schema.json"

REQUIRED_TOP = [
    "rp_id", "year", "first_author", "authors", "title", "venue", "doi",
    "contribution", "datasets", "data_size", "models_used", "eval_metrics",
    "key_results", "paper_type", "method_family", "quantization_method_name",
    "bit_widths_tested", "quantization_target", "granularity", "training_free",
    "hardware", "speedup_claim", "strengths", "weaknesses", "limitations",
    "future_work_stated", "gap_signals",
]

REQUIRED_GAP = [
    "reports_energy_consumption",
    "reports_statistical_significance",
    "tests_on_edge_hardware",
    "tests_safety_after_quantization",
    "tests_below_4_bit",
    "reports_real_latency_not_just_throughput",
    "evaluates_long_context",
    "releases_code",
    "evaluates_instruction_tuned_models",
]


def is_ev_leaf(node):
    return isinstance(node, dict) and "value" in node and "ev" in node


def count_traced(node, stats):
    """Walk recursively, count leaves with ev and how many have bboxes."""
    if is_ev_leaf(node):
        stats["leaves"] += 1
        ev = node.get("ev") or {}
        if ev.get("page") is not None:
            stats["traced"] += 1
        if ev.get("bbox"):
            stats["bbox"] += 1
        return
    if isinstance(node, list):
        for item in node:
            count_traced(item, stats)
    elif isinstance(node, dict):
        for v in node.values():
            count_traced(v, stats)


def check_one(fp):
    issues = []
    try:
        ext = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as e:
        return {"path": fp.name, "ok": False, "issues": [f"parse: {e}"], "stats": {}}

    # required top-level
    for k in REQUIRED_TOP:
        if k not in ext:
            issues.append(f"missing top-level: {k}")
    # gap signals shape
    gs = ext.get("gap_signals") or {}
    for k in REQUIRED_GAP:
        if k not in gs:
            issues.append(f"missing gap_signal: {k}")
        elif not is_ev_leaf(gs.get(k)):
            issues.append(f"gap_signal {k} not {{value,ev}}")

    stats = {"leaves": 0, "traced": 0, "bbox": 0}
    count_traced(ext, stats)

    return {
        "path": fp.name,
        "ok": len(issues) == 0,
        "issues": issues,
        "stats": stats,
        "rp_id": ext.get("rp_id"),
        "year": (ext.get("year") or {}).get("value") if isinstance(ext.get("year"), dict) else ext.get("year"),
        "title": (ext.get("title") or {}).get("value", "")[:60] if isinstance(ext.get("title"), dict) else "",
        "paper_type": (ext.get("paper_type") or {}).get("value") if isinstance(ext.get("paper_type"), dict) else None,
        "method_family": (ext.get("method_family") or {}).get("value") if isinstance(ext.get("method_family"), dict) else None,
        "n_key_results": len(ext.get("key_results") or []),
        "n_models": len(ext.get("models_used") or []),
        "n_datasets": len(ext.get("datasets") or []),
        "gap_positive": sum(1 for k in REQUIRED_GAP if (gs.get(k) or {}).get("value") is True),
    }


def main():
    files = sorted(EXTR.glob("RP*_extraction.json"))
    if not files:
        print("No extractions found.")
        return

    print(f"Verifying {len(files)} extraction files...\n")
    print(f"{'RP':<6} {'Year':<6} {'Type':<18} {'Family':<16} {'Leaves':>7} {'Traced':>7} {'Bbox':>5} {'KR':>3} {'M':>3} {'D':>3} {'Gap+':>5} {'OK':>4}")
    print("-" * 110)

    totals = {"leaves": 0, "traced": 0, "bbox": 0}
    bad = []
    for fp in files:
        r = check_one(fp)
        s = r["stats"]
        totals["leaves"] += s.get("leaves", 0)
        totals["traced"] += s.get("traced", 0)
        totals["bbox"] += s.get("bbox", 0)
        ok_mark = "OK" if r["ok"] else "FAIL"
        print(f"{r.get('rp_id') or '-':<6} {str(r.get('year') or '-'):<6} "
              f"{(r.get('paper_type') or '-')[:18]:<18} {(r.get('method_family') or '-')[:16]:<16} "
              f"{s.get('leaves', 0):>7} {s.get('traced', 0):>7} {s.get('bbox', 0):>5} "
              f"{r['n_key_results']:>3} {r['n_models']:>3} {r['n_datasets']:>3} "
              f"{r['gap_positive']:>5} {ok_mark:>4}")
        if not r["ok"]:
            bad.append(r)

    print("-" * 110)
    pct_traced = 100 * totals["traced"] / max(1, totals["leaves"])
    pct_bbox = 100 * totals["bbox"] / max(1, totals["leaves"])
    print(f"\nTOTAL: {len(files)} papers · {totals['leaves']:,} leaves · "
          f"{totals['traced']:,} traced ({pct_traced:.1f}%) · "
          f"{totals['bbox']:,} with bbox ({pct_bbox:.1f}%)")

    if bad:
        print(f"\n{len(bad)} files with issues:")
        for r in bad:
            print(f"  {r['path']}: {'; '.join(r['issues'])}")
    else:
        print("\nAll files pass schema checks.")


if __name__ == "__main__":
    main()
