"""
Backfill ev.bbox for any extraction leaf that has block_id + page but no bbox.

For each extraction file:
  1. Open the corresponding marker JSON (papers_json/<paper_dir>/<paper_dir>.json)
  2. Walk every block, building a map block_id -> bbox (or polygon-derived bbox)
  3. For every {value, ev} leaf in the extraction with block_id but bbox is null,
     fill ev.bbox from the lookup
  4. Save the extraction back

Reports per-paper backfill counts and a final total.

Usage:
  python backfill_bboxes.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # project root
EXTR = HERE / "extractions"
JSON_ROOT = HERE / "papers_json"


def build_bbox_map(marker_data):
    """Walk a marker-pdf JSON tree, build {block_id: [x0, y0, x1, y1]}."""
    blocks = {}

    def to_bbox(node):
        bb = node.get("bbox")
        if isinstance(bb, list) and len(bb) == 4 and all(isinstance(x, (int, float)) for x in bb):
            return [float(x) for x in bb]
        poly = node.get("polygon")
        if isinstance(poly, list) and poly and all(isinstance(p, list) and len(p) >= 2 for p in poly):
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            return [min(xs), min(ys), max(xs), max(ys)]
        return None

    def walk(node):
        if isinstance(node, dict):
            bid = node.get("id")
            if bid:
                bb = to_bbox(node)
                if bb is not None:
                    blocks[bid] = bb
            for v in node.values():
                if isinstance(v, (list, dict)):
                    walk(v)
        elif isinstance(node, list):
            for x in node:
                walk(x)

    walk(marker_data)
    return blocks


def is_ev(node):
    return isinstance(node, dict) and "value" in node and "ev" in node


def backfill(node, bbox_map, stats):
    """Recurse, fill ev.bbox where missing."""
    if is_ev(node):
        ev = node.get("ev") or {}
        if not ev.get("bbox") and ev.get("block_id"):
            bb = bbox_map.get(ev["block_id"])
            if bb is not None:
                ev["bbox"] = bb
                stats["filled"] += 1
            else:
                stats["unmatched"] += 1
        return
    if isinstance(node, list):
        for x in node:
            backfill(x, bbox_map, stats)
    elif isinstance(node, dict):
        for v in node.values():
            backfill(v, bbox_map, stats)


def main():
    files = sorted(EXTR.glob("RP*_extraction.json"))
    grand = {"filled": 0, "unmatched": 0}
    print(f"{'RP':<6} {'filled':>7} {'unmatched':>10} {'paper_dir'}")
    print("-" * 70)

    for fp in files:
        ext = json.loads(fp.read_text(encoding="utf-8"))
        rp_id = ext.get("rp_id") or fp.stem.split("_")[0]

        # Locate the matching marker JSON
        candidates = list(JSON_ROOT.glob(f"{rp_id}_*"))
        if not candidates:
            print(f"{rp_id:<6} {'-':>7} {'-':>10} (no papers_json folder)")
            continue
        paper_dir = candidates[0]
        marker_file = paper_dir / f"{paper_dir.name}.json"
        if not marker_file.exists():
            print(f"{rp_id:<6} {'-':>7} {'-':>10} (no marker JSON in {paper_dir.name})")
            continue

        marker = json.loads(marker_file.read_text(encoding="utf-8"))
        bbox_map = build_bbox_map(marker)

        stats = {"filled": 0, "unmatched": 0}
        backfill(ext, bbox_map, stats)

        if stats["filled"] > 0:
            fp.write_text(json.dumps(ext, indent=2, ensure_ascii=False), encoding="utf-8")

        grand["filled"] += stats["filled"]
        grand["unmatched"] += stats["unmatched"]
        print(f"{rp_id:<6} {stats['filled']:>7} {stats['unmatched']:>10} {paper_dir.name}")

    print("-" * 70)
    print(f"TOTAL  {grand['filled']:>7} {grand['unmatched']:>10}")


if __name__ == "__main__":
    main()
