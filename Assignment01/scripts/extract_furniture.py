"""
Recover the page-header / page-footer text that marker-pdf intentionally strips.

For each paper:
  1. Walk the marker JSON, collect every PageHeader / PageFooter block
     (these have correct bboxes but empty html).
  2. Use PyMuPDF to extract the actual text inside each bbox from the source PDF.
  3. Write a sidecar at  papers_pages/<paper_dir>/furniture.json  containing the
     recovered text + bbox + block_id, keyed by page.

The sidecar lets the explorer (and any future patcher) highlight year/venue
metadata that lives only in the footer of ACL/EMNLP-style proceedings papers.

Usage:
  python extract_furniture.py
"""

import json
from pathlib import Path

import fitz  # PyMuPDF

HERE = Path(__file__).resolve().parent.parent  # project root (parent of scripts/)
JSON_ROOT = HERE / "papers_json"
PDF_ROOT = HERE / "papers_pdf"
PAGES_ROOT = HERE / "papers_pages"

WANTED = ("PageHeader", "PageFooter")


def collect_furniture_blocks(marker_data):
    """Walk marker JSON, yield (page_idx, kind, block_id, bbox)."""
    pages = marker_data.get("children") or []
    for page_idx, page in enumerate(pages):
        for blk in (page.get("children") or []):
            bt = blk.get("block_type")
            if bt in WANTED:
                bbox = blk.get("bbox")
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    poly = blk.get("polygon") or []
                    if poly and all(isinstance(p, list) and len(p) >= 2 for p in poly):
                        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                        bbox = [min(xs), min(ys), max(xs), max(ys)]
                if not bbox:
                    continue
                yield page_idx + 1, bt, blk.get("id"), bbox


def extract_text_for_paper(paper_dir):
    pdf_path = PDF_ROOT / f"{paper_dir.name}.pdf"
    json_path = paper_dir / f"{paper_dir.name}.json"
    if not pdf_path.exists() or not json_path.exists():
        return None, "missing pdf or json"

    marker = json.loads(json_path.read_text(encoding="utf-8"))
    blocks = list(collect_furniture_blocks(marker))
    if not blocks:
        return {"rp_id": paper_dir.name.split("_")[0], "paper_dir": paper_dir.name,
                "pages": []}, None

    doc = fitz.open(pdf_path)
    by_page = {}
    pad = 1.0  # tiny pad to catch glyphs on the edge
    for page_num, kind, bid, bbox in blocks:
        try:
            page = doc[page_num - 1]
            x0, y0, x1, y1 = bbox
            rect = fitz.Rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad)
            text = page.get_text("text", clip=rect).strip()
        except Exception as e:
            text = ""
        entry = {"block_id": bid, "bbox": bbox, "text": text}
        slot = by_page.setdefault(page_num, {"page": page_num, "header": [], "footer": []})
        slot["header" if kind == "PageHeader" else "footer"].append(entry)
    doc.close()

    sidecar = {
        "rp_id": paper_dir.name.split("_")[0],
        "paper_dir": paper_dir.name,
        "pdf_file": pdf_path.name,
        "pages": [by_page[k] for k in sorted(by_page)],
    }
    return sidecar, None


def main():
    stats = {"papers": 0, "header_blocks": 0, "header_recovered": 0,
             "footer_blocks": 0, "footer_recovered": 0, "errors": 0}
    for d in sorted(JSON_ROOT.iterdir()):
        if not d.is_dir():
            continue
        sidecar, err = extract_text_for_paper(d)
        if err:
            print(f"  {d.name}: {err}")
            stats["errors"] += 1
            continue
        out_dir = PAGES_ROOT / d.name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "furniture.json").write_text(
            json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        ph = sum(len(p["header"]) for p in sidecar["pages"])
        pf = sum(len(p["footer"]) for p in sidecar["pages"])
        ph_ok = sum(1 for p in sidecar["pages"] for h in p["header"] if h["text"])
        pf_ok = sum(1 for p in sidecar["pages"] for f in p["footer"] if f["text"])
        stats["papers"] += 1
        stats["header_blocks"] += ph
        stats["header_recovered"] += ph_ok
        stats["footer_blocks"] += pf
        stats["footer_recovered"] += pf_ok
        print(f"  {d.name}: header {ph_ok}/{ph}  footer {pf_ok}/{pf}")

    print()
    print("=== TOTALS ===")
    print(f"  Papers processed:  {stats['papers']}")
    print(f"  Header blocks:     {stats['header_recovered']:>5} recovered / {stats['header_blocks']} total"
          f"   ({100*stats['header_recovered']/max(1,stats['header_blocks']):.1f}%)")
    print(f"  Footer blocks:     {stats['footer_recovered']:>5} recovered / {stats['footer_blocks']} total"
          f"   ({100*stats['footer_recovered']/max(1,stats['footer_blocks']):.1f}%)")
    if stats["errors"]:
        print(f"  Errors: {stats['errors']}")


if __name__ == "__main__":
    main()
