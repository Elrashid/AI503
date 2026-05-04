"""
Use the recovered page-furniture text to repair year (and venue when possible)
ev fields that were marked not_found.

For each extraction:
  - If year.value is set but ev.section in ("not_found", absence-style),
    search the paper's furniture.json for a footer block whose text contains
    the year as a 4-digit number AND has a "Proceedings" / "©" / journal
    keyword nearby. If found, populate ev with page/section/quote/bbox/block_id
    pointing to that footer.
  - Same for venue if its value is set and ev.section is not_found AND we can
    confirm the venue substring appears in the same footer text.

Conservative — only patches when we can match value to recovered text.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # project root (parent of scripts/)
EXTR = HERE / "extractions"
PAGES = HERE / "papers_pages"

PROCEEDINGS_HINTS = re.compile(
    r"(proceedings|conference|workshop|symposium|©|c⃝|copyright|annual meeting|journal of)",
    re.IGNORECASE,
)


def find_year_block(furniture, year_str):
    """Return (page, block_dict) for the first footer/header block whose text
    contains year_str AND a proceedings/copyright cue. Prefer footer over header."""
    candidates_footer = []
    candidates_header = []
    for page in furniture.get("pages", []):
        for f in page.get("footer", []):
            if year_str in (f.get("text") or "") and PROCEEDINGS_HINTS.search(f["text"]):
                candidates_footer.append((page["page"], f, "Proceedings footer"))
        for h in page.get("header", []):
            if year_str in (h.get("text") or "") and PROCEEDINGS_HINTS.search(h["text"]):
                candidates_header.append((page["page"], h, "Page header"))
    if candidates_footer:
        return candidates_footer[0]
    if candidates_header:
        return candidates_header[0]
    return None


def venue_in_block(text, venue):
    if not text or not venue:
        return False
    # Compare token-wise — venue often has expanded form in footer
    venue_lower = venue.lower()
    txt_lower = text.lower()
    if venue_lower in txt_lower:
        return True
    # try acronym vs expansion (e.g. "ACL" should match "Association for Computational Linguistics")
    EXPANSIONS = {
        "acl": "association for computational linguistics",
        "emnlp": "empirical methods in natural language processing",
        "naacl": "north american chapter of the association for computational linguistics",
        "neurips": "neural information processing systems",
        "nips": "neural information processing systems",
        "icml": "international conference on machine learning",
        "iclr": "international conference on learning representations",
        "aaai": "association for the advancement of artificial intelligence",
        "ijcai": "international joint conference on artificial intelligence",
    }
    for short, long in EXPANSIONS.items():
        if (short in venue_lower and long in txt_lower) or (long in venue_lower and short in txt_lower):
            return True
    # token overlap >= 50%
    venue_tokens = set(re.findall(r"\w{3,}", venue_lower))
    text_tokens = set(re.findall(r"\w{3,}", txt_lower))
    if venue_tokens and len(venue_tokens & text_tokens) / len(venue_tokens) >= 0.5:
        return True
    return False


def patch_one(ext_file):
    ext = json.loads(ext_file.read_text(encoding="utf-8"))
    rp_id = ext.get("rp_id")
    paper_dir_match = list(PAGES.glob(f"{rp_id}_*"))
    if not paper_dir_match:
        return {"rp_id": rp_id, "year": "no-furniture", "venue": "no-furniture"}
    f_path = paper_dir_match[0] / "furniture.json"
    if not f_path.exists():
        return {"rp_id": rp_id, "year": "no-furniture", "venue": "no-furniture"}
    furniture = json.loads(f_path.read_text(encoding="utf-8"))

    actions = {"year": "untouched", "venue": "untouched"}

    # ----- year -----
    yr = ext.get("year")
    if isinstance(yr, dict) and "value" in yr:
        ev = yr.get("ev") or {}
        sec = (ev.get("section") or "").lower()
        if yr.get("value") is not None and sec in ("not_found", "absence", "references"):
            yval = str(yr["value"])
            hit = find_year_block(furniture, yval)
            if hit:
                page_num, blk, sec_label = hit
                ev_new = {
                    "page": page_num,
                    "section": sec_label,
                    "quote": blk["text"][:300],
                    "bbox": blk["bbox"],
                    "block_id": blk["block_id"],
                    "recovered_via": "PyMuPDF furniture sidecar",
                }
                yr["ev"] = ev_new
                actions["year"] = f"patched -> p{page_num}"
            else:
                actions["year"] = "no match"

    # ----- venue -----
    ven = ext.get("venue")
    if isinstance(ven, dict) and "value" in ven and ven.get("value"):
        ev = ven.get("ev") or {}
        sec = (ev.get("section") or "").lower()
        if sec in ("not_found", "absence", "references"):
            # Search every footer/header for venue match
            picked = None
            for page in furniture.get("pages", []):
                for kind, items in (("footer", page.get("footer", [])),
                                    ("header", page.get("header", []))):
                    for blk in items:
                        if venue_in_block(blk.get("text", ""), ven["value"]):
                            picked = (page["page"], blk,
                                      "Proceedings footer" if kind == "footer" else "Page header")
                            break
                    if picked:
                        break
                if picked:
                    break
            if picked:
                page_num, blk, sec_label = picked
                ven["ev"] = {
                    "page": page_num,
                    "section": sec_label,
                    "quote": blk["text"][:300],
                    "bbox": blk["bbox"],
                    "block_id": blk["block_id"],
                    "recovered_via": "PyMuPDF furniture sidecar",
                }
                actions["venue"] = f"patched -> p{page_num}"
            else:
                actions["venue"] = "no match"

    if actions["year"].startswith("patched") or actions["venue"].startswith("patched"):
        ext_file.write_text(json.dumps(ext, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"rp_id": rp_id, **actions}


def main():
    rows = []
    for fp in sorted(EXTR.glob("RP*_extraction.json")):
        rows.append(patch_one(fp))

    print(f"{'RP':<6} {'year':<25} {'venue':<25}")
    print("-" * 60)
    yp = vp = 0
    for r in rows:
        print(f"{r['rp_id']:<6} {r['year']:<25} {r['venue']:<25}")
        if r["year"].startswith("patched"): yp += 1
        if r["venue"].startswith("patched"): vp += 1
    print("-" * 60)
    print(f"Patched: year {yp}/50,  venue {vp}/50")


if __name__ == "__main__":
    main()
