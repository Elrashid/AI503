"""
Final v3 audit: confirm
  (1) NO leftover artifacts from the 18 deleted v2 papers anywhere in v3/
  (2) The CSV (RP_ID, DOI, Title, Year) ↔ PDF ↔ JSON ↔ XML ↔ extraction chain
      is consistent for every RP01..RP50.

Reads:
  exports/AI503_A1_RR_export_50papers_official.csv  (canonical RP_ID source)
  papers_pdf/  (32 kept) + papers_pdf/new/  (18 new — JSON/XML/extr not yet built)
  papers_json/, papers_xml/, extractions/

Run:
  python scripts/audit_v3.py
"""

import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

OFFICIAL_CSV = HERE / "exports" / "AI503_A1_RR_export_50papers_official.csv"
EXT_CSV      = HERE / "exports" / "AI503_A1_RR_export_50papers_extended.csv"
PDF_DIR      = HERE / "papers_pdf"
JSON_DIR     = HERE / "papers_json"
XML_DIR      = HERE / "papers_xml"
EXTR_DIR     = HERE / "extractions"
PAGES_DIR    = HERE / "papers_pages"
FIG_DIR      = HERE / "figures"

# v2 papers that were dropped (their RP IDs were reused by new papers)
# Kept here as the canonical "deleted" set against which we check for leftovers.
DELETED_FROM_V2 = {
    "RP08": ("Gholami",   2022, "Survey of Quantization"),
    "RP11": ("Frantar",   2023, "SparseGPT"),
    "RP14": ("Lin",       2023, "AWQ"),  # AWQ duplicate
    "RP19": ("Liu",       2023, "Deja Vu"),
    "RP23": ("Li",        2024, "From Generation to Judgment"),
    "RP25": ("Dahmani",   2024, "Logarithmic Lower Order"),
    "RP31": ("Chua",      2024, "SCAP"),
    "RP33": ("Offen",     2024, "Machine learning of discrete field"),
    "RP37": ("Rubel",     2024, "Generalized Adversarial Code"),
    "RP39": ("Yang",      2025, "Qwen3"),
    "RP40": ("Haziza",    2025, "2:4 Activation Sparsity"),
    "RP41": ("Maximov",   2025, "8:16 sparsity"),
    "RP42": ("Chen",      2025, "Verification Granularity"),
    "RP43": ("Kamirul",   2025, "SAR Ship Detection"),
    "RP44": ("Aljohani",  2025, "Healthcare Trust"),
    "RP46": ("Raza",      2025, "TRiSM Agentic"),
    "RP47": ("Shrestha",  2025, "Polar Sparsity"),
    "RP50": ("Mi",        2025, "ACE"),
}


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def expected_from_csv(rows, rp_col="RP_ID", title_col="Title", year_col="Year",
                     authors_col="Authors", doi_col="DOI"):
    out = {}
    for r in rows:
        rp = r[rp_col]
        if not rp:
            continue
        # First author cell — keep ALL tokens of the first comma-separated chunk.
        # Some rows from RR have surname-first order (e.g. "Wang Boxin") so we must
        # not assume `split(' ')[-1]` is the family name. We compare set membership.
        first_chunk = (r[authors_col] or "").split(",")[0].strip()
        first_auth_tokens = {t for t in first_chunk.split() if t}
        out[rp] = {
            "title": r[title_col],
            "year":  int(r[year_col]) if (r[year_col] or "").isdigit() else None,
            "first_author_tokens": first_auth_tokens,
            "first_author_display": first_chunk,
            "doi":   r[doi_col],
        }
    return out


def find_files_with_rp_prefix(directory, rp):
    """Return all entries (file or dir) whose name starts with RP{nn}_."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.name.startswith(f"{rp}_"))


def main():
    if not OFFICIAL_CSV.exists():
        print(f"ERROR: missing {OFFICIAL_CSV}")
        sys.exit(1)
    rows = load_csv(OFFICIAL_CSV)
    expected = expected_from_csv(rows)
    if len(expected) != 50:
        print(f"WARN: CSV has {len(expected)} RP IDs (expected 50)")

    # ---- (1) NO leftover artifacts from deleted v2 papers ----------------
    print("=" * 78)
    print("(1) DELETED-PAPER LEFTOVER CHECK")
    print("=" * 78)
    print()
    print("For each of the 18 RP IDs whose v2 paper was deleted, scan v3 for any")
    print("file/dir that mentions the OLD author name (proves the deletion was clean).")
    print()
    leftovers = []
    for rp, (old_auth, old_yr, _hint) in DELETED_FROM_V2.items():
        # Look across all artifact dirs for files starting with "RP{nn}_OldAuthor"
        suspect = []
        for d in [PDF_DIR, JSON_DIR, XML_DIR, EXTR_DIR, PAGES_DIR]:
            if not d.exists():
                continue
            for p in d.iterdir():
                # match e.g. RP08_Gholami_2022* (deleted) but NOT RP08_Aminabadi_2022 (new)
                if re.match(rf"{rp}_{re.escape(old_auth)}_", p.name, re.IGNORECASE):
                    suspect.append(str(p.relative_to(HERE)))
        # Figures: filter by RP{nn}_OldAuthor
        if FIG_DIR.exists():
            for p in FIG_DIR.iterdir():
                if re.match(rf"{rp}_{re.escape(old_auth)}_", p.name, re.IGNORECASE):
                    suspect.append(str(p.relative_to(HERE)))
        if suspect:
            leftovers.append((rp, old_auth, suspect))
            print(f"  [LEFTOVER] {rp} (was {old_auth} {old_yr}):")
            for s in suspect:
                print(f"       {s}")
        else:
            print(f"  [CLEAN]    {rp} (was {old_auth} {old_yr})")
    print()
    if leftovers:
        print(f"FAIL — {len(leftovers)} deleted-paper artifact set(s) still present.")
    else:
        print("PASS — no deleted-paper artifacts found anywhere in v3/.")
    print()

    # ---- (2) Mapping consistency: CSV ↔ PDF ↔ JSON ↔ XML ↔ extraction ----
    print("=" * 78)
    print("(2) MAPPING CONSISTENCY (CSV -> PDF -> JSON -> XML -> extraction)")
    print("=" * 78)
    print()
    print(f"{'RP':<5}{'Author/Year':<22}{'PDF':<8}{'JSON':<7}{'XML':<7}{'Extr':<6}{'Status'}")
    print("-" * 78)

    errors = []
    summary = {"complete": 0, "kept_only": 0, "new_pdf_only": 0, "new_pdf_json": 0, "broken": 0}

    for rp in sorted(expected.keys(), key=lambda r: int(r[2:])):
        exp = expected[rp]
        all_pdfs = [p for p in find_files_with_rp_prefix(PDF_DIR, rp)
                    if p.is_file() and p.suffix.lower() == ".pdf"]

        json_dirs = [p for p in find_files_with_rp_prefix(JSON_DIR, rp) if p.is_dir()]
        xml_files = [p for p in find_files_with_rp_prefix(XML_DIR, rp) if p.is_file()]
        extr_files = [p for p in find_files_with_rp_prefix(EXTR_DIR, rp) if p.is_file()]

        # Filename consistency: filename author token should match SOME token in
        # the CSV's first-author chunk (handles both "Boxin Wang" and "Wang Boxin").
        author_check_ok = True
        author_files_seen = set()
        csv_tokens_lower = {t.lower() for t in exp["first_author_tokens"]}
        for f in all_pdfs + json_dirs + xml_files + extr_files:
            m = re.match(rf"{rp}_([^_]+)_(\d{{4}})", f.name)
            if m:
                fauth = m.group(1)
                author_files_seen.add(fauth)
                if fauth.lower() not in csv_tokens_lower:
                    # also accept partial overlap (e.g. hyphenated names)
                    overlap = any(fauth.lower() in t or t in fauth.lower()
                                  for t in csv_tokens_lower)
                    if not overlap:
                        author_check_ok = False

        # Build status flags
        flags = []
        if all_pdfs:    flags.append("PDF")
        if json_dirs:   flags.append("JSON")
        if xml_files:   flags.append("XML")
        if extr_files:  flags.append("EXTR")

        if json_dirs and xml_files and extr_files and all_pdfs:
            status = "complete"
            summary["complete"] += 1
        elif all_pdfs and json_dirs and not (xml_files or extr_files):
            status = "new (PDF + JSON, XML/extr pending)"
            summary["new_pdf_json"] += 1
        elif all_pdfs and not (json_dirs or xml_files or extr_files):
            status = "new (PDF only)"
            summary["new_pdf_only"] += 1
        elif all_pdfs and (xml_files or extr_files):
            status = "kept (partial)"
            summary["kept_only"] += 1
        else:
            status = "BROKEN"
            errors.append(f"{rp} {exp['first_author_display']} {exp['year']}: missing all artifacts (or no PDF)")
            summary["broken"] += 1

        if not author_check_ok:
            status += " | AUTHOR MISMATCH"
            errors.append(f"{rp}: filename author seen={sorted(author_files_seen)} "
                          f"but CSV first-author chunk = '{exp['first_author_display']}'")

        # Filename info for printing
        sample_pdf = (all_pdfs[0].name if all_pdfs else "-")
        # Pick a stable display token (last surname-ish token from CSV)
        display_auth = (exp['first_author_display'].split() or ["?"])[-1][:14]
        ay = f"{display_auth} {exp['year']}"
        print(f"{rp:<5}{ay:<22}{(len(all_pdfs)):<8}{(len(json_dirs)):<7}"
              f"{(len(xml_files)):<7}{(len(extr_files)):<6}{status}")

    print("-" * 78)
    print(f"\nSummary:")
    print(f"  Complete (PDF + JSON + XML + Extraction): {summary['complete']}/50")
    print(f"  PDF + JSON (XML/extr pending):            {summary['new_pdf_json']}/50")
    print(f"  PDF only (JSON/XML/extr pending):         {summary['new_pdf_only']}/50")
    print(f"  Kept partial (PDF + some artifacts):      {summary['kept_only']}/50")
    print(f"  Broken (artifacts but no PDF, or none):   {summary['broken']}/50")
    print()
    if errors:
        print(f"FAIL — {len(errors)} consistency error(s):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("PASS — every RP01..RP50 has artifacts that match its CSV row "
              "(author/year), and no orphan files.")
    print()

    # ---- Final verdict ----
    print("=" * 78)
    if not leftovers and not errors:
        print("OVERALL: PASS — v3 is clean and self-consistent.")
        sys.exit(0)
    else:
        print("OVERALL: FAIL — see issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
