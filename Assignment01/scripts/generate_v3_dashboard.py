"""
Build a single-file HTML dashboard for the AI503 SLR v3 corpus.

Reads:
  v3/extractions/*.json
  v3/exports/AI503_A1_RR_export_50papers_extended.csv (for citations, OA flag)
  v3/papers_xml/, v3/papers_pages/ (existence checks)

Output:
  v3/dashboard.html
"""

import csv
import html
import json
import os
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
EXTR_DIR = HERE / "extractions"
EXPORT_CSV = HERE / "exports" / "AI503_A1_RR_export_50papers_extended.csv"
XML_DIR = HERE / "papers_xml"
PAGES_DIR = HERE / "papers_pages"
OUT = HERE / "dashboard.html"


def v(node):
    """Unwrap {value, ev} -> value. Pass through scalars/lists."""
    if isinstance(node, dict) and "value" in node and "ev" in node:
        return node["value"]
    return node


def collect_values(arr):
    """Flatten a list of {value, ev} into [value, ...]."""
    if not arr:
        return []
    return [v(x) for x in arr if v(x) is not None]


def count_leaves_traced(j, parent_key=""):
    """Walk extraction JSON, count traced leaves."""
    total = traced = withbbox = 0
    if isinstance(j, dict):
        if "value" in j and "ev" in j:
            total = 1
            ev = j.get("ev") or {}
            if ev.get("page") and ev.get("section") and ev.get("section") != "not_found":
                traced = 1
            if ev.get("bbox"):
                withbbox = 1
            return total, traced, withbbox
        for k, val in j.items():
            t, tr, b = count_leaves_traced(val, k)
            total += t
            traced += tr
            withbbox += b
    elif isinstance(j, list):
        for item in j:
            t, tr, b = count_leaves_traced(item, parent_key)
            total += t
            traced += tr
            withbbox += b
    return total, traced, withbbox


def load_papers():
    # Load CSV for citations + OA + arxivId
    csv_meta = {}
    if EXPORT_CSV.exists():
        with open(EXPORT_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rp = row.get("RP_ID") or row.get("﻿RP_ID")
                if rp:
                    csv_meta[rp] = row

    papers = []
    for fp in sorted(EXTR_DIR.glob("RP*_extraction.json")):
        rp = fp.name.split("_")[0]
        with open(fp, encoding="utf-8") as f:
            j = json.load(f)
        total, traced, withbbox = count_leaves_traced(j)

        # ID from filename
        # Find xml + pages existence
        xml_exists = bool(list(XML_DIR.glob(f"{rp}_*_claude.xml")))
        pages_exists = any(d.is_dir() and d.name.startswith(rp + "_")
                          for d in PAGES_DIR.iterdir()) if PAGES_DIR.exists() else False

        # Pull from extraction
        title = v(j.get("title")) or "?"
        first_author = v(j.get("first_author")) or "?"
        year = v(j.get("year")) or 0
        venue = v(j.get("venue")) or "?"
        ptype = v(j.get("paper_type")) or "?"
        mfam = v(j.get("method_family")) or "?"
        decision = v((j.get("decision") or {}).get("include") if j.get("decision") else None) or "INCLUDE"
        bits = collect_values(v(j.get("bit_widths_tested")) or [])
        models = collect_values(v(j.get("models_used")) or [])
        datasets = collect_values(v(j.get("datasets")) or [])
        metrics = collect_values(v(j.get("eval_metrics")) or [])

        # Gap signals
        gs = j.get("gap_signals") or {}
        gap_pos = sum(1 for k, val in gs.items() if v(val) is True)
        gap_total = len(gs)

        # CSV crossreference
        meta = csv_meta.get(rp, {})
        cited_by = meta.get("forwardEdgeCount") or "0"
        try:
            cited_by = int(cited_by)
        except ValueError:
            cited_by = 0
        is_oa = meta.get("isOpenAccess") == "True"
        arxiv = meta.get("arxivId") or ""
        doi = meta.get("doi") or ""

        papers.append({
            "rp": rp,
            "title": title,
            "first_author": first_author,
            "year": int(year) if isinstance(year, (int, str)) and str(year).isdigit() else 0,
            "venue": venue,
            "ptype": ptype,
            "mfam": mfam,
            "decision": decision,
            "bits": bits,
            "n_models": len(models),
            "n_datasets": len(datasets),
            "n_metrics": len(metrics),
            "leaves_total": total,
            "leaves_traced": traced,
            "leaves_bbox": withbbox,
            "gap_positive": gap_pos,
            "gap_total": gap_total,
            "gap_signals": {k: v(val) for k, val in gs.items()},
            "cited_by": cited_by,
            "is_oa": is_oa,
            "arxiv": arxiv,
            "doi": doi,
            "xml_exists": xml_exists,
            "pages_exists": pages_exists,
        })
    return papers


def render(papers):
    n = len(papers)
    total_leaves = sum(p["leaves_total"] for p in papers)
    total_traced = sum(p["leaves_traced"] for p in papers)
    total_bbox = sum(p["leaves_bbox"] for p in papers)

    # Year distribution
    by_year = Counter(p["year"] for p in papers)
    # Paper type distribution
    by_ptype = Counter(p["ptype"] for p in papers)
    # Method family
    by_mfam = Counter(p["mfam"] for p in papers)
    # Recent vs older (last 5 yrs from 2026)
    recent_threshold = 2021
    recent = sum(1 for p in papers if p["year"] >= recent_threshold)
    # Gap-signal coverage (which signals are positive in N papers)
    sig_counts = Counter()
    for p in papers:
        for k, val in p["gap_signals"].items():
            if val is True:
                sig_counts[k] += 1

    def chart_bar(label_count_pairs, title, color="#4472C4", max_w=400):
        max_v = max((c for _, c in label_count_pairs), default=1)
        rows = []
        for label, count in label_count_pairs:
            w = int(count / max_v * max_w) if max_v else 0
            rows.append(
                f'<div class="row"><span class="lbl">{html.escape(str(label))}</span>'
                f'<span class="bar" style="width:{w}px;background:{color}"></span>'
                f'<span class="num">{count}</span></div>'
            )
        return f'<div class="chart"><h3>{html.escape(title)}</h3>{"".join(rows)}</div>'

    # Year chart (chronological)
    year_chart = chart_bar(sorted(by_year.items()), "Year distribution (n=%d)" % n)

    # Paper type chart
    ptype_chart = chart_bar(sorted(by_ptype.items(), key=lambda x: -x[1]),
                           "Paper type", color="#8E44AD")

    # Method family
    mfam_chart = chart_bar(sorted(by_mfam.items(), key=lambda x: -x[1]),
                          "Method family", color="#16A085")

    # Gap signals
    sig_pretty = {
        "reports_energy_consumption": "Reports energy",
        "reports_statistical_significance": "Stats / CIs",
        "tests_on_edge_hardware": "Edge HW tested",
        "tests_safety_after_quantization": "Safety post-quant",
        "tests_below_4_bit": "Sub-4-bit tested",
        "reports_real_latency_not_just_throughput": "Real latency",
        "evaluates_long_context": "Long-context eval",
        "releases_code": "Code released",
        "evaluates_instruction_tuned_models": "Instruction-tuned eval",
    }
    sig_pairs = sorted(
        ((sig_pretty.get(k, k), v) for k, v in sig_counts.items()),
        key=lambda x: -x[1],
    )
    gap_chart = chart_bar(sig_pairs, "Gap-signal coverage (papers reporting yes, n=%d)" % n,
                         color="#E67E22")

    # Per-paper rows
    table_rows = []
    for p in sorted(papers, key=lambda x: int(x["rp"][2:])):
        bits_str = "/".join(str(b) for b in p["bits"]) if p["bits"] else "—"
        gap_pct = int(p["gap_positive"] / p["gap_total"] * 100) if p["gap_total"] else 0
        traced_pct = int(p["leaves_traced"] / p["leaves_total"] * 100) if p["leaves_total"] else 0
        oa_badge = '<span class="oa">OA</span>' if p["is_oa"] else ""
        arxiv_link = f'<a href="https://arxiv.org/abs/{p["arxiv"]}" target="_blank">arXiv</a>' if p["arxiv"] else ""
        explorer_link = f'<a href="paper_explorer.html#{p["rp"]}">explorer</a>'
        table_rows.append(f"""<tr>
<td class="rp">{html.escape(p["rp"])}</td>
<td>{html.escape(str(p["first_author"]))} {p["year"]}</td>
<td class="title" title="{html.escape(p["title"])}">{html.escape(p["title"][:62])}</td>
<td>{html.escape(p["ptype"])}</td>
<td>{html.escape(p["mfam"])}</td>
<td>{bits_str}</td>
<td class="num">{p["n_models"]}</td>
<td class="num">{p["n_datasets"]}</td>
<td class="num">{p["n_metrics"]}</td>
<td class="num"><span title="{p["leaves_traced"]}/{p["leaves_total"]} leaves">{traced_pct}%</span></td>
<td class="num"><span title="{p["gap_positive"]}/{p["gap_total"]} gap signals positive">{gap_pct}%</span></td>
<td class="num">{p["cited_by"]:,}</td>
<td>{oa_badge} {arxiv_link} {explorer_link}</td>
</tr>""")

    # Stats summary
    stats_html = f"""<div class="stats">
<div class="stat"><div class="big">{n}</div><div>papers</div></div>
<div class="stat"><div class="big">{recent}</div><div>recent (≥{recent_threshold})</div></div>
<div class="stat"><div class="big">{int(recent / n * 100)}%</div><div>recency rate</div></div>
<div class="stat"><div class="big">{total_leaves:,}</div><div>extraction leaves</div></div>
<div class="stat"><div class="big">{int(total_traced / total_leaves * 100)}%</div><div>traced</div></div>
<div class="stat"><div class="big">{int(total_bbox / total_leaves * 100)}%</div><div>with bbox</div></div>
</div>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>AI503 Assignment 1 — SLR Dashboard (v3)</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, system-ui, sans-serif;
         margin: 0; padding: 24px 32px; background: #f7f8fa; color: #1f2329; }}
  h1 {{ margin: 0 0 4px 0; font-size: 24px; }}
  h2 {{ margin: 24px 0 8px 0; font-size: 18px; color: #444; }}
  h3 {{ margin: 0 0 8px 0; font-size: 14px; color: #666; }}
  .sub {{ color: #555; margin-bottom: 24px; font-size: 14px; }}
  .stats {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; }}
  .stat {{ background: white; padding: 16px; border-radius: 8px;
          border: 1px solid #e1e4e8; text-align: center; }}
  .stat .big {{ font-size: 26px; font-weight: 600; color: #4472C4; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }}
  .chart {{ background: white; padding: 16px; border-radius: 8px;
           border: 1px solid #e1e4e8; }}
  .row {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 13px; }}
  .lbl {{ display: inline-block; width: 220px; color: #444; overflow: hidden;
         text-overflow: ellipsis; white-space: nowrap; }}
  .bar {{ height: 16px; border-radius: 3px; }}
  .num {{ color: #555; min-width: 30px; text-align: right; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
          border-radius: 8px; border: 1px solid #e1e4e8; overflow: hidden;
          margin-top: 12px; font-size: 13px; }}
  th {{ background: #f0f2f5; text-align: left; padding: 8px 10px;
       border-bottom: 1px solid #e1e4e8; font-weight: 600; cursor: pointer; }}
  th:hover {{ background: #e6e9ec; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #f0f2f5; vertical-align: middle; }}
  td.rp {{ font-family: monospace; font-weight: 600; color: #4472C4; }}
  td.title {{ max-width: 380px; overflow: hidden; text-overflow: ellipsis;
              white-space: nowrap; }}
  td.num {{ text-align: right; }}
  tr:hover {{ background: #fafbfc; }}
  .oa {{ background: #16A085; color: white; padding: 2px 6px; border-radius: 3px;
        font-size: 11px; font-weight: 600; }}
  a {{ color: #4472C4; text-decoration: none; margin-right: 6px; }}
  a:hover {{ text-decoration: underline; }}
  .filter {{ margin-bottom: 12px; }}
  .filter input {{ padding: 6px 10px; width: 320px; border: 1px solid #ccc;
                  border-radius: 4px; font-size: 14px; }}
</style>
</head><body>
<h1>AI503 Assignment 1 — SLR Dashboard</h1>
<div class="sub">Quantization Safety SLR · v3 · {n} papers · regenerated from
<code>extractions/</code> + <code>exports/AI503_A1_RR_export_50papers_extended.csv</code></div>

{stats_html}

<h2>Coverage charts</h2>
<div class="charts">
{year_chart}
{ptype_chart}
{mfam_chart}
{gap_chart}
</div>

<h2>Per-paper status (sortable, filterable)</h2>
<div class="filter">
  <input type="text" id="filter" placeholder="Filter by title, author, RP id, paper_type…"
         oninput="filterTable()">
</div>
<table id="papers">
<thead><tr>
<th onclick="sortBy(0)">RP</th>
<th onclick="sortBy(1)">Author / Year</th>
<th onclick="sortBy(2)">Title</th>
<th onclick="sortBy(3)">Type</th>
<th onclick="sortBy(4)">Method family</th>
<th onclick="sortBy(5)">Bits</th>
<th onclick="sortBy(6)">#Models</th>
<th onclick="sortBy(7)">#Data</th>
<th onclick="sortBy(8)">#Metrics</th>
<th onclick="sortBy(9)">Traced</th>
<th onclick="sortBy(10)">Gap signals</th>
<th onclick="sortBy(11)">Cited by</th>
<th>Links</th>
</tr></thead>
<tbody>
{"".join(table_rows)}
</tbody>
</table>

<script>
function filterTable() {{
  const q = document.getElementById('filter').value.toLowerCase();
  const rows = document.querySelectorAll('#papers tbody tr');
  for (const r of rows) {{
    r.style.display = r.textContent.toLowerCase().includes(q) ? '' : 'none';
  }}
}}
let sortDir = {{}};
function sortBy(col) {{
  const tbody = document.querySelector('#papers tbody');
  const rows = Array.from(tbody.rows);
  const dir = sortDir[col] = !sortDir[col];
  const isNum = (s) => !isNaN(parseFloat(s.replace(/[%,]/g, '')));
  rows.sort((a, b) => {{
    const av = a.cells[col].textContent.trim();
    const bv = b.cells[col].textContent.trim();
    if (isNum(av) && isNum(bv)) {{
      return dir ? parseFloat(av.replace(/[%,]/g, '')) - parseFloat(bv.replace(/[%,]/g, ''))
                 : parseFloat(bv.replace(/[%,]/g, '')) - parseFloat(av.replace(/[%,]/g, ''));
    }}
    return dir ? av.localeCompare(bv) : bv.localeCompare(av);
  }});
  for (const r of rows) tbody.appendChild(r);
}}
</script>
</body></html>"""


if __name__ == "__main__":
    papers = load_papers()
    html_out = render(papers)
    OUT.write_text(html_out, encoding="utf-8")
    sz = OUT.stat().st_size
    print(f"Wrote {OUT.relative_to(HERE)}: {sz / 1024:.1f} KB · {len(papers)} papers")
    print(f"Open: file:///{OUT.as_posix()}")
