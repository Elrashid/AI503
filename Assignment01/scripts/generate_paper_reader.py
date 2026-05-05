"""
Build a focus-mode paper reader (paper_reader.html) — ADHD-friendly + mobile-first.

Reads:
  extractions/*.json   — for the paper list + key findings sidebar
  papers_md/*.md       — the rendered paper content (loaded on demand at runtime)

Writes:
  paper_reader.html    — single-file app (~50 KB), loads papers via fetch()
  paper_reader.json    — manifest with paper metadata + key findings
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
EXTR_DIR = HERE / "extractions"
MD_DIR = HERE / "papers_md"
OUT_HTML = HERE / "paper_reader.html"
OUT_MANIFEST = HERE / "paper_reader.json"


def v(node):
    if isinstance(node, dict) and "value" in node and "ev" in node:
        return node["value"]
    return node


def collect(arr):
    if not arr:
        return []
    return [v(x) for x in arr if v(x) is not None]


def build_manifest():
    # Pages manifest (DPI + per-paper page count + dir name)
    pages_manifest_path = HERE / "papers_pages" / "manifest.json"
    pages_meta = {}
    if pages_manifest_path.exists():
        with open(pages_manifest_path, encoding="utf-8") as f:
            pages_meta = json.load(f).get("papers", {})

    papers = []
    for fp in sorted(EXTR_DIR.glob("RP*_extraction.json")):
        rp = fp.name.split("_")[0]
        with open(fp, encoding="utf-8") as f:
            j = json.load(f)
        title = v(j.get("title")) or "?"
        first_author = v(j.get("first_author")) or "?"
        authors = v(j.get("authors")) or []
        year = v(j.get("year")) or 0
        venue = v(j.get("venue")) or ""
        doi = v(j.get("doi")) or ""
        ptype = v(j.get("paper_type")) or ""
        mfam = v(j.get("method_family")) or ""
        contribution = v(j.get("contribution")) or ""

        # Key findings — pull just the value strings + page numbers
        kr_raw = j.get("key_results") or []
        if isinstance(kr_raw, dict) and "value" in kr_raw:
            kr_raw = kr_raw["value"] or []
        findings = []
        for kr in (kr_raw or []):
            if isinstance(kr, dict):
                value = kr.get("value", "") if "value" in kr else kr
                ev = kr.get("ev") or {}
                page = ev.get("page") if isinstance(ev, dict) else None
                section = ev.get("section") if isinstance(ev, dict) else None
                findings.append({"text": value if isinstance(value, str) else str(value),
                                "page": page, "section": section})

        # MD file existence
        md_files = list(MD_DIR.glob(f"{rp}_*.md"))
        md_path = f"papers_md/{md_files[0].name}" if md_files else None

        # Pages
        pmeta = pages_meta.get(rp, {})
        pages_dir = f"papers_pages/{pmeta['dir']}/" if pmeta.get("dir") else None
        page_count = pmeta.get("page_count") or 0
        pdf_path = f"papers_pdf/{pmeta['pdf_file']}" if pmeta.get("pdf_file") else None

        papers.append({
            "rp_id": rp,
            "title": title,
            "first_author": first_author,
            "authors_count": len(authors) if isinstance(authors, list) else 0,
            "year": int(year) if str(year).isdigit() else 0,
            "venue": venue,
            "doi": doi,
            "paper_type": ptype,
            "method_family": mfam,
            "contribution": contribution,
            "findings": findings,
            "md_path": md_path,
            "extraction_path": f"extractions/{fp.name}",
            "pages_dir": pages_dir,
            "page_count": page_count,
            "pdf_path": pdf_path,
        })

    papers.sort(key=lambda p: int(p["rp_id"][2:]))
    return papers


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5">
<meta name="theme-color" content="#fdf6e3">
<title>AI503 — Paper Reader</title>
<style>
  /* === Calm, low-stim palette (Solarized-light–inspired) === */
  :root {
    --bg: #fdf6e3; --bg-alt: #eee8d5;
    --ink: #073642; --ink-soft: #586e75; --muted: #93a1a1;
    --accent: #268bd2; --accent-soft: #cfe6f5;
    --warn: #b58900; --good: #859900;
    --border: #d6cfb8; --shadow: 0 2px 12px rgba(7, 54, 66, 0.08);
    --reader-width: 780px;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #002b36; --bg-alt: #073642;
      --ink: #eee8d5; --ink-soft: #93a1a1; --muted: #586e75;
      --accent-soft: #073642; --border: #073642;
    }
  }
  * { box-sizing: border-box; }
  html { font-size: 18px; }
  body {
    margin: 0;
    font-family: -apple-system, "Segoe UI", "Atkinson Hyperlegible", system-ui, sans-serif;
    background: var(--bg); color: var(--ink); line-height: 1.6;
    -webkit-font-smoothing: antialiased;
    overscroll-behavior-y: contain;
  }

  /* === Top bar === */
  .topbar {
    position: sticky; top: 0; z-index: 30;
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px;
    background: var(--bg-alt); border-bottom: 1px solid var(--border);
    backdrop-filter: blur(6px);
  }
  .topbar h1 { font-size: 17px; margin: 0; font-weight: 600; flex: 1; }
  .topbar h1 small { font-size: 13px; color: var(--ink-soft); font-weight: 400; }
  .iconbtn {
    appearance: none; border: 1px solid var(--border);
    background: var(--bg); color: var(--ink);
    width: 44px; height: 40px; border-radius: 8px;
    cursor: pointer; font-size: 17px;
    display: flex; align-items: center; justify-content: center;
  }
  .iconbtn.active { background: var(--accent); color: white; border-color: var(--accent); }
  .iconbtn:active { background: var(--bg-alt); }
  .progress {
    color: var(--ink-soft); font-size: 13px;
    padding: 4px 10px; background: var(--bg); border-radius: 6px;
    border: 1px solid var(--border);
  }

  /* === Drawer === */
  .drawer-bg {
    position: fixed; inset: 0; background: rgba(0,0,0,0.35);
    opacity: 0; pointer-events: none; transition: opacity 0.18s; z-index: 40;
  }
  .drawer-bg.open { opacity: 1; pointer-events: auto; }
  .drawer {
    position: fixed; top: 0; left: 0; bottom: 0;
    width: min(420px, 90vw);
    background: var(--bg);
    box-shadow: var(--shadow);
    z-index: 50;
    transform: translateX(-100%); transition: transform 0.22s;
    display: flex; flex-direction: column;
  }
  .drawer.open { transform: translateX(0); }
  .drawer-head {
    padding: 12px 14px; border-bottom: 1px solid var(--border);
    display: flex; gap: 8px; align-items: center;
  }
  .drawer-head input {
    flex: 1; padding: 12px; font-size: 16px;
    border: 1px solid var(--border); border-radius: 8px;
    background: var(--bg-alt); color: var(--ink);
  }
  .drawer-list { flex: 1; overflow-y: auto; padding: 8px 0; }
  .paper-item {
    display: block; padding: 14px 18px; border-bottom: 1px solid var(--border);
    cursor: pointer; line-height: 1.4;
  }
  .paper-item:hover { background: var(--bg-alt); }
  .paper-item.active {
    background: var(--accent-soft); border-left: 4px solid var(--accent); padding-left: 14px;
  }
  .paper-item .rp {
    font-family: ui-monospace, SFMono-Regular, monospace;
    color: var(--accent); font-weight: 600; font-size: 14px;
  }
  .paper-item .meta { color: var(--ink-soft); font-size: 13px; margin-top: 2px; }
  .paper-item .title { font-size: 15px; margin-top: 4px; }

  /* === Main content === */
  main {
    max-width: var(--reader-width);
    margin: 0 auto;
    padding: 20px 18px 90px;
  }
  body.wide main { max-width: 100%; padding: 20px 32px 90px; }

  .paper-header {
    margin: 8px 0 24px; padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }
  .paper-header .rp-tag {
    display: inline-block; background: var(--accent); color: white;
    padding: 4px 10px; border-radius: 6px;
    font-family: ui-monospace, monospace; font-size: 14px; font-weight: 600;
  }
  .paper-header h2 { font-size: 24px; line-height: 1.3; margin: 12px 0 8px; }
  .paper-header .author-line { color: var(--ink-soft); font-size: 15px; }
  .paper-header .pills { margin-top: 12px; }
  .pill {
    display: inline-block; font-size: 12px;
    padding: 3px 8px; margin-right: 6px;
    border-radius: 12px; border: 1px solid var(--border);
    color: var(--ink-soft);
  }

  /* === Tabs === */
  .modes {
    display: flex; gap: 4px; margin-bottom: 16px;
    background: var(--bg-alt); padding: 4px; border-radius: 10px;
    overflow-x: auto;
  }
  .mode-btn {
    flex: 1; min-width: 100px;
    padding: 10px 6px; border: none; cursor: pointer;
    background: transparent; color: var(--ink-soft);
    border-radius: 6px; font-size: 14px; font-weight: 500;
    white-space: nowrap;
  }
  .mode-btn.active { background: var(--bg); color: var(--ink); box-shadow: var(--shadow); }

  .pane { display: none; animation: fadeIn 0.18s; }
  .pane.active { display: block; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }

  /* === Markdown styling === */
  .md h1 { font-size: 22px; margin: 32px 0 12px; padding-top: 8px; }
  .md h2 { font-size: 19px; margin: 28px 0 10px; }
  .md h3 { font-size: 17px; margin: 24px 0 8px; }
  .md p { margin: 0 0 18px; line-height: 1.7; }
  .md a { color: var(--accent); }
  .md code { background: var(--bg-alt); padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }
  .md pre { background: var(--bg-alt); padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 14px; }
  .md table { border-collapse: collapse; margin: 16px 0; font-size: 14px; width: 100%; }
  .md table th, .md table td { border: 1px solid var(--border); padding: 6px 10px; }
  .md table th { background: var(--bg-alt); }
  .md img {
    max-width: 100%; height: auto; border-radius: 4px; margin: 12px 0;
    cursor: zoom-in; background: var(--bg-alt);
  }
  .md blockquote {
    border-left: 3px solid var(--accent); margin: 16px 0; padding: 4px 12px;
    color: var(--ink-soft); background: var(--bg-alt); border-radius: 0 4px 4px 0;
  }
  .md hr { border: none; border-top: 1px dashed var(--border); margin: 32px 0; }
  .md .page-marker {
    color: var(--muted); font-size: 11px; text-align: center; margin: 24px 0 8px;
    text-transform: uppercase; letter-spacing: 1px;
  }

  /* === Findings cards === */
  .finding {
    background: var(--bg);
    border: 1px solid var(--border);
    border-left: 3px solid var(--good);
    padding: 14px 16px; margin-bottom: 12px;
    border-radius: 6px; font-size: 16px;
  }
  .finding .where {
    display: block; font-size: 12px; color: var(--ink-soft); margin-top: 8px;
    font-family: ui-monospace, monospace;
  }
  .summary-block {
    background: var(--bg-alt); padding: 14px 16px;
    border-radius: 8px; margin-bottom: 24px;
  }

  /* === Pages tab === */
  .pages-grid {
    display: grid;
    gap: 16px;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  }
  body.wide .pages-grid { grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); }
  .page-thumb {
    background: var(--bg-alt); border: 1px solid var(--border); border-radius: 6px;
    overflow: hidden; cursor: zoom-in; position: relative;
  }
  .page-thumb img { display: block; width: 100%; height: auto; }
  .page-thumb .num {
    position: absolute; top: 6px; left: 6px;
    background: rgba(7, 54, 66, 0.85); color: var(--bg);
    padding: 2px 8px; border-radius: 12px;
    font-size: 11px; font-weight: 600; font-family: ui-monospace, monospace;
  }

  /* === Lightbox === */
  .lightbox {
    position: fixed; inset: 0; z-index: 100;
    background: rgba(0, 0, 0, 0.92);
    display: none; align-items: center; justify-content: center;
    padding: 20px;
  }
  .lightbox.open { display: flex; }
  .lightbox img { max-width: 100%; max-height: 100%; cursor: zoom-out; }
  .lightbox .lb-close {
    position: absolute; top: 12px; right: 12px;
    background: var(--bg); color: var(--ink);
    width: 44px; height: 44px; border-radius: 8px; border: none; cursor: pointer;
    font-size: 22px; font-weight: 600;
  }
  .lightbox .lb-info {
    position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%);
    background: var(--bg); color: var(--ink); padding: 6px 14px;
    border-radius: 18px; font-size: 13px;
  }

  /* === Data tab (extraction tree) === */
  .data-tree { font-family: ui-monospace, SFMono-Regular, monospace; font-size: 13px; }
  .data-key { color: var(--accent); font-weight: 600; }
  .data-leaf {
    margin: 4px 0; padding: 8px 12px;
    background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
  }
  .data-leaf .v { color: var(--ink); margin: 4px 0; line-height: 1.5;
                  white-space: pre-wrap; word-break: break-word;
                  font-family: -apple-system, system-ui, sans-serif; font-size: 14px; }
  .data-leaf .v.empty { color: var(--muted); font-style: italic; }
  .data-leaf .ev {
    color: var(--ink-soft); font-size: 11px;
    border-top: 1px dashed var(--border); padding-top: 4px; margin-top: 6px;
  }
  .data-leaf .ev .pg { color: var(--warn); font-weight: 600; }
  .data-section {
    margin: 16px 0; padding: 12px; background: var(--bg-alt);
    border-radius: 6px; border: 1px solid var(--border);
  }
  .data-section > h3 {
    margin: 0 0 8px; font-size: 14px; color: var(--accent);
    font-family: ui-monospace, monospace;
  }
  .data-list-item {
    margin-left: 12px; padding-left: 12px;
    border-left: 2px solid var(--border);
  }

  /* === Bottom nav === */
  .bottom-nav {
    position: fixed; bottom: 0; left: 0; right: 0;
    display: flex; gap: 8px; padding: 8px;
    background: var(--bg-alt); border-top: 1px solid var(--border);
    z-index: 20;
  }
  .nav-btn {
    flex: 1; padding: 14px; border: 1px solid var(--border);
    background: var(--bg); color: var(--ink);
    border-radius: 8px; cursor: pointer; font-size: 14px;
  }
  .nav-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* === Wide screens: drawer permanent + 2-column reader if 'side-by-side' === */
  @media (min-width: 900px) {
    body { display: grid; grid-template-columns: 320px 1fr; min-height: 100vh; }
    body.no-drawer { grid-template-columns: 1fr; }
    .topbar { grid-column: 1 / -1; }
    .drawer {
      position: static; transform: none; box-shadow: none;
      border-right: 1px solid var(--border);
      width: 320px; height: calc(100vh - 65px);
    }
    body.no-drawer .drawer { display: none; }
    .drawer-bg, .iconbtn.menu { display: none; }
    main { padding: 24px 40px 40px; }
    .bottom-nav { left: 320px; }
    body.no-drawer .bottom-nav { left: 0; }
  }

  @media (min-width: 1400px) {
    :root { --reader-width: 920px; }
  }

  /* Loading + empty states */
  .loading, .empty { text-align: center; padding: 60px 20px; color: var(--ink-soft); }

  /* Focus mode */
  body.focus-mode .topbar > *:not(h1):not(.iconbtn.focus) { display: none; }
  body.focus-mode .drawer-bg, body.focus-mode .drawer { display: none; }
  body.focus-mode .modes, body.focus-mode .paper-header .pills { display: none; }
  body.focus-mode .bottom-nav { display: none; }
  body.focus-mode main { max-width: 720px; padding-bottom: 40px; }
  body.focus-mode { grid-template-columns: 1fr !important; }
</style>
</head>
<body>

<header class="topbar">
  <button class="iconbtn menu" onclick="toggleDrawer()" aria-label="Open paper list" title="Papers">☰</button>
  <h1>AI503 Reader <small id="paperCount">— loading…</small></h1>
  <span class="progress" id="progress">—</span>
  <button class="iconbtn" id="wideBtn" onclick="toggleWide()" title="Toggle wide layout (W)" aria-label="Wide">↔</button>
  <button class="iconbtn focus" onclick="toggleFocus()" title="Focus mode (F)" aria-label="Focus">🎯</button>
</header>

<div class="drawer-bg" onclick="toggleDrawer(false)"></div>
<aside class="drawer" id="drawer">
  <div class="drawer-head">
    <input type="search" id="search" placeholder="Search title, author, RP…" oninput="filterPapers()">
  </div>
  <div class="drawer-list" id="paperList"><div class="loading">Loading…</div></div>
</aside>

<main id="main">
  <div class="loading">Loading paper list…</div>
</main>

<nav class="bottom-nav" id="bottomNav" style="display:none">
  <button class="nav-btn" id="prevBtn" onclick="navigate(-1)">← Previous (k)</button>
  <button class="nav-btn" id="nextBtn" onclick="navigate(1)">Next (j) →</button>
</nav>

<div class="lightbox" id="lightbox" onclick="closeLightbox(event)">
  <button class="lb-close" onclick="closeLightbox()">×</button>
  <img id="lightboxImg" alt="">
  <div class="lb-info" id="lightboxInfo"></div>
</div>

<script>
let manifest = [];
let currentIndex = -1;
let currentMode = 'read';
let currentExtraction = null;   // cached for the data tab

const $ = (q) => document.querySelector(q);
const $$ = (q) => document.querySelectorAll(q);

async function loadManifest() {
  try {
    manifest = await (await fetch('paper_reader.json')).json();
    $('#paperCount').textContent = `· ${manifest.length} papers`;
    renderPaperList();
    const initial = location.hash.replace('#', '') || manifest[0].rp_id;
    showPaper(initial);
  } catch (e) {
    $('#main').innerHTML = `<div class="empty">Failed to load manifest: ${e.message}</div>`;
  }
}

function renderPaperList(filter = '') {
  const f = filter.toLowerCase();
  const html = manifest.map((p, i) => {
    const hay = `${p.rp_id} ${p.first_author} ${p.year} ${p.title}`.toLowerCase();
    if (f && !hay.includes(f)) return '';
    const isActive = i === currentIndex ? 'active' : '';
    return `<a class="paper-item ${isActive}" onclick="showPaper('${p.rp_id}')" tabindex="0">
      <span class="rp">${p.rp_id}</span>
      <span class="meta">  · ${p.first_author} ${p.year}${p.venue ? ' · ' + escapeHtml(p.venue) : ''}</span>
      <div class="title">${escapeHtml(p.title)}</div>
    </a>`;
  }).join('');
  $('#paperList').innerHTML = html || '<div class="empty">No matches</div>';
}

function filterPapers() { renderPaperList($('#search').value); }

async function showPaper(rpId) {
  const i = manifest.findIndex(p => p.rp_id === rpId);
  if (i === -1) return;
  currentIndex = i;
  currentExtraction = null;
  const p = manifest[i];
  history.replaceState(null, '', `#${rpId}`);
  $('#progress').textContent = `${i + 1} / ${manifest.length}`;
  $('#prevBtn').disabled = i === 0;
  $('#nextBtn').disabled = i === manifest.length - 1;
  $('#bottomNav').style.display = '';
  renderPaperList($('#search').value);

  const findingsHtml = (p.findings || []).slice(0, 12).map(f => `
    <div class="finding">${escapeHtml(typeof f.text === 'string' ? f.text : JSON.stringify(f.text))}
      ${f.page ? `<span class="where">page ${f.page}${f.section ? ' · ' + escapeHtml(f.section) : ''}</span>` : ''}
    </div>
  `).join('');

  const tabs = [
    { id: 'read',     label: '📖 Full text',  enabled: !!p.md_path },
    { id: 'pages',    label: '📄 Pages',       enabled: !!p.pages_dir && p.page_count > 0 },
    { id: 'findings', label: '⭐ Findings',     enabled: (p.findings || []).length > 0 },
    { id: 'data',     label: '🗂 All data',    enabled: !!p.extraction_path },
    { id: 'summary',  label: '📋 Summary',     enabled: true },
  ];

  $('#main').innerHTML = `
    <div class="paper-header">
      <span class="rp-tag">${p.rp_id}</span>
      <h2>${escapeHtml(p.title)}</h2>
      <div class="author-line">${escapeHtml(p.first_author)}${p.authors_count > 1 ? ' et al.' : ''} · ${p.year}${p.venue ? ' · ' + escapeHtml(p.venue) : ''}</div>
      <div class="pills">
        ${p.paper_type ? `<span class="pill">${escapeHtml(p.paper_type)}</span>` : ''}
        ${p.method_family ? `<span class="pill">${escapeHtml(p.method_family)}</span>` : ''}
        ${p.page_count ? `<span class="pill">${p.page_count} pages</span>` : ''}
        ${p.pdf_path ? `<a class="pill" href="${p.pdf_path}" target="_blank" style="color:var(--accent); text-decoration:none">⬇ PDF</a>` : ''}
        ${p.doi ? `<a class="pill" href="https://doi.org/${escapeAttr(p.doi.replace(/^arXiv:/i, '10.48550/arXiv.'))}" target="_blank" style="color:var(--accent); text-decoration:none">DOI</a>` : ''}
      </div>
    </div>
    <div class="modes">
      ${tabs.map(t => `<button class="mode-btn ${currentMode === t.id ? 'active' : ''}" ${!t.enabled ? 'disabled style="opacity:0.4"' : ''} onclick="setMode('${t.id}')">${t.label}</button>`).join('')}
    </div>
    <div class="pane ${currentMode === 'read' ? 'active' : ''}" id="paneRead">
      <div class="loading">Loading paper content…</div>
    </div>
    <div class="pane ${currentMode === 'pages' ? 'active' : ''}" id="panePages">
      <div class="loading">Loading pages…</div>
    </div>
    <div class="pane ${currentMode === 'findings' ? 'active' : ''}" id="paneFindings">
      ${findingsHtml || '<div class="empty">No key findings extracted.</div>'}
    </div>
    <div class="pane ${currentMode === 'data' ? 'active' : ''}" id="paneData">
      <div class="loading">Loading extraction…</div>
    </div>
    <div class="pane ${currentMode === 'summary' ? 'active' : ''}" id="paneSummary">
      <div class="summary-block">
        <strong>Contribution:</strong><br>
        ${p.contribution ? escapeHtml(p.contribution) : '<em>(none extracted)</em>'}
      </div>
      <div class="summary-block">
        <strong>${(p.findings || []).length} key results</strong> · type <em>${escapeHtml(p.paper_type || '?')}</em> · method family <em>${escapeHtml(p.method_family || '?')}</em>
      </div>
    </div>
  `;

  loadPaneIfNeeded(currentMode, p);

  if (window.innerWidth < 900) toggleDrawer(false);
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function loadPaneIfNeeded(mode, p) {
  if (mode === 'read')  loadMarkdown(p);
  if (mode === 'pages') loadPages(p);
  if (mode === 'data')  loadDataTree(p);
}

async function loadMarkdown(p) {
  const pane = $('#paneRead');
  if (!p.md_path) { pane.innerHTML = '<div class="empty">No Markdown for this paper.</div>'; return; }
  if (pane.querySelector('.md')) return;  // cached
  try {
    const md = await (await fetch(p.md_path)).text();
    pane.innerHTML = `<div class="md">${markdownToHtml(md, p)}</div>`;
    bindLightbox(pane);
  } catch (e) {
    pane.innerHTML = `<div class="empty">Failed to load: ${e.message}</div>`;
  }
}

function loadPages(p) {
  const pane = $('#panePages');
  if (!p.pages_dir || !p.page_count) {
    pane.innerHTML = '<div class="empty">Page renders not available for this paper.</div>';
    return;
  }
  if (pane.querySelector('.pages-grid')) return;  // cached
  const pad = (n) => String(n).padStart(2, '0');
  const grid = [];
  for (let i = 1; i <= p.page_count; i++) {
    const src = `${p.pages_dir}page_${pad(i)}.png`;
    grid.push(`<div class="page-thumb" onclick="openLightbox('${src}', '${p.rp_id} · page ${i}')">
      <span class="num">page ${i}</span>
      <img src="${src}" alt="page ${i}" loading="lazy">
    </div>`);
  }
  pane.innerHTML = `<div class="pages-grid">${grid.join('')}</div>`;
}

async function loadDataTree(p) {
  const pane = $('#paneData');
  if (!p.extraction_path) { pane.innerHTML = '<div class="empty">No extraction JSON.</div>'; return; }
  if (pane.querySelector('.data-tree')) return;  // cached
  try {
    if (!currentExtraction) {
      currentExtraction = await (await fetch(p.extraction_path)).json();
    }
    pane.innerHTML = `<div class="data-tree">${renderDataNode(currentExtraction)}</div>`;
  } catch (e) {
    pane.innerHTML = `<div class="empty">Failed: ${e.message}</div>`;
  }
}

function renderDataNode(node, key = null, depth = 0) {
  if (node === null || node === undefined) {
    return `<div class="data-leaf"><div class="v empty">null</div></div>`;
  }
  // {value, ev} leaf
  if (typeof node === 'object' && !Array.isArray(node) && 'value' in node && 'ev' in node) {
    const v = node.value;
    const ev = node.ev || {};
    let valStr;
    if (v === null) valStr = '<span class="v empty">null</span>';
    else if (typeof v === 'object') valStr = `<div class="v"><pre>${escapeHtml(JSON.stringify(v, null, 2))}</pre></div>`;
    else valStr = `<div class="v">${escapeHtml(String(v))}</div>`;
    const evParts = [];
    if (ev.page) evParts.push(`<span class="pg">page ${ev.page}</span>`);
    if (ev.section) evParts.push(`<span>§ ${escapeHtml(ev.section)}</span>`);
    if (ev.quote) evParts.push(`<em>"${escapeHtml(ev.quote.slice(0,140))}${ev.quote.length>140?'…':''}"</em>`);
    return `<div class="data-leaf">${valStr}${evParts.length ? `<div class="ev">${evParts.join(' · ')}</div>` : ''}</div>`;
  }
  if (Array.isArray(node)) {
    if (node.length === 0) return `<div class="data-leaf"><span class="v empty">[]</span></div>`;
    return node.map((item, i) =>
      `<div class="data-list-item">${renderDataNode(item, `[${i}]`, depth + 1)}</div>`).join('');
  }
  if (typeof node === 'object') {
    return Object.entries(node).map(([k, val]) => {
      // Skip sections that are obviously empty
      if (val === null || val === undefined) return '';
      return `<div class="data-section">
        <h3>${escapeHtml(k)}</h3>
        ${renderDataNode(val, k, depth + 1)}
      </div>`;
    }).join('');
  }
  return `<div class="data-leaf"><div class="v">${escapeHtml(String(node))}</div></div>`;
}

function setMode(mode) {
  if (currentIndex === -1) return;
  currentMode = mode;
  $$('.mode-btn').forEach(b => b.classList.remove('active'));
  document.querySelector(`.mode-btn[onclick*="'${mode}'"]`)?.classList.add('active');
  $$('.pane').forEach(p => p.classList.remove('active'));
  document.getElementById('pane' + mode.charAt(0).toUpperCase() + mode.slice(1))?.classList.add('active');
  loadPaneIfNeeded(mode, manifest[currentIndex]);
}

function navigate(delta) {
  const i = currentIndex + delta;
  if (i >= 0 && i < manifest.length) showPaper(manifest[i].rp_id);
}

function toggleDrawer(force) {
  const drawer = $('#drawer');
  const bg = $('.drawer-bg');
  const open = (force === undefined) ? !drawer.classList.contains('open') : force;
  drawer.classList.toggle('open', open);
  bg.classList.toggle('open', open);
}

function toggleFocus() { document.body.classList.toggle('focus-mode'); }
function toggleWide()  {
  document.body.classList.toggle('wide');
  $('#wideBtn').classList.toggle('active', document.body.classList.contains('wide'));
}

function openLightbox(src, info) {
  $('#lightboxImg').src = src;
  $('#lightboxInfo').textContent = info || '';
  $('#lightbox').classList.add('open');
}
function closeLightbox(e) {
  // Only close if clicking the bg or the close btn (not the img zoom-out)
  if (e && e.target && e.target.tagName === 'IMG') {
    $('#lightbox').classList.remove('open');
    return;
  }
  $('#lightbox').classList.remove('open');
}

function bindLightbox(scope) {
  scope.querySelectorAll('img').forEach(img => {
    img.addEventListener('click', () => openLightbox(img.src, img.alt));
  });
}

function escapeHtml(s) {
  if (typeof s !== 'string') s = String(s ?? '');
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function escapeAttr(s) { return escapeHtml(s).replace(/"/g, '&quot;'); }

/* Markdown → HTML.  Rewrites figure paths so they resolve from
   paper_reader.html (located at Assignment01/) instead of from
   papers_md/*.md. */
function markdownToHtml(md, paper) {
  md = md.replace(/<!-- page (\d+) -->/g, (_, n) => `\n<div class="page-marker">page ${n}</div>\n`);
  md = md.replace(/<!--[\s\S]*?-->/g, '');
  md = md.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${escapeHtml(code.replace(/^\n/, ''))}</code></pre>`);
  md = md.replace(/((?:^\|.*\|\s*$\n?){2,})/gm, block => {
    const rows = block.trim().split('\n').map(r => r.replace(/^\||\|$/g, '').split('|').map(c => c.trim()));
    if (rows.length < 2) return block;
    const head = rows[0]; const body = rows.slice(2);
    return `<table><thead><tr>${head.map(c => `<th>${inline(c)}</th>`).join('')}</tr></thead>
      <tbody>${body.map(r => '<tr>' + r.map(c => `<td>${inline(c)}</td>`).join('') + '</tr>').join('')}</tbody></table>`;
  });
  const blocks = md.split(/\n{2,}/);
  return blocks.map(block => {
    block = block.trim();
    if (!block) return '';
    if (block.startsWith('<')) return block;
    let m;
    if ((m = block.match(/^(#{1,6})\s+(.+)$/))) {
      return `<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`;
    }
    if (block.startsWith('>')) {
      return `<blockquote>${inline(block.split('\n').map(l => l.replace(/^>\s?/, '')).join(' '))}</blockquote>`;
    }
    if (/^\s*[-*]\s+/.test(block)) {
      return `<ul>${block.split('\n').map(l => l.replace(/^\s*[-*]\s+/, '').trim()).filter(Boolean).map(i => `<li>${inline(i)}</li>`).join('')}</ul>`;
    }
    if (/^\s*\d+\.\s+/.test(block)) {
      return `<ol>${block.split('\n').map(l => l.replace(/^\s*\d+\.\s+/, '').trim()).filter(Boolean).map(i => `<li>${inline(i)}</li>`).join('')}</ol>`;
    }
    return `<p>${inline(block.replace(/\n/g, ' '))}</p>`;
  }).join('\n');

  function inline(t) {
    t = t.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_, alt, src) => {
      // Markdown comes from papers_md/*.md, which contains paths like
      // "../figures/RP09_fig01.jpg".  paper_reader.html lives at
      // Assignment01/, so "../figures/" would resolve one level too high.
      // Strip the "../" so the path becomes "figures/..." (correct from here).
      const fixed = src.replace(/^\.\.\//, '');
      return `<img src="${escapeAttr(fixed)}" alt="${escapeHtml(alt)}" loading="lazy">`;
    });
    t = t.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, text, href) =>
      `<a href="${escapeAttr(href)}" target="_blank">${escapeHtml(text)}</a>`);
    t = t.replace(/`([^`]+)`/g, (_, c) => `<code>${escapeHtml(c)}</code>`);
    t = t.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    t = t.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    t = t.replace(/\^([^\^]+)\^/g, '<sup>$1</sup>');
    t = t.replace(/~([^~]+)~/g, '<sub>$1</sub>');
    return t;
  }
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'Escape') {
    if ($('#lightbox').classList.contains('open')) closeLightbox();
    else toggleDrawer(false);
  }
  if (e.key === 'ArrowLeft' || e.key === 'k') navigate(-1);
  if (e.key === 'ArrowRight' || e.key === 'j') navigate(1);
  if (e.key === 'f') toggleFocus();
  if (e.key === 'w') toggleWide();
  if (e.key === '/') { e.preventDefault(); $('#search').focus(); toggleDrawer(true); }
  // Tab numbers 1-5 switch panes
  if (['1','2','3','4','5'].includes(e.key)) {
    const tabs = ['read','pages','findings','data','summary'];
    setMode(tabs[parseInt(e.key,10) - 1]);
  }
});

window.addEventListener('hashchange', () => {
  const rp = location.hash.replace('#', '');
  if (rp) showPaper(rp);
});

loadManifest();
</script>
</body>
</html>
"""


def main():
    print("Building manifest from extractions/…")
    manifest = build_manifest()
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  Wrote {OUT_MANIFEST.relative_to(HERE)}: {OUT_MANIFEST.stat().st_size:,} bytes ({len(manifest)} papers)")

    OUT_HTML.write_text(HTML_TEMPLATE, encoding="utf-8")
    print(f"  Wrote {OUT_HTML.relative_to(HERE)}: {OUT_HTML.stat().st_size:,} bytes")

    print()
    print("Done. Open in browser:")
    print(f"  file:///{OUT_HTML.as_posix()}")
    print(f"  https://elrashid.github.io/AI503/Assignment01/paper_reader.html")


if __name__ == "__main__":
    main()
