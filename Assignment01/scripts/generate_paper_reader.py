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
    papers = []
    for fp in sorted(EXTR_DIR.glob("RP*_extraction.json")):
        rp = fp.name.split("_")[0]
        with open(fp, encoding="utf-8") as f:
            j = json.load(f)
        title = v(j.get("title")) or "?"
        first_author = v(j.get("first_author")) or "?"
        year = v(j.get("year")) or 0
        venue = v(j.get("venue")) or ""
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

        papers.append({
            "rp_id": rp,
            "title": title,
            "first_author": first_author,
            "year": int(year) if str(year).isdigit() else 0,
            "venue": venue,
            "paper_type": ptype,
            "method_family": mfam,
            "contribution": contribution,
            "findings": findings,
            "md_path": md_path,
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
    --bg:        #fdf6e3;   /* warm cream — easy on the eyes */
    --bg-alt:   #eee8d5;
    --ink:      #073642;   /* deep teal-grey, not pure black */
    --ink-soft: #586e75;
    --muted:    #93a1a1;
    --accent:   #268bd2;   /* calm blue */
    --accent-soft: #cfe6f5;
    --warn:     #b58900;
    --good:     #859900;
    --border:   #d6cfb8;
    --shadow:   0 2px 12px rgba(7, 54, 66, 0.08);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg:        #002b36;
      --bg-alt:   #073642;
      --ink:      #eee8d5;
      --ink-soft: #93a1a1;
      --muted:    #586e75;
      --accent:   #268bd2;
      --accent-soft: #073642;
      --border:   #073642;
    }
  }
  * { box-sizing: border-box; }
  html { font-size: 18px; }                          /* big base size */
  body {
    margin: 0;
    font-family: -apple-system, "Segoe UI", "Atkinson Hyperlegible", "Open Sans",
                 system-ui, sans-serif;
    background: var(--bg);
    color: var(--ink);
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
    overscroll-behavior-y: contain;
  }

  /* === Top bar (sticky, calm) === */
  .topbar {
    position: sticky; top: 0; z-index: 30;
    display: flex; align-items: center; gap: 12px;
    padding: 10px 14px;
    background: var(--bg-alt);
    border-bottom: 1px solid var(--border);
    backdrop-filter: blur(6px);
  }
  .topbar h1 { font-size: 17px; margin: 0; font-weight: 600; flex: 1; }
  .topbar h1 small { font-size: 13px; color: var(--ink-soft); font-weight: 400; }
  .iconbtn {
    appearance: none; border: 1px solid var(--border);
    background: var(--bg);
    color: var(--ink);
    width: 44px; height: 44px;            /* touch target */
    border-radius: 8px;
    cursor: pointer; font-size: 18px;
    display: flex; align-items: center; justify-content: center;
  }
  .iconbtn:active { background: var(--bg-alt); }
  .progress {
    color: var(--ink-soft); font-size: 13px;
    padding: 4px 10px; background: var(--bg); border-radius: 6px;
    border: 1px solid var(--border);
  }

  /* === Paper picker drawer (full-screen on mobile, side drawer on desktop) === */
  .drawer-bg {
    position: fixed; inset: 0; background: rgba(0,0,0,0.35);
    opacity: 0; pointer-events: none; transition: opacity 0.18s;
    z-index: 40;
  }
  .drawer-bg.open { opacity: 1; pointer-events: auto; }
  .drawer {
    position: fixed; top: 0; left: 0; bottom: 0;
    width: min(420px, 90vw);
    background: var(--bg);
    box-shadow: var(--shadow);
    z-index: 50;
    transform: translateX(-100%);
    transition: transform 0.22s;
    display: flex; flex-direction: column;
  }
  .drawer.open { transform: translateX(0); }
  .drawer-head { padding: 12px 14px; border-bottom: 1px solid var(--border); display: flex; gap: 8px; align-items: center; }
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
  .paper-item.active { background: var(--accent-soft); border-left: 4px solid var(--accent); padding-left: 14px; }
  .paper-item .rp { font-family: ui-monospace, SFMono-Regular, monospace; color: var(--accent); font-weight: 600; font-size: 14px; }
  .paper-item .meta { color: var(--ink-soft); font-size: 13px; margin-top: 2px; }
  .paper-item .title { font-size: 15px; margin-top: 4px; }

  /* === Main content === */
  main {
    max-width: 720px;                     /* readable line length */
    margin: 0 auto;
    padding: 20px 18px 80px;
  }
  .paper-header {
    margin: 8px 0 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }
  .paper-header .rp-tag {
    display: inline-block;
    background: var(--accent); color: white;
    padding: 4px 10px; border-radius: 6px;
    font-family: ui-monospace, monospace; font-size: 14px; font-weight: 600;
  }
  .paper-header h2 {
    font-size: 24px; line-height: 1.3;
    margin: 12px 0 8px;
  }
  .paper-header .author-line {
    color: var(--ink-soft); font-size: 15px;
  }
  .paper-header .pills {
    margin-top: 12px;
  }
  .pill {
    display: inline-block; font-size: 12px;
    padding: 3px 8px; margin-right: 6px;
    border-radius: 12px; border: 1px solid var(--border);
    color: var(--ink-soft);
  }

  /* === Mode toggle === */
  .modes {
    display: flex; gap: 4px; margin-bottom: 16px;
    background: var(--bg-alt); padding: 4px; border-radius: 10px;
  }
  .mode-btn {
    flex: 1; padding: 10px; border: none; cursor: pointer;
    background: transparent; color: var(--ink-soft);
    border-radius: 6px; font-size: 14px; font-weight: 500;
  }
  .mode-btn.active { background: var(--bg); color: var(--ink); box-shadow: var(--shadow); }

  /* === Content panes === */
  .pane { display: none; animation: fadeIn 0.18s; }
  .pane.active { display: block; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }

  /* Markdown body — generous whitespace */
  .md h1 { font-size: 22px; margin: 32px 0 12px; padding-top: 8px; }
  .md h2 { font-size: 19px; margin: 28px 0 10px; color: var(--ink); }
  .md h3 { font-size: 17px; margin: 24px 0 8px; color: var(--ink); }
  .md p { margin: 0 0 18px; line-height: 1.7; }
  .md a { color: var(--accent); }
  .md code { background: var(--bg-alt); padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }
  .md pre {
    background: var(--bg-alt); padding: 12px; border-radius: 6px; overflow-x: auto;
    font-size: 14px;
  }
  .md table { border-collapse: collapse; margin: 16px 0; font-size: 14px; width: 100%; }
  .md table th, .md table td { border: 1px solid var(--border); padding: 6px 10px; }
  .md table th { background: var(--bg-alt); }
  .md img { max-width: 100%; height: auto; border-radius: 4px; margin: 12px 0; }
  .md blockquote {
    border-left: 3px solid var(--accent); margin: 16px 0; padding: 4px 12px;
    color: var(--ink-soft); background: var(--bg-alt); border-radius: 0 4px 4px 0;
  }
  .md hr, .md .page-break {
    border: none; border-top: 1px dashed var(--border); margin: 32px 0;
  }
  .md .page-marker {
    color: var(--muted); font-size: 11px; text-align: center; margin: 24px 0 8px;
    text-transform: uppercase; letter-spacing: 1px;
  }

  /* Findings cards */
  .finding {
    background: var(--bg);
    border: 1px solid var(--border);
    border-left: 3px solid var(--good);
    padding: 14px 16px; margin-bottom: 12px;
    border-radius: 6px;
    font-size: 16px;
  }
  .finding .where {
    display: block; font-size: 12px; color: var(--ink-soft); margin-top: 8px;
    font-family: ui-monospace, monospace;
  }
  .summary-block {
    background: var(--bg-alt); padding: 14px 16px;
    border-radius: 8px; margin-bottom: 24px;
  }
  .summary-block strong { color: var(--ink); }

  /* Bottom nav (mobile prev/next) */
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
  main { padding-bottom: 90px; }

  /* === Wide screens: drawer always visible === */
  @media (min-width: 900px) {
    body { display: grid; grid-template-columns: 320px 1fr; }
    .topbar { grid-column: 1 / -1; }
    .drawer {
      position: static; transform: none; box-shadow: none;
      border-right: 1px solid var(--border);
      width: 320px; height: calc(100vh - 65px);
    }
    .drawer-bg, .iconbtn.menu { display: none; }
    main { padding: 24px 40px 40px; }
    .bottom-nav { left: 320px; }
  }

  /* Loading + empty states */
  .loading, .empty {
    text-align: center; padding: 60px 20px; color: var(--ink-soft);
  }

  /* Focus mode (just markdown, hide everything else) */
  body.focus-mode .topbar > *:not(h1):not(.iconbtn.focus) { display: none; }
  body.focus-mode .drawer-bg, body.focus-mode .drawer { display: none; }
  body.focus-mode .modes, body.focus-mode .paper-header .pills { display: none; }
  body.focus-mode .bottom-nav { display: none; }
  body.focus-mode main { max-width: 680px; padding-bottom: 40px; }
  body.focus-mode { grid-template-columns: 1fr !important; }

  /* Dyslexia-friendly toggle */
  body.dyslexia-friendly { font-family: "Atkinson Hyperlegible", "Open Dyslexic", system-ui, sans-serif; letter-spacing: 0.02em; }
</style>
</head>
<body>

<header class="topbar">
  <button class="iconbtn menu" onclick="toggleDrawer()" aria-label="Open paper list">☰</button>
  <h1>AI503 Reader <small id="paperCount">— loading…</small></h1>
  <span class="progress" id="progress">—</span>
  <button class="iconbtn focus" onclick="toggleFocus()" aria-label="Toggle focus mode" title="Focus mode (F)">🎯</button>
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
  <button class="nav-btn" id="prevBtn" onclick="navigate(-1)">← Previous paper</button>
  <button class="nav-btn" id="nextBtn" onclick="navigate(1)">Next paper →</button>
</nav>

<script>
let manifest = [];
let currentIndex = -1;
let currentMode = 'read';

const $ = (q) => document.querySelector(q);
const $$ = (q) => document.querySelectorAll(q);

async function loadManifest() {
  try {
    const r = await fetch('paper_reader.json');
    manifest = await r.json();
    $('#paperCount').textContent = `· ${manifest.length} papers`;
    renderPaperList();
    // Open the first paper, or the one in URL hash
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
  const p = manifest[i];
  history.replaceState(null, '', `#${rpId}`);
  $('#progress').textContent = `${i + 1} / ${manifest.length}`;
  $('#prevBtn').disabled = i === 0;
  $('#nextBtn').disabled = i === manifest.length - 1;
  $('#bottomNav').style.display = '';
  renderPaperList($('#search').value);

  // Render header + mode tabs
  const findingsHtml = (p.findings || []).slice(0, 8).map(f => `
    <div class="finding">${escapeHtml(typeof f.text === 'string' ? f.text : JSON.stringify(f.text))}
      ${f.page ? `<span class="where">page ${f.page}${f.section ? ' · ' + escapeHtml(f.section) : ''}</span>` : ''}
    </div>
  `).join('');

  $('#main').innerHTML = `
    <div class="paper-header">
      <span class="rp-tag">${p.rp_id}</span>
      <h2>${escapeHtml(p.title)}</h2>
      <div class="author-line">${escapeHtml(p.first_author)} · ${p.year}${p.venue ? ' · ' + escapeHtml(p.venue) : ''}</div>
      <div class="pills">
        ${p.paper_type ? `<span class="pill">${escapeHtml(p.paper_type)}</span>` : ''}
        ${p.method_family ? `<span class="pill">${escapeHtml(p.method_family)}</span>` : ''}
      </div>
    </div>
    <div class="modes">
      <button class="mode-btn ${currentMode === 'read' ? 'active' : ''}" onclick="setMode('read')">📖 Full text</button>
      <button class="mode-btn ${currentMode === 'findings' ? 'active' : ''}" onclick="setMode('findings')">⭐ Key findings</button>
      <button class="mode-btn ${currentMode === 'summary' ? 'active' : ''}" onclick="setMode('summary')">📋 Summary</button>
    </div>
    <div class="pane ${currentMode === 'read' ? 'active' : ''}" id="paneRead">
      <div class="loading">Loading paper content…</div>
    </div>
    <div class="pane ${currentMode === 'findings' ? 'active' : ''}" id="paneFindings">
      ${findingsHtml || '<div class="empty">No key findings extracted.</div>'}
    </div>
    <div class="pane ${currentMode === 'summary' ? 'active' : ''}" id="paneSummary">
      <div class="summary-block">
        <strong>Contribution:</strong><br>
        ${p.contribution ? escapeHtml(p.contribution) : '<em>(none extracted)</em>'}
      </div>
      <div class="summary-block">
        <strong>${(p.findings || []).length} key results</strong> · paper type <em>${escapeHtml(p.paper_type || '?')}</em> · method family <em>${escapeHtml(p.method_family || '?')}</em>
      </div>
      <p style="color: var(--ink-soft); font-size: 14px;">
        Switch to "Key findings" for the full list, or "Full text" for the markdown render.
      </p>
    </div>
  `;

  // Lazy-load the markdown only when read mode is shown
  if (currentMode === 'read') loadMarkdown(p);

  // Close drawer on mobile after picking
  if (window.innerWidth < 900) toggleDrawer(false);
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function loadMarkdown(p) {
  if (!p.md_path) {
    $('#paneRead').innerHTML = '<div class="empty">No Markdown available for this paper.</div>';
    return;
  }
  try {
    const r = await fetch(p.md_path);
    const md = await r.text();
    $('#paneRead').innerHTML = `<div class="md">${markdownToHtml(md, p.rp_id)}</div>`;
  } catch (e) {
    $('#paneRead').innerHTML = `<div class="empty">Failed to load: ${e.message}</div>`;
  }
}

function setMode(mode) {
  currentMode = mode;
  $$('.mode-btn').forEach(b => b.classList.remove('active'));
  document.querySelector(`.mode-btn[onclick*="'${mode}'"]`).classList.add('active');
  $$('.pane').forEach(p => p.classList.remove('active'));
  document.getElementById('pane' + mode.charAt(0).toUpperCase() + mode.slice(1)).classList.add('active');
  if (mode === 'read' && currentIndex >= 0) {
    const pane = $('#paneRead');
    if (pane && !pane.querySelector('.md')) loadMarkdown(manifest[currentIndex]);
  }
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

function toggleFocus() {
  document.body.classList.toggle('focus-mode');
}

function escapeHtml(s) {
  if (typeof s !== 'string') s = String(s ?? '');
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

/* Tiny Markdown → HTML renderer (no deps).
   Supports: headings, paragraphs, bold, italic, code (inline + block),
   links, images, lists (bulleted + ordered), tables, blockquotes,
   <!-- page N --> separators. Good enough for our marker output. */
function markdownToHtml(md, paperId) {
  // Page-break comments → visible separator
  md = md.replace(/<!-- page (\d+) -->/g, (_, n) => `\n<div class="page-marker">page ${n}</div>\n`);
  // Strip other HTML comments
  md = md.replace(/<!--[\s\S]*?-->/g, '');

  // Code fences
  md = md.replace(/```([\s\S]*?)```/g, (_, code) =>
    `<pre><code>${escapeHtml(code.replace(/^\n/, ''))}</code></pre>`);

  // Tables (simple pipe tables)
  md = md.replace(/((?:^\|.*\|\s*$\n?){2,})/gm, block => {
    const rows = block.trim().split('\n').map(r =>
      r.replace(/^\||\|$/g, '').split('|').map(c => c.trim()));
    if (rows.length < 2) return block;
    const head = rows[0];
    const body = rows.slice(2);  // skip --- separator
    const ths = head.map(c => `<th>${inline(c)}</th>`).join('');
    const tds = body.map(r => '<tr>' + r.map(c => `<td>${inline(c)}</td>`).join('') + '</tr>').join('');
    return `<table><thead><tr>${ths}</tr></thead><tbody>${tds}</tbody></table>`;
  });

  // Block-level: split on blank lines
  const blocks = md.split(/\n{2,}/);
  return blocks.map(block => {
    block = block.trim();
    if (!block) return '';
    if (block.startsWith('<')) return block;  // already HTML (table, page-marker)
    // Heading
    let m;
    if ((m = block.match(/^(#{1,6})\s+(.+)$/))) {
      const lvl = m[1].length;
      return `<h${lvl}>${inline(m[2])}</h${lvl}>`;
    }
    // Blockquote
    if (block.startsWith('>')) {
      const t = block.split('\n').map(l => l.replace(/^>\s?/, '')).join(' ');
      return `<blockquote>${inline(t)}</blockquote>`;
    }
    // List (bulleted)
    if (/^\s*[-*]\s+/.test(block)) {
      const items = block.split('\n').map(l => l.replace(/^\s*[-*]\s+/, '').trim())
        .filter(Boolean).map(i => `<li>${inline(i)}</li>`).join('');
      return `<ul>${items}</ul>`;
    }
    // Ordered list
    if (/^\s*\d+\.\s+/.test(block)) {
      const items = block.split('\n').map(l => l.replace(/^\s*\d+\.\s+/, '').trim())
        .filter(Boolean).map(i => `<li>${inline(i)}</li>`).join('');
      return `<ol>${items}</ol>`;
    }
    // Paragraph
    return `<p>${inline(block.replace(/\n/g, ' '))}</p>`;
  }).join('\n');

  function inline(t) {
    // Image → <img>
    t = t.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_, alt, src) =>
      `<img src="${escapeAttr(src)}" alt="${escapeHtml(alt)}" loading="lazy">`);
    // Link → <a>
    t = t.replace(/\[([^\]]+)\]\(([^)]+)\)/g,
      (_, text, href) => `<a href="${escapeAttr(href)}" target="_blank">${escapeHtml(text)}</a>`);
    // Inline code
    t = t.replace(/`([^`]+)`/g, (_, c) => `<code>${escapeHtml(c)}</code>`);
    // Bold then italic (markdown order matters)
    t = t.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    t = t.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    // Sup/sub (our render uses ^x^ and ~x~)
    t = t.replace(/\^([^\^]+)\^/g, '<sup>$1</sup>');
    t = t.replace(/~([^~]+)~/g, '<sub>$1</sub>');
    return t;
  }
  function escapeAttr(s) { return escapeHtml(s).replace(/"/g, '&quot;'); }
}

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft' || e.key === 'k') navigate(-1);
  if (e.key === 'ArrowRight' || e.key === 'j') navigate(1);
  if (e.key === 'f') toggleFocus();
  if (e.key === '/') { e.preventDefault(); $('#search').focus(); toggleDrawer(true); }
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
