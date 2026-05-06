"""
Generate paper_explorer.html — a single self-contained interactive viewer.

Reads:
  - extractions/*.json
  - papers_pages/manifest.json

Output:
  - paper_explorer.html (open in any browser)

UI:
  - Left:  paper list  (click a paper to load its extraction)
  - Mid:   extraction tree (every traced field; hover or click → highlight source)
  - Right: page image with yellow bbox overlay on the source block

Page PNGs stay external (loaded via relative paths to papers_pages/<dir>/page_NN.png).

Usage:
  python generate_paper_explorer.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
EXTRACTIONS = HERE / "extractions"
MANIFEST = HERE / "papers_pages" / "manifest.json"
OUT = HERE / "paper_explorer.html"


def load_extractions():
    """Load all RPxx_extraction.json files into a dict keyed by RP id."""
    data = {}
    for p in sorted(EXTRACTIONS.glob("RP*_extraction.json")):
        try:
            ext = json.loads(p.read_text(encoding="utf-8"))
            rp_id = ext.get("rp_id") or p.stem.split("_")[0]
            data[rp_id] = ext
        except Exception as e:
            print(f"WARN: skipped {p.name}: {e}")
    return data


def build_html(extractions, manifest):
    """Build the full HTML document with extractions embedded as JS data."""
    payload = {
        "manifest": manifest,
        "extractions": extractions,
    }
    payload_js = json.dumps(payload, ensure_ascii=False)

    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI503 SLR Paper Explorer</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #0f172a; color: #e2e8f0; }
  #app { display: grid; grid-template-columns: 220px 480px 1fr; height: 100vh; overflow: hidden; }
  .panel { overflow: auto; border-right: 1px solid #334155; }
  #papers { background: #0b1220; padding: 8px 0; }
  #papers h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em;
               color: #94a3b8; padding: 0 12px; margin: 8px 0; }
  .paper-row { padding: 6px 12px; cursor: pointer; font-size: 13px; border-left: 3px solid transparent; }
  .paper-row:hover { background: #1e293b; }
  .paper-row.active { background: #1e3a8a; border-left-color: #60a5fa; color: #fff; }
  .paper-row .y { color: #64748b; font-size: 11px; }

  #tree { background: #0f172a; padding: 12px 16px; }
  #tree h1 { font-size: 18px; margin: 0 0 4px; color: #f1f5f9; }
  #tree .meta { color: #94a3b8; font-size: 12px; margin-bottom: 12px; }
  .field { margin: 2px 0; padding: 4px 6px; border-radius: 3px; cursor: pointer;
           font-size: 12.5px; line-height: 1.4; }
  .field:hover, .field.hot { background: #1e293b; }
  .field.hot { outline: 1px solid #60a5fa; }
  .fname { color: #93c5fd; font-weight: 600; }
  .ftype { color: #64748b; font-size: 11px; margin-left: 4px; }
  .fval  { color: #e2e8f0; }
  .fval.empty { color: #475569; font-style: italic; }
  .fev   { color: #64748b; font-size: 10.5px; margin-top: 2px; }
  .fev .pg { color: #fbbf24; font-weight: 600; }
  .fev .sec { color: #a3e635; }
  .arr-item { margin-left: 14px; border-left: 2px solid #334155; padding-left: 8px; }
  .obj-block { margin-left: 14px; border-left: 2px solid #334155; padding-left: 8px; }
  details > summary { cursor: pointer; padding: 4px 0; color: #cbd5e1; font-weight: 600; user-select: none; }
  details[open] > summary { color: #f1f5f9; }
  .gap-pos { color: #4ade80; }
  .gap-neg { color: #f87171; }

  #viewer { background: #1e293b; position: relative; }
  #viewer-header { padding: 8px 12px; background: #0b1220; border-bottom: 1px solid #334155;
                   font-size: 12px; color: #cbd5e1; height: 36px; display: flex; align-items: center; gap: 12px; }
  #viewer-header .crumb { color: #fbbf24; font-weight: 600; }
  #viewer-header .quote { color: #94a3b8; font-style: italic; flex: 1;
                          overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .page-nav { display: flex; align-items: center; gap: 4px; }
  .page-nav button { background: #1e293b; color: #e2e8f0; border: 1px solid #334155;
                     padding: 3px 10px; cursor: pointer; border-radius: 3px; font-size: 12px; }
  .page-nav button:hover:not(:disabled) { background: #334155; border-color: #60a5fa; }
  .page-nav button:disabled { opacity: 0.3; cursor: not-allowed; }
  .page-nav .counter { color: #cbd5e1; font-variant-numeric: tabular-nums; min-width: 70px;
                       text-align: center; font-size: 11.5px; }
  .page-nav .counter input { width: 36px; background: #0b1220; color: #fbbf24; border: 1px solid #334155;
                             text-align: center; font-size: 11.5px; padding: 1px 2px; border-radius: 2px; }
  .page-nav .counter .src-marker { color: #fbbf24; font-weight: 600; margin-left: 4px; }
  #viewer-stage { position: relative; height: calc(100vh - 36px); overflow: auto; padding: 16px; }
  #viewer-stage .page-wrap { position: relative; display: inline-block; box-shadow: 0 6px 24px rgba(0,0,0,0.4); }
  #viewer-stage img { display: block; max-width: 100%; }
  .bbox-overlay { position: absolute; border: 3px solid #fbbf24;
                  background: rgba(251, 191, 36, 0.18); pointer-events: none;
                  box-shadow: 0 0 0 3px rgba(251, 191, 36, 0.4); transition: all 0.15s ease; }
  .empty-stage { color: #64748b; padding: 40px; font-size: 13px; }
</style>
</head>
<body>
<div id="app">
  <aside class="panel" id="papers">
    <h2>Papers (50)</h2>
    <div id="paper-list"></div>
  </aside>
  <section class="panel" id="tree">
    <div class="empty-stage">Pick a paper from the left to view its extraction.</div>
  </section>
  <section id="viewer">
    <div id="viewer-header">
      <span id="vh-rp">—</span>
      <span class="crumb" id="vh-loc">no source selected</span>
      <span class="quote" id="vh-quote"></span>
      <div class="page-nav">
        <button id="nav-prev" title="Previous page (←)">◀</button>
        <span class="counter">
          <input id="nav-page" type="number" min="1" value="" /> / <span id="nav-total">—</span>
          <span class="src-marker" id="nav-src" title="Jump back to the source page"></span>
        </span>
        <button id="nav-next" title="Next page (→)">▶</button>
      </div>
    </div>
    <div id="viewer-stage">
      <div class="empty-stage">Hover or click a field with a page reference to see its source.</div>
    </div>
  </section>
</div>

<script>
const DATA = __DATA__;
const SCALE = (DATA.manifest && DATA.manifest.scale_factor) || (150/72);
let CURRENT = null;       // RP id of the loaded paper
let CURRENT_PAGE = null;  // page number currently shown in viewer
let CURRENT_EV = null;    // ev object of the field that triggered the source view

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function isEv(obj) {
  return obj && typeof obj === "object" && !Array.isArray(obj)
      && "value" in obj && "ev" in obj;
}

// Detect a metrics_results-style entry: value is an object with model+metric+value keys.
function isMetricEntry(node) {
  return isEv(node)
      && node.value && typeof node.value === "object" && !Array.isArray(node.value)
      && "model" in node.value && "metric" in node.value && "value" in node.value;
}

// Render one metric tuple as a compact, readable one-liner instead of raw JSON.
// Example: "OPT-175B / GPTQ / 4-bit / WikiText-2 / Perplexity = 8.37 (vs FP16 8.34, Δ=0.03)"
function metricSummary(v) {
  const parts = [];
  if (v.model) parts.push(v.model);
  if (v.method) parts.push(v.method);
  if (v.bit_width) parts.push(v.bit_width);
  if (v.group_size) parts.push(v.group_size);
  if (v.dataset) parts.push(v.dataset);
  let line = parts.join(" / ") + " / " + (v.metric || "?") + " = " + v.value;
  if (v.baseline_label != null && v.baseline_value != null) {
    let cmp = " (vs " + v.baseline_label + " " + v.baseline_value;
    if (v.delta != null) cmp += ", Δ=" + v.delta + (v.delta_unit && v.delta_unit !== "absolute" ? " " + v.delta_unit : "");
    cmp += ")";
    line += cmp;
  }
  if (v.source_kind === "cited") line += "  [cited]";
  if (v.headline === true) line = "★ " + line;
  return line;
}

function renderField(name, node, depth) {
  // Metrics_results entry — friendly summary line
  if (isMetricEntry(node)) {
    const div = el("div", "field");
    const label = el("span", "fname", name);
    const valStr = metricSummary(node.value);
    div.appendChild(label);
    div.appendChild(el("span", "ftype", " ="));
    div.appendChild(el("span", "fval", " " + valStr));
    if (node.ev && (node.ev.page || node.ev.section)) {
      const ev = el("div", "fev");
      const pg = node.ev.page == null ? "—" : "p" + node.ev.page;
      const sec = node.ev.section || "";
      const tbl = node.ev.source_table ? " · " + escapeHtml(node.ev.source_table) : "";
      ev.innerHTML = `<span class="pg">${pg}</span> · <span class="sec">${escapeHtml(sec)}</span>${tbl}`;
      div.appendChild(ev);
    }
    div.addEventListener("mouseenter", () => showSource(node.ev, name));
    div.addEventListener("click", () => showSource(node.ev, name));
    return div;
  }
  // Leaf = {value, ev}
  if (isEv(node)) {
    const div = el("div", "field");
    const label = el("span", "fname", name);
    const val = node.value;
    let valStr;
    if (val === null || val === undefined) { valStr = "null"; }
    else if (typeof val === "object")        { valStr = JSON.stringify(val); }
    else                                     { valStr = String(val); }
    if (valStr.length > 220) valStr = valStr.slice(0, 220) + "…";
    div.appendChild(label);
    div.appendChild(el("span", "ftype", " ="));
    const vEl = el("span", "fval" + (val === null ? " empty" : ""), " " + valStr);
    div.appendChild(vEl);
    if (node.ev && (node.ev.page || node.ev.section)) {
      const ev = el("div", "fev");
      const pg = node.ev.page == null ? "—" : "p" + node.ev.page;
      const sec = node.ev.section || "";
      ev.innerHTML = `<span class="pg">${pg}</span> · <span class="sec">${escapeHtml(sec)}</span>`;
      div.appendChild(ev);
    }
    div.addEventListener("mouseenter", () => showSource(node.ev, name));
    div.addEventListener("click", () => showSource(node.ev, name));
    return div;
  }
  // Array of evs
  if (Array.isArray(node)) {
    const block = el("div");
    const sum = el("details");
    sum.open = node.length <= 6;
    sum.appendChild(el("summary", null, `${name} [${node.length}]`));
    const inner = el("div", "arr-item");
    node.forEach((item, i) => {
      inner.appendChild(renderField(`${name}[${i}]`, item, depth + 1));
    });
    sum.appendChild(inner);
    block.appendChild(sum);
    return block;
  }
  // Nested object (e.g. gap_signals or speedup_claim)
  if (node && typeof node === "object") {
    const block = el("div");
    const sum = el("details");
    sum.open = true;
    sum.appendChild(el("summary", null, name));
    const inner = el("div", "obj-block");
    Object.keys(node).forEach(k => {
      inner.appendChild(renderField(k, node[k], depth + 1));
    });
    sum.appendChild(inner);
    block.appendChild(sum);
    return block;
  }
  // Plain scalar (e.g. rp_id)
  const div = el("div", "field");
  div.appendChild(el("span", "fname", name));
  div.appendChild(el("span", "ftype", " ="));
  div.appendChild(el("span", "fval", " " + String(node)));
  return div;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"
  }[c]));
}

function showSource(ev, fieldName) {
  if (!ev || !CURRENT) return;
  const stage = document.getElementById("viewer-stage");
  const vhLoc = document.getElementById("vh-loc");
  const vhQuote = document.getElementById("vh-quote");
  if (ev.page == null || ev.section === "absence" || ev.section === "not_found") {
    stage.innerHTML = `<div class="empty-stage">No source page for <b>${escapeHtml(fieldName)}</b>
      (section: <i>${escapeHtml(ev.section || "—")}</i>)</div>`;
    vhLoc.textContent = ev.section || "—";
    vhQuote.textContent = "";
    CURRENT_EV = null;
    CURRENT_PAGE = null;
    updatePageNav();
    return;
  }
  CURRENT_EV = ev;
  vhQuote.textContent = ev.quote ? "“" + ev.quote.slice(0, 180) + "”" : "";
  renderPage(ev.page);
}

function renderPage(pageNum) {
  if (!CURRENT) return;
  const stage = document.getElementById("viewer-stage");
  const vhLoc = document.getElementById("vh-loc");
  const pmeta = (DATA.manifest.papers || {})[CURRENT];
  if (!pmeta) {
    stage.innerHTML = `<div class="empty-stage">Manifest missing for ${CURRENT}.</div>`;
    return;
  }
  const total = pmeta.page_count || (pmeta.pages || []).length;
  if (pageNum < 1) pageNum = 1;
  if (pageNum > total) pageNum = total;
  CURRENT_PAGE = pageNum;
  const pageInfo = (pmeta.pages || []).find(p => p.page === pageNum);
  const file = pageInfo ? pageInfo.file : `page_${String(pageNum).padStart(2,"0")}.png`;
  const url = `papers_pages/${pmeta.dir}/${file}`;

  const isSourcePage = CURRENT_EV && CURRENT_EV.page === pageNum;
  const sec = (isSourcePage && CURRENT_EV.section) ? CURRENT_EV.section : "";
  vhLoc.textContent = `${CURRENT} · p${pageNum}${sec ? " · " + sec : ""}`;

  stage.innerHTML = "";
  const wrap = el("div", "page-wrap");
  const img = document.createElement("img");
  img.src = url;
  img.alt = `${CURRENT} page ${pageNum}`;
  wrap.appendChild(img);

  // Capture bbox NOW so the async load callback doesn't depend on CURRENT_EV,
  // which may have been replaced or cleared by a subsequent click before load fires.
  const bboxSnapshot = (isSourcePage && CURRENT_EV && Array.isArray(CURRENT_EV.bbox)
                        && CURRENT_EV.bbox.length === 4) ? CURRENT_EV.bbox.slice() : null;
  if (bboxSnapshot && pageInfo) {
    img.addEventListener("load", () => {
      const renderedW = img.clientWidth || img.naturalWidth;
      const renderedH = img.clientHeight || img.naturalHeight;
      const ptW = pageInfo.width_pt || (img.naturalWidth / SCALE);
      const ptH = pageInfo.height_pt || (img.naturalHeight / SCALE);
      const sx = renderedW / ptW;
      const sy = renderedH / ptH;
      const [x0, y0, x1, y1] = bboxSnapshot;
      const overlay = el("div", "bbox-overlay");
      overlay.style.left   = (x0 * sx) + "px";
      overlay.style.top    = (y0 * sy) + "px";
      overlay.style.width  = ((x1 - x0) * sx) + "px";
      overlay.style.height = ((y1 - y0) * sy) + "px";
      wrap.appendChild(overlay);
      overlay.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  }
  stage.appendChild(wrap);
  updatePageNav();
}

function updatePageNav() {
  const prevBtn = document.getElementById("nav-prev");
  const nextBtn = document.getElementById("nav-next");
  const pageInput = document.getElementById("nav-page");
  const totalEl = document.getElementById("nav-total");
  const srcEl = document.getElementById("nav-src");
  const pmeta = CURRENT ? (DATA.manifest.papers || {})[CURRENT] : null;
  const total = pmeta ? (pmeta.page_count || (pmeta.pages || []).length) : 0;
  totalEl.textContent = total || "—";
  pageInput.value = CURRENT_PAGE || "";
  pageInput.max = total || 1;
  prevBtn.disabled = !CURRENT_PAGE || CURRENT_PAGE <= 1;
  nextBtn.disabled = !CURRENT_PAGE || CURRENT_PAGE >= total;
  if (CURRENT_EV && CURRENT_EV.page && CURRENT_EV.page !== CURRENT_PAGE) {
    srcEl.textContent = `(src p${CURRENT_EV.page})`;
    srcEl.style.cursor = "pointer";
  } else {
    srcEl.textContent = "";
    srcEl.style.cursor = "";
  }
}

function goPrev() { if (CURRENT_PAGE > 1) renderPage(CURRENT_PAGE - 1); }
function goNext() {
  const pmeta = (DATA.manifest.papers || {})[CURRENT];
  const total = pmeta ? (pmeta.page_count || (pmeta.pages || []).length) : 0;
  if (CURRENT_PAGE < total) renderPage(CURRENT_PAGE + 1);
}
function goToSource() { if (CURRENT_EV && CURRENT_EV.page) renderPage(CURRENT_EV.page); }

function loadPaper(rpId) {
  CURRENT = rpId;
  CURRENT_EV = null;
  CURRENT_PAGE = null;
  updatePageNav();
  document.querySelectorAll(".paper-row").forEach(r => r.classList.toggle("active", r.dataset.rp === rpId));
  const ext = DATA.extractions[rpId];
  const tree = document.getElementById("tree");
  tree.innerHTML = "";
  if (!ext) {
    tree.appendChild(el("div", "empty-stage", "No extraction found for " + rpId));
    return;
  }
  const titleVal = (ext.title && (ext.title.value || ext.title)) || "(no title)";
  const yearVal = (ext.year && (ext.year.value || ext.year)) || "—";
  const ptype = (ext.paper_type && ext.paper_type.value) || "—";
  const mfam = (ext.method_family && ext.method_family.value) || "—";
  tree.appendChild(el("h1", null, titleVal));
  tree.appendChild(el("div", "meta", `${rpId} · ${yearVal} · ${ptype} / ${mfam}`));
  Object.keys(ext).forEach(k => {
    if (k === "rp_id") return;
    tree.appendChild(renderField(k, ext[k], 0));
  });
  document.getElementById("vh-rp").textContent = rpId;
}

function init() {
  const list = document.getElementById("paper-list");
  const ids = Object.keys(DATA.extractions).sort();
  ids.forEach(rp => {
    const ext = DATA.extractions[rp];
    const title = (ext.title && (ext.title.value || ext.title)) || "(no title)";
    const year = (ext.year && (ext.year.value || ext.year)) || "—";
    const row = el("div", "paper-row");
    row.dataset.rp = rp;
    const rpEl = el("span", null, rp + " ");
    rpEl.style.color = "#60a5fa";
    rpEl.style.fontWeight = "600";
    row.appendChild(rpEl);
    row.appendChild(el("span", "y", `(${year})`));
    row.appendChild(document.createElement("br"));
    const tEl = el("span", null, title.slice(0, 60));
    tEl.style.color = "#cbd5e1";
    row.appendChild(tEl);
    row.addEventListener("click", () => loadPaper(rp));
    list.appendChild(row);
  });
  if (ids.length) loadPaper(ids[0]);

  // page-nav controls
  document.getElementById("nav-prev").addEventListener("click", goPrev);
  document.getElementById("nav-next").addEventListener("click", goNext);
  document.getElementById("nav-src").addEventListener("click", goToSource);
  const pageInput = document.getElementById("nav-page");
  pageInput.addEventListener("change", () => {
    const n = parseInt(pageInput.value, 10);
    if (!isNaN(n)) renderPage(n);
  });
  pageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); pageInput.blur(); }
  });

  // arrow-key shortcuts (when not typing in the page-input)
  document.addEventListener("keydown", (e) => {
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
    if (e.key === "ArrowLeft")  { goPrev(); e.preventDefault(); }
    if (e.key === "ArrowRight") { goNext(); e.preventDefault(); }
    if (e.key === "Home" && CURRENT_EV) { goToSource(); e.preventDefault(); }
  });
}

init();
</script>
</body>
</html>
""".replace("__DATA__", payload_js)


def main():
    if not MANIFEST.exists():
        print(f"ERROR: manifest not found at {MANIFEST} — run render_pages.py first.")
        return
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    extractions = load_extractions()
    if not extractions:
        print(f"ERROR: no extractions found in {EXTRACTIONS}")
        return
    html = build_html(extractions, manifest)
    OUT.write_text(html, encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"Wrote {OUT.name}: {size_kb:.1f} KB · {len(extractions)} papers · {sum(len(p.get('pages', [])) for p in manifest.get('papers', {}).values())} page images linked")
    print(f"Open: file:///{OUT.as_posix()}")


if __name__ == "__main__":
    main()
