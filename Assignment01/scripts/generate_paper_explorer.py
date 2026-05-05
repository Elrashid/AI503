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
  html, body { -webkit-text-size-adjust: 100%; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #0f172a; color: #e2e8f0;
         /* viewport-aware height for mobile browsers (handles dynamic chrome) */
         min-height: 100vh; min-height: 100dvh; }
  #app { display: grid; grid-template-columns: 240px 480px 1fr; height: 100vh; height: 100dvh;
         overflow: hidden; }
  .panel { overflow: auto; border-right: 1px solid #334155; -webkit-overflow-scrolling: touch; }

  /* === Mobile/tablet tab bar — hidden on desktop === */
  #tabs {
    display: none;
    background: #0b1220; border-bottom: 1px solid #334155;
    padding: 4px 4px 0;
  }
  #tabs button {
    flex: 1; padding: 12px 4px; border: none; background: transparent;
    color: #94a3b8; font-size: 13px; font-weight: 600; cursor: pointer;
    border-bottom: 3px solid transparent; min-height: 44px;
  }
  #tabs button.active { color: #f1f5f9; border-bottom-color: #60a5fa; }
  #tabs button:active { background: #1e293b; }

  /* === Tablet: viewer overlays as a slide-out === */
  @media (max-width: 1100px) {
    #app { grid-template-columns: 220px 1fr; }
    #viewer { display: none; position: fixed; top: 0; right: 0; bottom: 0;
              width: min(720px, 100vw); z-index: 30;
              box-shadow: -4px 0 20px rgba(0,0,0,0.6); }
    #app.show-viewer #viewer { display: block; }
    #app.show-viewer::before {
      content: ''; position: fixed; inset: 0; background: rgba(0,0,0,0.4);
      z-index: 25; pointer-events: auto;
    }
    #close-viewer { display: inline-block; }
  }

  /* === Phone: single column, tab-switched === */
  @media (max-width: 700px) {
    body { display: flex; flex-direction: column; height: 100vh; height: 100dvh; }
    #tabs { display: flex; flex-shrink: 0; }
    #app { grid-template-columns: 1fr; flex: 1; min-height: 0; height: auto; }
    #papers, #tree, #viewer { display: none; border-right: none; }
    #app.tab-papers #papers { display: block; }
    #app.tab-tree #tree     { display: block; }
    #app.tab-viewer #viewer { display: block; position: static; width: 100%;
                              box-shadow: none; }
    #app.tab-viewer::before { display: none; }
    #viewer-stage { height: calc(100vh - 130px) !important; padding: 8px !important; }
    #viewer-header { flex-wrap: wrap; height: auto; padding: 8px; gap: 6px; }
    #viewer-header .quote { width: 100%; flex-basis: 100%; white-space: normal;
                            font-size: 11px; line-height: 1.35; max-height: 3em;
                            overflow: hidden; }
    .page-nav button { padding: 8px 14px; min-height: 40px; }
    .page-nav .counter { font-size: 13px; }
    .page-nav .counter input { font-size: 13px; width: 44px; padding: 4px; min-height: 32px; }
    #tree { padding: 12px 14px 60px; }
    #tree h1 { font-size: 16px; }
    .field { padding: 6px 8px; font-size: 13.5px; }
    .paper-row { padding: 12px 14px; font-size: 14px; min-height: 44px;
                 display: flex; flex-direction: column; justify-content: center; }
  }

  #close-viewer { display: none; appearance: none; border: 1px solid #334155;
                  background: #1e293b; color: #fbbf24; width: 36px; height: 36px;
                  border-radius: 4px; cursor: pointer; font-size: 16px; margin-left: auto; }
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
<div id="tabs">
  <button data-tab="papers" class="active">📚 Papers</button>
  <button data-tab="tree">🧾 Extraction</button>
  <button data-tab="viewer">📄 Source page</button>
</div>
<div id="app" class="tab-papers">
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
      <button id="close-viewer" title="Close source view" aria-label="Close">×</button>
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

function renderField(name, node, depth) {
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

  if (isSourcePage && Array.isArray(CURRENT_EV.bbox) && CURRENT_EV.bbox.length === 4 && pageInfo) {
    img.addEventListener("load", () => {
      const renderedW = img.clientWidth || img.naturalWidth;
      const renderedH = img.clientHeight || img.naturalHeight;
      const ptW = pageInfo.width_pt || (img.naturalWidth / SCALE);
      const ptH = pageInfo.height_pt || (img.naturalHeight / SCALE);
      const sx = renderedW / ptW;
      const sy = renderedH / ptH;
      const [x0, y0, x1, y1] = CURRENT_EV.bbox;
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
    row.addEventListener("click", () => { loadPaper(rp); setTab("tree"); });
    list.appendChild(row);
  });
  if (ids.length) loadPaper(ids[0]);

  // === Responsive tab + slide-out wiring ===
  const app = document.getElementById("app");
  window.setTab = function(name) {
    // Phone (tab-switched layout)
    app.classList.remove("tab-papers", "tab-tree", "tab-viewer");
    app.classList.add("tab-" + name);
    document.querySelectorAll("#tabs button").forEach(b =>
      b.classList.toggle("active", b.dataset.tab === name));
    // Tablet (viewer = slide-out)
    if (name === "viewer") app.classList.add("show-viewer");
    else app.classList.remove("show-viewer");
  };
  document.querySelectorAll("#tabs button").forEach(b =>
    b.addEventListener("click", () => setTab(b.dataset.tab)));
  document.getElementById("close-viewer").addEventListener("click",
    () => app.classList.remove("show-viewer"));
  // Delegated click on tree fields — when a field with ev.page is clicked,
  // switch to the viewer panel (mobile) / slide it in (tablet).
  document.getElementById("tree").addEventListener("click", (e) => {
    const f = e.target.closest(".field");
    if (!f) return;
    // Only switch if this field has a page reference shown in the viewer
    if (CURRENT_EV && CURRENT_EV.page) setTab("viewer");
  });

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
