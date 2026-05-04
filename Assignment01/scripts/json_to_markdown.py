"""
Convert marker-pdf JSON output to clean Markdown — one .md per paper.

Reads:
  papers_json/<paper_id>/<paper_id>.json

Writes:
  papers_md/<paper_id>.md

Usage:
  python scripts/json_to_markdown.py
  python scripts/json_to_markdown.py --only RP09
"""

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
JSON_DIR = HERE / "papers_json"
MD_DIR = HERE / "papers_md"


def html_to_md(html):
    """Convert marker's HTML fragment to Markdown text."""
    if not html:
        return ""
    t = html
    # Inline formatting
    t = re.sub(r"<b>(.*?)</b>", r"**\1**", t, flags=re.S)
    t = re.sub(r"<strong>(.*?)</strong>", r"**\1**", t, flags=re.S)
    t = re.sub(r"<i>(.*?)</i>", r"*\1*", t, flags=re.S)
    t = re.sub(r"<em>(.*?)</em>", r"*\1*", t, flags=re.S)
    t = re.sub(r"<sup>(.*?)</sup>", r"^\1^", t, flags=re.S)
    t = re.sub(r"<sub>(.*?)</sub>", r"~\1~", t, flags=re.S)
    t = re.sub(r"<a[^>]*href=\"([^\"]*)\"[^>]*>(.*?)</a>", r"[\2](\1)", t, flags=re.S)
    # Marker's content-ref placeholders
    t = re.sub(r"<content-ref[^>]*></content-ref>", "", t)
    t = re.sub(r"<content-ref[^>]*/>", "", t)
    # Line breaks
    t = re.sub(r"<br\s*/?>", "\n", t)
    # Strip remaining tags
    t = re.sub(r"<[^>]+>", "", t)
    # Decode common entities
    t = (t.replace("&amp;", "&")
         .replace("&lt;", "<")
         .replace("&gt;", ">")
         .replace("&quot;", '"')
         .replace("&apos;", "'")
         .replace("&nbsp;", " "))
    # Normalize whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def html_table_to_md(html):
    """Convert an <table> HTML to a Markdown table."""
    if not html or "<table" not in html.lower():
        return html_to_md(html)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S | re.I)
    if not rows:
        return html_to_md(html)

    table_rows = []
    for row in rows:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, flags=re.S | re.I)
        cells = [html_to_md(c).replace("|", "\\|").replace("\n", " ") for c in cells]
        table_rows.append(cells)

    if not table_rows:
        return html_to_md(html)

    # Pad each row to max cols
    max_cols = max(len(r) for r in table_rows)
    table_rows = [r + [""] * (max_cols - len(r)) for r in table_rows]

    # First row = header (if it has any tH-style content); else create fake header
    md_rows = ["| " + " | ".join(table_rows[0]) + " |"]
    md_rows.append("| " + " | ".join(["---"] * max_cols) + " |")
    for r in table_rows[1:]:
        md_rows.append("| " + " | ".join(r) + " |")
    return "\n".join(md_rows)


def heading_for(block, page_num, default_level=2):
    """Return Markdown heading line for SectionHeader block."""
    text = html_to_md(block.get("html", "") or "")
    if not text:
        return ""
    # Marker may set heading_level on metadata (rare); fall back to depth heuristic
    level = block.get("heading_level")
    if level is None:
        # Infer from numbering: 1 INTRO -> 1, 2.1 SUB -> 2, 2.1.1 -> 3
        m = re.match(r"^(\d+(?:\.\d+){0,4})\s", text)
        if m:
            level = m.group(1).count(".") + 1
        else:
            level = default_level
    level = max(1, min(level, 6))
    return "#" * level + " " + text


def render_block(block, paper_id, fig_counter=None):
    """Render one marker block into a Markdown string (or '' to skip).
    `fig_counter` is a mutable single-element list [n] used to track sequential
    figure numbering matching json_to_claude_xml.py's naming scheme:
    `<paper_id>_fig01.jpg`, `<paper_id>_fig02.jpg`, ...
    """
    if fig_counter is None:
        fig_counter = [0]
    bt = block.get("block_type", "")
    html = block.get("html", "") or ""

    if bt in ("PageHeader", "PageFooter"):
        return ""  # Skip noise (and known marker bug — empty html)

    if bt == "SectionHeader":
        return heading_for(block, page_num=None) + "\n"

    if bt == "Text":
        text = html_to_md(html)
        return text + "\n" if text else ""

    if bt in ("Table", "TableGroup"):
        md = html_table_to_md(html)
        return md + "\n" if md else ""

    if bt == "Equation":
        text = html_to_md(html)
        if not text:
            return ""
        if "\n" in text or len(text) > 80:
            return f"$$\n{text}\n$$\n"
        return f"$$ {text} $$\n"

    if bt == "Code":
        text = html_to_md(html)
        return f"```\n{text}\n```\n" if text else ""

    if bt in ("ListGroup", "OrderedListGroup"):
        items = []
        ordered = bt.startswith("Ordered")
        for i, child in enumerate(block.get("children", []) or [], start=1):
            if child.get("block_type") in ("ListItem", "OrderedListItem"):
                t = html_to_md(child.get("html", ""))
                if not t:
                    continue
                marker = f"{i}. " if ordered else "- "
                items.append(marker + t)
            else:
                # Recurse into other child types
                sub = render_block(child, paper_id, fig_counter)
                if sub.strip():
                    items.append(sub.rstrip())
        return "\n".join(items) + "\n" if items else ""

    if bt in ("ListItem", "OrderedListItem"):
        t = html_to_md(html)
        return f"- {t}\n" if t else ""

    if bt == "Caption":
        text = html_to_md(html)
        return f"*{text}*\n" if text else ""

    if bt == "FigureGroup":
        # Wrapper only — recurse into children (one of which is the actual Figure
        # carrying the image). DO NOT increment counter here, let the inner
        # Figure/Picture do it.
        parts = []
        for child in block.get("children", []) or []:
            child_md = render_block(child, paper_id, fig_counter)
            if child_md:
                parts.append(child_md.strip())
        return "\n".join(parts) + "\n" if parts else ""

    if bt in ("Figure", "Picture"):
        # Sequential numbering matches json_to_claude_xml.py figure-naming scheme:
        # <paper_id>_fig01.jpg, <paper_id>_fig02.jpg, ...  Only count when the
        # block itself carries at least one image.
        parts = []
        if block.get("images"):
            fig_counter[0] += 1
            parts.append(
                f"![{paper_id} fig{fig_counter[0]:02d}]"
                f"(../figures/{paper_id}_fig{fig_counter[0]:02d}.jpg)"
            )
        # Walk children for any caption / nested content (no further image counting
        # because Figure shouldn't contain another Figure normally).
        for child in block.get("children", []) or []:
            if child.get("block_type") in ("Figure", "Picture"):
                continue  # avoid double-count if marker ever nests
            child_md = render_block(child, paper_id, fig_counter)
            if child_md:
                parts.append(child_md.strip())
        return "\n".join(parts) + "\n" if parts else ""

    if bt == "Footnote":
        text = html_to_md(html)
        return f"> {text}\n" if text else ""

    if bt == "Reference":
        text = html_to_md(html)
        return text + "\n" if text else ""

    if bt == "Document":
        # Walked at the top level — nothing to render directly
        return ""

    if bt == "Page":
        return ""  # Walked separately

    # Unknown — fall back to text
    text = html_to_md(html)
    return text + "\n" if text else ""


def convert_paper(json_path: Path, md_dir: Path):
    """Convert one paper JSON to Markdown."""
    paper_id = json_path.stem  # RP09_Frantar_2022
    out_md = md_dir / f"{paper_id}.md"
    md_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    pages = data.get("children") or []
    # Top of doc is Document; walk into pages
    if data.get("block_type") == "Document":
        pages = data.get("children", []) or []

    parts = []
    # Try to get title from metadata or first SectionHeader
    metadata = data.get("metadata") or {}
    toc = metadata.get("table_of_contents") or []
    title = ""
    if toc:
        title = toc[0].get("title", "") if isinstance(toc[0], dict) else ""
    if not title and pages:
        for child in (pages[0].get("children") or []):
            if child.get("block_type") == "SectionHeader":
                title = html_to_md(child.get("html", ""))
                break
    if title:
        parts.append(f"<!-- {paper_id} | source: papers_json/{paper_id}/ -->\n")

    # Sequential figure counter shared across the whole document
    fig_counter = [0]
    # Render each page
    for i, page in enumerate(pages, start=1):
        if page.get("block_type") != "Page":
            continue
        # Optional page separator (subtle)
        if i > 1:
            parts.append(f"\n<!-- page {i} -->\n")
        for child in page.get("children", []) or []:
            md = render_block(child, paper_id, fig_counter)
            if md and md.strip():
                parts.append(md)

    out_text = "\n".join(parts).strip() + "\n"
    # Collapse triple-blank lines
    out_text = re.sub(r"\n{3,}", "\n\n", out_text)
    out_md.write_text(out_text, encoding="utf-8")
    return out_md, len(out_text), out_text.count("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="convert only this RP id (e.g. RP09)")
    args = ap.parse_args()

    json_dirs = sorted(d for d in JSON_DIR.iterdir() if d.is_dir())
    if args.only:
        json_dirs = [d for d in json_dirs if d.name.startswith(args.only)]

    total_bytes = 0
    print(f"{'Paper':<30}{'bytes':>10}{'lines':>8}")
    print("-" * 50)
    for d in json_dirs:
        json_path = d / f"{d.name}.json"
        if not json_path.exists():
            continue
        out, n_bytes, n_lines = convert_paper(json_path, MD_DIR)
        total_bytes += n_bytes
        print(f"{d.name:<30}{n_bytes:>10,}{n_lines:>8}")

    print("-" * 50)
    print(f"Wrote {len(json_dirs)} papers, {total_bytes:,} bytes total to {MD_DIR.relative_to(HERE)}/")


if __name__ == "__main__":
    main()
