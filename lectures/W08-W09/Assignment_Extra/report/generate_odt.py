# -*- coding: utf-8 -*-
"""Generate the AI503 CNN report ODT (BUiD formatting) from draft/ or final/ section files.
Cover page, section headings, tables, embedded figures (caption above), references hanging indent,
page numbers. Run: python generate_odt.py
"""
import re, os
from PIL import Image as PILImage
from odf.opendocument import OpenDocumentText
from odf.style import (Style, TextProperties, ParagraphProperties, TableProperties,
                       TableColumnProperties, TableCellProperties, GraphicProperties,
                       PageLayout, PageLayoutProperties, FooterStyle, HeaderFooterProperties,
                       MasterPage, Footer)
from odf.text import P, H, Span, PageNumber, A
from odf.table import Table, TableColumn, TableRow, TableCell
from odf.draw import Frame, Image
from odf import teletype

BASE = "AI503_APR26_Machine_Learning/TeachingMaterial/W08-W09/Assignment_Extra/report"
COVER = {
    "university": "The British University in Dubai (BUiD)",
    "faculty": "Faculty of Engineering and Information Technology",
    "programme": "MSc in Artificial Intelligence",
    "module": "AI503: Machine Learning",
    "title": "Deep-Learning Image Classification\nUsing Convolutional Neural Networks",
    "assessment": "W08-W09 Extra Assignment",
    "student": "Mohamed Elrashid",
    "student_id": "22002576",
    "date": "June 2026",
}
RELATIVE_FILES = [
    "01_abstract/01_abstract.md", "02_introduction/01_introduction.md",
    "03_dataset/01_dataset.md", "04_methodology/01_methodology.md",
    "05_results/01_results.md", "06_discussion/01_discussion.md",
    "07_improvement/01_improvement.md", "08_limitations/01_limitations.md",
    "09_conclusion/01_conclusion.md", "10_references/01_references.md",
    "11_appendix_a/01_appendix_a.md", "12_appendix_b/01_appendix_b.md",
]
# Big/tall charts that get their own page (rendered large, on a fresh page).
FULLPAGE = {"fig_f1_bar", "fig_metrics_heatmap", "fig_group_curves", "fig_confusion_grid"}


def generate_odt(output_path, source_folder):
    FILES = [f"{BASE}/{source_folder}/{f}" for f in RELATIVE_FILES]
    doc = OpenDocumentText()

    def pstyle(name, **para):
        txt = {k[2:]: v for k, v in para.items() if k.startswith("t_")}
        par = {k: v for k, v in para.items() if not k.startswith("t_")}
        s = Style(name=name, family="paragraph")
        s.addElement(ParagraphProperties(**par))
        s.addElement(TextProperties(fontfamily="Times New Roman", **txt))
        doc.styles.addElement(s)
        return s

    DS = "0.635cm"  # double spacing
    cover_uni = pstyle("CoverUni", textalign="center", margintop="3cm", marginbottom="0.5cm", t_fontsize="14pt", t_fontweight="bold")
    cover_fac = pstyle("CoverFac", textalign="center", marginbottom="0.3cm", t_fontsize="12pt")
    cover_title = pstyle("CoverTitle", textalign="center", margintop="2cm", marginbottom="0.5cm", t_fontsize="20pt", t_fontweight="bold")
    cover_assess = pstyle("CoverAssess", textalign="center", marginbottom="1.5cm", t_fontsize="14pt")
    cover_info = pstyle("CoverInfo", textalign="center", marginbottom="0.2cm", t_fontsize="12pt")
    cover_date = pstyle("CoverDate", textalign="center", margintop="1.5cm", t_fontsize="12pt")

    sec_style = pstyle("Section", textalign="start", margintop="0.5cm", marginbottom="0.3cm",
                       linespacing=DS, breakbefore="page", keepwithnext="always",
                       t_fontsize="14pt", t_fontweight="bold")
    sub_style = pstyle("Subsection", textalign="start", margintop="0.3cm", marginbottom="0.15cm",
                       linespacing=DS, keepwithnext="always", t_fontsize="12pt", t_fontweight="bold")
    body_style = pstyle("Body", textalign="justify", textindent="1.27cm", linespacing=DS, t_fontsize="12pt")
    bullet_style = pstyle("Bullet", textalign="start", marginleft="1.27cm", textindent="-0.5cm", linespacing=DS, t_fontsize="12pt")
    ref_style = pstyle("Reference", textalign="start", marginleft="1.27cm", textindent="-1.27cm", linespacing=DS, t_fontsize="12pt")
    fig_num_style = pstyle("FigNum", textalign="center", margintop="0.2cm", linespacing=DS, keepwithnext="always", t_fontsize="11pt", t_fontweight="bold")
    img_p_style = pstyle("ImgP", textalign="center", margintop="0.2cm", linespacing=DS)
    pagebreak_style = pstyle("PageBreak", breakbefore="page")

    bold_style = Style(name="Bold", family="text"); bold_style.addElement(TextProperties(fontweight="bold")); doc.styles.addElement(bold_style)
    italic_style = Style(name="Italic", family="text"); italic_style.addElement(TextProperties(fontstyle="italic")); doc.styles.addElement(italic_style)

    img_frame_style = Style(name="ImgFrame", family="graphic")
    img_frame_style.addElement(GraphicProperties(verticalpos="top", verticalrel="paragraph", horizontalpos="center", horizontalrel="paragraph"))
    doc.automaticstyles.addElement(img_frame_style)

    page_dims = dict(pagewidth="21cm", pageheight="29.7cm", margintop="3cm", marginbottom="2cm", marginleft="2.5cm", marginright="3cm")
    pl_body = PageLayout(name="PL_Body"); pl_body.addElement(PageLayoutProperties(**page_dims))
    fs = FooterStyle(); fs.addElement(HeaderFooterProperties(minheight="0.5cm", margintop="0.3cm")); pl_body.addElement(fs)
    doc.automaticstyles.addElement(pl_body)
    pn_style = pstyle("PageNum", textalign="center", t_fontsize="11pt")
    mp = MasterPage(name="Standard", pagelayoutname="PL_Body")
    footer = Footer(); fp = P(stylename=pn_style); fp.addElement(PageNumber(selectpage="current")); footer.addElement(fp); mp.addElement(footer)
    doc.masterstyles.addElement(mp)

    URL_RE = re.compile(r'(https?://[^\s)]+)')
    def addtext(parent, s):
        # add text, turning bare URLs into clickable hyperlinks
        for seg in URL_RE.split(s):
            if seg.startswith('http'):
                a = A(href=seg); teletype.addTextToElement(a, seg); parent.addElement(a)
            elif seg:
                teletype.addTextToElement(parent, seg)

    def render_inline(p, text):
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)   # strip md links
        for part in re.split(r"(\*\*.*?\*\*)", text):
            if part.startswith("**") and part.endswith("**"):
                sp = Span(stylename=bold_style); teletype.addTextToElement(sp, part[2:-2]); p.addElement(sp)
            elif "*" in part:
                for ip in re.split(r"(\*[^*]+\*)", part):
                    if ip.startswith("*") and ip.endswith("*") and not ip.startswith("**"):
                        sp = Span(stylename=italic_style); teletype.addTextToElement(sp, ip[1:-1]); p.addElement(sp)
                    else:
                        addtext(p, ip)
            else:
                addtext(p, part)

    # COVER
    p = P(stylename=cover_uni); teletype.addTextToElement(p, COVER["university"]); doc.text.addElement(p)
    for k in ["faculty", "programme", "module"]:
        p = P(stylename=cover_fac); teletype.addTextToElement(p, COVER[k]); doc.text.addElement(p)
    for line in COVER["title"].split("\n"):
        p = P(stylename=cover_title); teletype.addTextToElement(p, line); doc.text.addElement(p)
    p = P(stylename=cover_assess); teletype.addTextToElement(p, COVER["assessment"]); doc.text.addElement(p)
    for label, k in [("Student:", "student"), ("Student ID:", "student_id")]:
        p = P(stylename=cover_info); teletype.addTextToElement(p, f"{label} {COVER[k]}"); doc.text.addElement(p)
    p = P(stylename=cover_date); teletype.addTextToElement(p, COVER["date"]); doc.text.addElement(p)

    tbl_count = [0]

    def emit_table(table_lines):
        rows = []
        for tl in table_lines:
            if re.match(r"^\|[\s:|-]+\|$", tl):
                continue
            rows.append([c.strip() for c in tl.split("|")[1:-1]])
        if not rows:
            return
        tbl_count[0] += 1
        tname = f"T{tbl_count[0]}"
        ts = Style(name=tname, family="table"); ts.addElement(TableProperties(width="16cm", align="center"))
        doc.automaticstyles.addElement(ts)
        ncols = len(rows[0])
        hcs = Style(name=f"{tname}H", family="table-cell")
        hcs.addElement(TableCellProperties(padding="0.08cm", bordertop="0.04cm solid #000000", borderbottom="0.04cm solid #000000", backgroundcolor="#DAEEF3"))
        doc.automaticstyles.addElement(hcs)
        bcs = Style(name=f"{tname}B", family="table-cell")
        bcs.addElement(TableCellProperties(padding="0.08cm", borderbottom="0.02cm solid #CCCCCC"))
        doc.automaticstyles.addElement(bcs)
        hts = Style(name=f"{tname}HT", family="paragraph"); hts.addElement(ParagraphProperties(textalign="center")); hts.addElement(TextProperties(fontsize="10pt", fontweight="bold", fontfamily="Times New Roman")); doc.automaticstyles.addElement(hts)
        bts = Style(name=f"{tname}BT", family="paragraph"); bts.addElement(TextProperties(fontsize="10pt", fontfamily="Times New Roman")); doc.automaticstyles.addElement(bts)
        table = Table(name=tname, stylename=ts)
        for ci in range(ncols):
            cs = Style(name=f"{tname}C{ci}", family="table-column"); cs.addElement(TableColumnProperties(columnwidth=f"{16.0/ncols:.2f}cm")); doc.automaticstyles.addElement(cs)
            table.addElement(TableColumn(stylename=cs))
        for ridx, cells in enumerate(rows):
            tr = TableRow(); hdr = (ridx == 0)
            while len(cells) < ncols:
                cells.append("")
            for ct in cells:
                tc = TableCell(stylename=hcs if hdr else bcs)
                cp = P(stylename=hts if hdr else bts); render_inline(cp, ct); tc.addElement(cp); tr.addElement(tc)
            table.addElement(tr)
        doc.text.addElement(table)

    for fname in FILES:
        text = re.sub(r"<!--.*?-->", "", open(fname, encoding="utf-8").read(), flags=re.DOTALL).strip()
        is_ref = "10_references" in fname
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            if not line:
                i += 1; continue

            # image
            if line.startswith("!["):
                m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
                if m:
                    img_abs = os.path.normpath(os.path.join(os.path.dirname(fname), m.group(2)))
                    base = os.path.splitext(os.path.basename(img_abs))[0]
                    full = base in FULLPAGE
                    if full:
                        doc.text.addElement(P(stylename=pagebreak_style))   # start big figures on a fresh page
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines) and lines[j].strip().startswith(">"):
                        cap = re.match(r">\s*\*\*(Figure [A-Z]?[\d.]+):?\*\*\s*(.*)", lines[j].strip())
                        if cap:
                            p1 = P(stylename=fig_num_style)
                            sp = Span(stylename=bold_style); teletype.addTextToElement(sp, cap.group(1)); p1.addElement(sp)
                            teletype.addTextToElement(p1, ": ")
                            sp2 = Span(stylename=italic_style); teletype.addTextToElement(sp2, cap.group(2)); p1.addElement(sp2)
                            doc.text.addElement(p1)
                        i = j + 1
                    else:
                        i += 1
                    if os.path.exists(img_abs):
                        href = doc.addPicture(img_abs)
                        try:
                            with PILImage.open(img_abs) as im:
                                pw, ph = im.size
                            aspect = pw / ph
                        except Exception:
                            aspect = 1.5
                        appendix = os.path.basename(os.path.dirname(img_abs)) == 'appendix'
                        max_w, max_h = 16.0, (21.0 if full else (18.0 if appendix else 11.0))   # appendix figs fill the width
                        w_cm = max_w; h_cm = w_cm / aspect
                        if h_cm > max_h:
                            h_cm = max_h; w_cm = h_cm * aspect
                        ip = P(stylename=img_p_style)
                        fr = Frame(stylename=img_frame_style, width=f"{w_cm:.2f}cm", height=f"{h_cm:.2f}cm", anchortype="as-char")
                        fr.addElement(Image(href=href)); ip.addElement(fr); doc.text.addElement(ip)
                        if full:
                            doc.text.addElement(P(stylename=pagebreak_style))
                    continue
                i += 1; continue

            # standalone caption (already consumed normally; skip stray)
            if line.startswith(">"):
                i += 1; continue

            # table caption
            if line.startswith("**Table "):
                m = re.match(r"\*\*(Table [\d.]+):?\s*(.*?)\*\*", line)
                p1 = P(stylename=fig_num_style)
                if m:
                    sp = Span(stylename=bold_style); teletype.addTextToElement(sp, m.group(1)); p1.addElement(sp)
                    teletype.addTextToElement(p1, ": ")
                    sp2 = Span(stylename=italic_style); teletype.addTextToElement(sp2, m.group(2).rstrip('*')); p1.addElement(sp2)
                else:
                    render_inline(p1, line)
                doc.text.addElement(p1); i += 1; continue

            # table
            if line.startswith("|"):
                tl = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    tl.append(lines[i].rstrip()); i += 1
                emit_table(tl); continue

            # bullet
            if line.lstrip().startswith("- "):
                p = P(stylename=bullet_style)
                teletype.addTextToElement(p, "•  ")
                render_inline(p, line.lstrip()[2:])
                doc.text.addElement(p); i += 1; continue

            # heading
            hm = re.match(r"^(#{1,6})\s+(.*)", line)
            if hm:
                level = len(hm.group(1))
                style = sec_style if level <= 2 else sub_style
                h = H(outlinelevel=min(level, 2), stylename=style)
                teletype.addTextToElement(h, hm.group(2).strip())
                doc.text.addElement(h); i += 1; continue

            # paragraph
            para = []
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||>|!\[|\*\*Table |- )", lines[i].lstrip()):
                para.append(lines[i].rstrip()); i += 1
            if para:
                p = P(stylename=ref_style if is_ref else body_style)
                render_inline(p, " ".join(para))
                doc.text.addElement(p)
            else:
                i += 1

    doc.save(output_path)
    print("Created:", output_path)


if __name__ == "__main__":
    generate_odt(f"{BASE}/draft/A2_AI503_CNN_Report_Draft.odt", "draft")
    generate_odt(f"{BASE}/final/A2_AI503_CNN_Report.odt", "final")
