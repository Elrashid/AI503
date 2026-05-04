"""
Render every PDF in papers_pdf/ to PNG-per-page in papers_pages/.

Output layout:
    papers_pages/
        RP01_Paperno_2016/
            page_01.png
            page_02.png
            ...
        ...
        manifest.json

The manifest stores DPI + scale_factor so the explorer can convert marker JSON
bboxes (in PDF point space, 72 DPI origin top-left) to pixel rectangles on the
rendered PNGs.

Usage:
    python render_pages.py
    python render_pages.py --dpi 150
    python render_pages.py --only RP09
"""

import argparse
import json
from pathlib import Path

import fitz  # PyMuPDF

HERE = Path(__file__).resolve().parent.parent  # project root (parent of scripts/)
PDF_DIR = HERE / "papers_pdf"
OUT_DIR = HERE / "papers_pages"


def render_pdf(pdf_path: Path, out_dir: Path, dpi: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pages = []
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        png_path = out_dir / f"page_{i:02d}.png"
        if not png_path.exists():
            pix.save(png_path)
        pages.append({
            "page": i,
            "file": png_path.name,
            "width_px": pix.width,
            "height_px": pix.height,
            "width_pt": page.rect.width,
            "height_pt": page.rect.height,
        })
    doc.close()
    return {"page_count": len(pages), "pages": pages}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--only", help="render only this RP id (e.g. RP09)")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    pdfs = sorted(PDF_DIR.glob("RP*.pdf"))
    if args.only:
        pdfs = [p for p in pdfs if p.stem.startswith(args.only + "_") or p.stem == args.only]
        if not pdfs:
            print(f"No PDFs matched {args.only}")
            return

    manifest = {
        "render_dpi": args.dpi,
        "scale_factor": args.dpi / 72.0,
        "papers": {},
    }

    for i, pdf in enumerate(pdfs, start=1):
        stem = pdf.stem
        rp_id = stem.split("_")[0]
        out = OUT_DIR / stem
        info = render_pdf(pdf, out, args.dpi)
        manifest["papers"][rp_id] = {
            "dir": stem,
            "pdf_file": pdf.name,
            **info,
        }
        print(f"[{i}/{len(pdfs)}] {stem} -> {info['page_count']} pages")

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {manifest_path}")
    print(f"Total papers: {len(manifest['papers'])}")
    print(f"Total pages: {sum(p['page_count'] for p in manifest['papers'].values())}")


if __name__ == "__main__":
    main()
