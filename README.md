# AI503 — Machine Learning (Assignment 01) · Quantization-Safety SLR

A systematic literature review of **post-training quantization for transformer LLMs with a safety-evaluation lens**, conducted by Mohamed Elrashid for AI503 at the British University in Dubai.

50 papers · marker-pdf full-text extraction · source-traced JSON extractions · interactive paper explorer · comparative analysis table.

> **Live demo (paper explorer):** open `Assignment01/paper_explorer.html` after cloning. Click any paper on the left, then any extracted field in the middle pane to see the source page on the right with the bbox highlighted.

## What's in here

| Path | What it is |
|---|---|
| `Assignment01/Assignment_1.md` | Assignment brief from the module coordinator |
| `Assignment01/exports/` | Deliverable #1 — ResearchRabbit CSV exports (50 papers, bibliographic) |
| `Assignment01/docs/` | Methodology docs (off-topic audit, replacement candidates, move cheatsheet) |
| `Assignment01/papers_pdf/` | The 50 source PDFs (arXiv preprints + published versions) |
| `Assignment01/papers_json/` | Marker-pdf JSON parses of each PDF (LLM-assisted via Gemini 2.5 Flash) |
| `Assignment01/papers_xml/` | Claude-optimized XML derived from the JSONs (per-page, per-section, per-block) |
| `Assignment01/papers_pages/` | 150 DPI PNG renders of every paper page (1,371 pages) + bbox manifest |
| `Assignment01/figures/` | 520 figure images extracted from the papers |
| `Assignment01/extractions/` | 50 structured extraction JSONs — every leaf has `{value, ev: {page, section, quote, bbox, block_id}}` |
| `Assignment01/comparative_analysis_table.csv` | Deliverable #3 source — 50 rows, the assignment's required comparative table |
| `Assignment01/dashboard.html` | Birds-eye dashboard: stats, charts, sortable per-paper table |
| `Assignment01/paper_explorer.html` | **Interactive paper explorer** with extraction tree + page-image bbox overlay |
| `Assignment01/scripts/` | Build pipeline (Python) — comparative table, dashboard, paper explorer, audit, downloads |
| `Assignment01/.slr/` | Schema, system prompt, and config used for extraction (Claude agent inputs) |
| `Assignment02/experiments/` | **Assignment 2 experiments** — Exp 1: quantization × safety diagnosis (Qwen2.5-1.5B at FP16/INT8/NF4; Colab notebook + per-prompt generations + results). Exp 2: SAQC selective-escalation cascade (scripts, CSV results, figures; no GPU needed). Each experiment folder has its own README. |
| `notebooks/marker_convert_v5.ipynb` | Colab notebook — PDF → marker JSON conversion (Gemini 2.5 Flash backend) |
| `lectures/` | **Course lecture slides + lab notebooks (Weeks 1–9)** — see `lectures/index.html`. Logistic regression, KNN/DT/SVM/NB, clustering, deep learning, LSTM, ensembles. |
| `lectures/W08-W09/Assignment_Extra/` | **CNN Image Classification — Extra Assignment**: a 26-model CNN bake-off on CIFAR-10 (notebook + 43-page report, champion 0.94 macro-F1) |

## Quick start

```bash
# 1. Clone with LFS — heavy binaries (PDFs, page PNGs, figures) are stored in git-lfs
git lfs install
git clone https://github.com/<you>/<this-repo>.git
cd <this-repo>

# 2. Open the explorer in any browser
start Assignment01/paper_explorer.html   # Windows
open  Assignment01/paper_explorer.html   # macOS
xdg-open Assignment01/paper_explorer.html  # Linux

# 3. Or open the dashboard
start Assignment01/dashboard.html
```

The HTML files are self-contained — no web server needed. All page-image / figure links are relative paths inside the repo.

## Reproducing the pipeline

The 50 papers were curated through ResearchRabbit. To re-run the conversion + extraction from scratch:

```bash
# 1. Convert PDFs to marker JSON (Colab + GPU, takes hours for the full 50)
#    Open notebooks/marker_convert_v5.ipynb, paste your Gemini API key,
#    upload PDFs to Drive, run all cells.

# 2. Convert JSON to source-traced XML
python ../slr-extraction/scripts/json_to_claude_xml.py \
    Assignment01/papers_json/ \
    --xml-dir Assignment01/papers_xml/ \
    --figures-dir Assignment01/figures/

# 3. Render 150 DPI page PNGs (needed for the paper explorer)
python Assignment01/scripts/render_pages.py

# 4. Extract page-furniture text marker-pdf strips (PageHeader / PageFooter)
python Assignment01/scripts/extract_furniture.py

# 5. Extract structured data (Claude agent — see Assignment01/.slr/system_prompt.txt)
#    Each paper produces extractions/RPxx_extraction.json with the {value, ev} schema.

# 6. Backfill bboxes on extractions
python Assignment01/scripts/backfill_bboxes.py

# 7. Build the comparative table, dashboard, and explorer
python Assignment01/scripts/generate_comparative_table.py
python Assignment01/scripts/generate_v3_dashboard.py
python Assignment01/scripts/generate_paper_explorer.py

# 8. Audit consistency
python Assignment01/scripts/audit_v3.py
```

## Extraction schema

The 50 extraction JSONs all share a uniform shape: every analytical leaf is wrapped in `{value, ev}` for source tracing.

```jsonc
{
  "rp_id": "RP09",
  "year": {
    "value": 2022,
    "ev": {
      "page": 1, "section": "header",
      "quote": "Frantar et al., 2022",
      "bbox": [123, 456, 789, 480], "block_id": 17
    }
  },
  "key_results": [
    {
      "value": "GPTQ on OPT-175B preserves perplexity within 0.1 at 4-bit",
      "ev": { "page": 5, "section": "Results", "quote": "...", "bbox": [...], "block_id": 89 }
    },
    ...
  ],
  ...
}
```

Total: **3,894 leaves across 50 papers, 88.5% traced** (page+section recorded), 88.4% with bbox (clickable in the explorer).

## Repository policy

- **Code** (`scripts/`, `notebooks/`, JS in HTML): MIT
- **Data / docs / extractions / CSVs / README**: CC BY 4.0
- **PDFs / page renders / marker outputs / figures**: third-party; retained under fair-use research exemption. See [LICENSE](LICENSE) for full terms.

If you are a copyright holder and want material removed, please file an issue.

## Acknowledgements

- Module: AI503 Machine Learning, BUiD, Spring 2026
- Coordinator: Dr. Manar Alkhatib
- Supervisor (thesis): Prof. Khaled Shaalan
- Tools: [ResearchRabbit](https://app.researchrabbit.ai), [marker-pdf](https://github.com/datalab-to/marker), Google Gemini 2.5 Flash, Anthropic Claude Code
