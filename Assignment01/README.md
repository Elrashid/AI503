# Assignment 01 — v3 (clean 50-paper corpus)

`v3/` is the working directory for the AI503 Machine Learning A1 deliverable: a 5,000-word systematic literature review of post-training quantization for transformer LLMs, with a comparative table of 50 papers.

## Why v3 exists

| Version | Purpose | Status |
|---|---|---|
| `v1/` | Original 50-paper attempt | frozen, do not edit |
| `v2/` | Reverse-engineered RR API + extracted all 50 + flagged 19 problematic (5 off-topic, 2 wrong PDF, 3 dup, 4 surveys, 7 tangential) | frozen, audit reference |
| `v3/` | **Clean 50-paper corpus**: 32 kept from v2 + 18 added via RR Similar Work / Recently Found | **active** |

The v3 corpus matches the **AI503 A1 - Quantization Safety SLR** parent collection in ResearchRabbit (https://app.researchrabbit.ai/library/collection/df7b76c5-78b1-4c5f-bcbf-9ffba871ca31).

## Corpus state (snapshot 2026-05-04)

| Folder in RR | Papers | In v3? |
|---|---:|---|
| AI503 A1 - Quantization Safety SLR (parent) | **50** | yes — 32 fully extracted, 18 awaiting marker → XML → extract |
| AI503 A1 - QS - duplicate | 3 | no |
| AI503 A1 - QS - wrong PDF | 2 | no |
| AI503 A1 - QS - off topic | 3 | no |
| AI503 A1 - QS - tangential | 7 | no |
| AI503 A1 - QS - survey | 5 | no |

## Layout

```
v3/
├── README.md                              ← this file
├── comparative_analysis_table.csv         ← 32 rows (regenerates from extractions/)
├── paper_explorer.html                    ← interactive viewer for the 32 extracted papers
│
├── exports/                               ← Deliverable #1 (RR Export)
│   ├── README.md                          ← (missing DOI) gotcha + provenance policy
│   ├── AI503_A1_RR_export_50papers_official.csv     ← submit this
│   └── AI503_A1_RR_export_50papers_extended.csv     ← supplementary (citation counts, OA flag)
│
├── docs/
│   ├── off_topic_papers.md                ← 4-tier audit of v2's 50
│   ├── replacement_candidates.md          ← rationale for the 18 new picks
│   └── researchrabbit_move_cheatsheet.md  ← query → folder map used during moves
│
├── scripts/                               ← project-specific (skill-coupled scripts went to slr-extraction)
│   ├── generate_comparative_table.py      ← reads extractions/ → comparative_analysis_table.csv
│   ├── verify_extractions.py              ← schema check + trace stats
│   ├── patch_year_venue_from_furniture.py ← fills year/venue ev from PageFooter text
│   └── rr_moves.csv                       ← move-plan history
│
├── papers_pdf/                            ← 50 PDFs (32 from v2 + 18 downloaded from arXiv via scripts/download_new_papers.py)
├── papers_json/                           ← 50 marker JSONs (32 from v2 + 18 from Colab marker run)
├── papers_xml/                            ← 50 source-traced XMLs
├── papers_pages/                          ← 50 dirs of rendered page PNGs (1,371 pages @ 150 DPI) + furniture.json sidecars + manifest.json
├── extractions/                           ← 32 JSONs (88.4% leaves traced, all schema-valid; 18 pending extractor agent run)
├── figures/                               ← 520 figure assets (316 from v2 + 204 from new XML conversion)
└── verifications/                         ← empty until verifier agent runs
```

## Reusable utilities (live in skills, not here)

| Tool | Skill |
|---|---|
| `render_pages.py` (PDF→PNG @ DPI) | `slr-extraction/scripts/` |
| `extract_furniture.py` (PageHeader/PageFooter recovery) | `slr-extraction/scripts/` |
| `backfill_bboxes.py` (bbox post-processing) | `slr-extraction/scripts/` |
| `generate_paper_explorer.py` (interactive HTML viewer) | `slr-extraction/scripts/` |
| `rr_api.js` + RR helpers | `researchrabbit/` |

The 4 schema-coupled scripts in `v3/scripts/` are kept locally because they hardcode AI503-specific column lists and required-field sets. They could move to `slr-extraction` if generalized to read columns/schema from `.slr/config.json`.

## Verification status

```
$ python scripts/verify_extractions.py
TOTAL: 32 papers · 2,382 leaves · 2,106 traced (88.4%) · 2,105 with bbox (88.4%)
All files pass schema checks.
```

## Pipeline (next steps for the 18 new papers)

1. **Re-download PDFs** for the 18 new papers from arXiv (DOIs in `docs/replacement_candidates.md`).
2. **Run marker** on the 18 new PDFs to produce JSONs.
3. **Convert** new JSONs to source-traced XML via [slr-extraction/scripts/json_to_claude_xml.py](../../../.claude/skills/slr-extraction/scripts/json_to_claude_xml.py).
4. **Extract** the 18 new papers using `slr-extraction`'s extractor agent — assign RP51-RP68.
5. **Re-run** `scripts/generate_comparative_table.py` and `scripts/generate_paper_explorer.py` — both auto-pick up the new extractions.
6. **Refresh** the RR visual graph PNG (Deliverable #2) from RR's graph view, save to `exports/`.
7. **Write** the 5,000-word paper.

## RP-ID numbering policy

The 50 papers in v3's parent collection use a contiguous **RP01–RP50** numbering with no gaps. Two sources contribute IDs:

- **32 IDs preserved from v2** for papers that survived the audit. v2 cross-references stay valid for these.
- **18 IDs reused** in the gaps left when v2's problematic papers were moved to sibling folders (duplicate / wrong-PDF / off-topic / tangential / survey). Filled by year ascending in numerical order — see [`exports/README.md`](exports/README.md) for the full mapping.

Reusing the gaps (rather than continuing RP51–RP68) keeps every RP01-RP50 in active use and allows the comparative table and citation lists to be sequential. Trade-off: 8 of the 18 reused slots have ≤1-year back-dating relative to their neighbors (e.g. RP14 = ZeroQuant 2022 between two 2023 papers). Acceptable in exchange for sequence continuity and minimal disruption to v2 references.

**Mapping at a glance:**

```
RP01  Paperno 2016   ┐  v2 (kept)
RP02  Vaswani 2017    │
...                   │
RP07  Nagel 2020      │
RP08  DeepSpeed-Inf 2022   ← gap-filled (was RP08 v2 Gholami 2022 → moved to survey)
RP09  Frantar 2022   ┐
RP10  Xiao 2022      │
RP11  LLM.int8 2022      ← gap-filled (was RP11 v2 SparseGPT → tangential)
RP12  Touvron 2023   │
RP13  Chee 2023      │
RP14  ZeroQuant 2022     ← gap-filled (was RP14 v2 Lin AWQ dup)
... (and so on through RP50)
```

Full per-paper mapping documented in [`exports/README.md`](exports/README.md).
