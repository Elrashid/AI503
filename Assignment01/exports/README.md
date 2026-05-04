# Exports

Deliverable #1 (RR Export) for AI503 Assignment 1.

| File | Source | Use |
|---|---|---|
| `AI503_A1_RR_export_50papers_official.csv` | RR's UI export (Library → Select all → Export → CSV) | **Submit this** as the assignment's "Research Rabbit Export" deliverable |
| `AI503_A1_RR_export_50papers_extended.csv` | Custom build via `GET /user-articles?projectId=…&folderIds[]=PARENT` | Supplementary — adds `arxivId`, `isOpenAccess`, `forwardEdgeCount`, `backwardEdgeCount`, `firstAuthor`. Use when the comparative table needs citation counts. |

Both files cover the same 50 papers (the parent folder `AI503 A1 - Quantization Safety SLR`).

## `(missing DOI)` placeholder — official.csv preserved as-is

RR's UI CSV export writes the **literal string `(missing DOI)`** into the DOI column when the source paper lacks a venue DOI. Two papers in this corpus exhibit it:

| Title | Year | DOI value in `official.csv` | DOI value in `extended.csv` |
|---|---|---|---|
| Attention is All You Need | 2017 | `(missing DOI)` *(verbatim from RR)* | `10.48550/arXiv.1706.03762` |
| Up or Down? Adaptive Rounding for Post-Training Quantization | 2020 | `(missing DOI)` *(verbatim from RR)* | `10.48550/arXiv.2004.10568` |

Both are conference papers (NeurIPS / ICML) without formal venue DOIs.

**Policy:** `official.csv` keeps RR's `(missing DOI)` placeholder verbatim (provenance for the DOI column). The `extended.csv` carries patched arXiv DOIs (`10.48550/arXiv.<id>`) for downstream tools that need a non-empty DOI for every row.

## RP_ID column

Both CSVs have `RP_ID` as the first column. Every RP01–RP50 ID is used exactly once.

**Assignment policy:**
- **The 32 papers carried over from v2** keep their original v2 RP IDs (matched by exact title, arXiv-ID-in-DOI, or fuzzy title ≥ 0.80 — the last needed once for RP24, whose NAACL version dropped "Strategies" from the arXiv title).
- **The 18 new papers** (added via Similar Work / Recently Found) reuse the 18 RP IDs freed up when v2's problematic papers were moved to sibling folders. Sequential gap-fill in numerical order, with NEW-01..NEW-18 sorted by year ascending then title alphabetical:

| Filled gap | Year | Paper |
|---|---|---|
| RP08 | 2022 | DeepSpeed-Inference |
| RP11 | 2022 | LLM.int8() |
| RP14 | 2022 | ZeroQuant |
| RP19 | 2023 | Compressing LLMs: The Truth is Rarely Pure |
| RP23 | 2023 | DecodingTrust |
| RP25 | 2023 | PagedAttention / vLLM |
| RP31 | 2023 | H2O: Heavy-Hitter Oracle |
| RP33 | 2023 | FlexGen (High-throughput Generative Inference) |
| RP37 | 2023 | LLM in a flash |
| RP39 | 2023 | FastGen (Adaptive KV Cache) |
| RP40 | 2023 | QLoRA |
| RP41 | 2024 | Beyond Perplexity (Multi-dim Safety) |
| RP42 | 2024 | DuQuant |
| RP43 | 2024 | Edge to Giant (Lee, IJCAI) |
| RP44 | 2024 | HarmBench |
| RP46 | 2024 | TrustLLM |
| RP47 | 2025 | Inference economics |
| RP50 | 2025 | OpenMiniSafety (Investigating Quant Safety) |

**Caveats:**
- Adding the column means `official.csv` is **no longer byte-equivalent** to RR's raw UI export — the column is a project-level annotation. The original RR-emitted DOI value (`(missing DOI)` for Vaswani 2017 / Nagel 2020) is still preserved in the DOI column so the export's provenance is recoverable by dropping the `RP_ID` column.
- Internal chronology has minor (≤1 year) back-dating in 8 of the 18 reused slots (e.g. RP14 = 2022 ZeroQuant sits between RP13/RP15 = 2023 papers). This is the cost of preserving v2's RP numbering — option α was chosen over a strict chronological renumber.

**Caveat for any audit script reading `official.csv`:** a `null`/empty check on the DOI column will report 0 missing — `(missing DOI)` is a non-empty string. Use a check like:

```python
missing = sum(1 for r in rows
              if not (r.get('DOI') or '').strip()
              or (r.get('DOI') or '').strip() == '(missing DOI)')
```
