# Corpus Audit: Off-Topic, Wrong-PDF, and Duplicate Entries

The 50-paper corpus was pulled via ResearchRabbit and pre-screened by the student. After full-text reading by the extractor agents, **and a cross-check against the actual ResearchRabbit collection** (`https://app.researchrabbit.ai/library/collection/df7b76c5-78b1-4c5f-bcbf-9ffba871ca31`), four classes of issue surfaced:

1. **2 duplicate pairs** (RP14↔RP15, RP24↔RP25) — same paper indexed twice in ResearchRabbit under different DOIs (arXiv preprint vs. published version). One slot in each pair is wasted.
2. **2 wrong-PDF cases** (RP33, RP37) — the ResearchRabbit collection lists an on-topic paper, but the file actually downloaded into `papers_pdf/` is a different, unrelated paper. These can be **fixed by re-downloading** the correct PDF.
3. **5 truly off-topic entries** (RP39, RP42, RP43, RP44, RP46) — the ResearchRabbit collection itself contains these unrelated papers. They were imported wrongly during search expansion. **Swap with on-topic replacements.**
4. **7 tangential papers** — pruning / activation sparsity / inference engines. Adjacent to quantization but not core. **Keep or swap depending on SLR scope wording.**

After reconciling: the 50-slot corpus contains **48 unique papers**, of which **2 have wrong PDFs (recoverable)** and **5 are truly off-topic (need replacement)**. Net usable on-topic papers right now: **41**.

The audit was done by full-text reading; for each paper the actual title and authorship was extracted from the PDF, **then compared to the ResearchRabbit collection's listed title for the same RP slot**.

---

## Tier 1 — Exact duplicates (drop one of each pair)

ResearchRabbit indexes some papers under multiple DOIs (arXiv preprint + published version) as separate entries. When the bulk-download fetched both, the same paper landed in two RP slots.

| Pair | RR collection entry (arXiv version) | RR collection entry (published version) | Verification |
|---|---|---|---|
| **RP14 ↔ RP15** | RP14: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" — arXiv.org · DOI `10.48550/ARXIV.2306.00978` | RP15: "AWQ: Activation-aware Weight Quantization for **On-Device** LLM Compression and Acceleration" — Conference on Machine Learning and Systems · DOI `10.1145/3714983.3714987` | PDF MD5 identical: `2c803d59...` (19,804,976 bytes) |
| **RP24 ↔ RP25** | RP24: "The Impact of Inference Acceleration **Strategies** on Bias of LLMs" — arXiv (Cornell) | RP25: "The Impact of Inference Acceleration on Bias of LLMs" — NAACL 2024 | Same paper (NAACL version drops "Strategies" from the title) — but RP25's actual PDF is **also** wrong (see Tier 2 below) |

### Recommendation
- Drop **RP14**, keep RP15 (the published-version DOI is more authoritative).
- Drop **RP25** entirely — even if its PDF were correct, it would be a duplicate of RP24.

A corpus-wide MD5 scan confirmed RP14/RP15 is the **only byte-level duplicate**. The RP24/RP25 duplication is at the metadata level, not the PDF level (RP25's downloaded PDF doesn't match what its slot was supposed to contain).

---

## Tier 2 — Wrong PDF downloaded (re-download to fix)

For these RP slots, the ResearchRabbit collection lists an **on-topic** paper, but the file actually present in `papers_pdf/` is a **completely different paper**. Likely cause: the bulk-download grabbed the wrong arXiv ID or followed a stale link.

| RP | RR collection says (correct, on-topic) | Actual PDF in papers_pdf/ (wrong) | Status |
|---|---|---|---|
| **RP33** | Zhang 2024 — *"Plug-and-Play: An Efficient Post-training Pruning Method for Large Language Models"* — ICLR | A pure-physics paper by **Christian Offen** (Paderborn) — *"Machine learning of discrete field theories with guaranteed convergence and uncertainty quantification"* | **Re-download** — the intended paper is on-topic (LLM post-training pruning) |
| **RP37** | Guo 2024 — *"Leveraging logit uncertainty for better knowledge distillation"* — Scientific Reports | A security paper by **Rubel, Noppel, Wressnegger** (KIT) — *"Generalized Adversarial Code-Suggestions: Exploiting Contexts of LLM-based Code-Completion"* | **Re-download** — the intended paper is adjacent (knowledge distillation, a compression method) |
| **RP25** | Kirsten 2024 — *"The Impact of Inference Acceleration on Bias of LLMs"* — NAACL | A pure-mathematics paper by **Dahmani & Belaïdi** — complex differential equations | **Drop** — intended paper is a duplicate of RP24 anyway (see Tier 1) |

### Recommendation
- Re-download **RP33** from the ICLR DOI and re-run the marker pipeline + extraction agent for that slot.
- Re-download **RP37** from the Scientific Reports DOI and re-run.
- Drop **RP25** — no recovery, it's a Tier 1 duplicate as well.

---

## Tier 3 — Truly off-topic in the ResearchRabbit collection itself

These RR collection entries are correctly downloaded — the PDF matches what RR says — but the paper itself doesn't fit the SLR's quantization / edge-LLM scope. They were imported during ResearchRabbit's citation-expansion / "Recently Found" suggestions. **Swap them with on-topic replacements.**

| RP | Title | Authors | Why it does not fit |
|---|---|---|---|
| RP39 | *Qwen3 Technical Report* | Qwen team (Alibaba) | Foundation-model release / training methodology. No PTQ, no edge inference. Useful as a downstream target for quantization studies but adds nothing to the SLR's evidence base. |
| RP42 | *Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling* | Chen et al. (NeurIPS 2025) | Test-time scaling / verifier-guided search for reasoning. Different efficiency lever (search budget, not weights/activations). |
| RP43 | *R-Sparse R-CNN: SAR Ship Detection Based on Background-Aware Sparse Learnable Proposals* | Kamirul et al. | Computer-vision SAR ship detection with a Sparse R-CNN extension. No LLMs at all. |
| RP44 | *A Comprehensive Survey on the Trustworthiness of Large Language Models in Healthcare* | Aljohani et al. | Healthcare-LLM trustworthiness survey. Discusses GPT-4 / Med-PaLM-2 governance; never touches quantization, edge, or compression. |
| RP46 | *TRiSM for Agentic AI: A Review of Trust, Risk, and Security Management in LLM-based Agentic Multi-Agent Systems* | Raza et al. | Multi-agent governance survey. Zero quantization content. |

### Patterns
- 4/5 (RP39, RP43, RP44, RP46) are **surveys with non-overlapping topics** — they dilute the evidence base instead of strengthening it.
- 2/5 (RP43, RP44) are entirely outside the LLM domain (SAR remote sensing, healthcare governance).

### Recommendation
Remove all 5 from the ResearchRabbit collection and the corpus, then add 5+ on-topic replacements targeting the gaps listed below.

---

## Tier 4 — Tangential (relevant but not core quantization)

These 7 (or 8 — RP23 borderline) papers sit in pruning / activation-sparsity / inference-engine territory. Pruning composes with quantization (several explicitly test their methods on top of quantized models). **Whether to keep them depends on the SLR scope wording.**

| RP | Title | Method family | Argument for keeping | Argument for swapping |
|---|---|---|---|---|
| RP11 | SparseGPT | pruning | Joint 50% sparsity + 4-bit GPTQ on OPT-175B; the joint-quant column is directly comparable | Headline contribution is unstructured pruning, not a quantization algorithm |
| RP19 | Deja Vu (Liu 2023) | contextual sparsity | §5.2 Table 7 explicitly tests DEJAVU + W4A16; cross-validates that contextual sparsity composes with PTQ | Pruning paper at heart |
| RP23 | LLM-as-a-Judge survey (Li 2024) | none (survey) | Citation justifies use of LLM judges as an evaluation protocol | Survey topic is orthogonal; could move to Tier 3 |
| RP31 | SCAP (Chua 2024) | activation sparsity | Tests on pre-quantized 4-bit Llama-2-70B and 8-bit Llama-3.1; demonstrates sparsity stacks with PTQ | Headline contribution is activation sparsity, not quantization |
| RP40 | 2:4 Activation Sparsity (Haziza 2025) | structured sparsity | Uses FP8 row-wise quantization throughout the FFN; quantization is a load-bearing assumption | Primary novelty is sparsity |
| RP41 | 8:16 Sparsity (Maximov 2025) | structured sparsity | Adapts SmoothQuant scaling for variance correction in sparsity | Pure sparsity paper, theoretical hardware speedup only |
| RP47 | Polar Sparsity (Shrestha 2025) | head sparsity | Long-context, instruction-tuned, code-released; touches the same edge-deployment concern | Pruning paper, no quantization study |
| RP50 | ACE (Mi 2025) | pruning | Calibration-efficient pruning on Llama-3 family | No quantization tests; paper itself flags quantization as future work |

### Recommendation
- **If the SLR scope is "post-training compression for edge LLMs"** → keep all 7 tangential papers. Pruning + quantization tradeoffs become a sub-theme.
- **If the SLR scope is "post-training quantization specifically"** → swap RP19, RP23, RP31, RP40, RP41, RP47, RP50 for additional pure-PTQ papers (consider QuaRot variants, KV-int4 papers, recent W4A4 work).

---

## Net corpus state

| Category | Count | Net usable |
|---|---:|---:|
| Total RP slots | 50 | — |
| Tier 1 duplicates (drop RP14, RP25) | 2 | 48 |
| Tier 2 wrong-PDF, recoverable (RP33, RP37) | 2 | 48 once re-downloaded |
| Tier 3 truly off-topic (RP39, RP42, RP43, RP44, RP46) | 5 | 43 |
| Tier 4 tangential (judgment call) | 7-8 | 35-43 depending on scope |

**Practical action plan**:

1. Drop RP14, RP25 from the corpus index (one less unique paper each).
2. Re-download RP33 (Zhang 2024 *Plug-and-Play* pruning) and RP37 (Guo 2024 *Logit Uncertainty distillation*) from the correct DOIs; re-run marker + extraction.
3. Remove RP39, RP42, RP43, RP44, RP46 from the corpus and from the ResearchRabbit collection; replace with on-topic papers from the gap themes below.
4. Decide Tier 4 (tangential) keep-vs-swap based on final SLR scope.

---

## Suggested replacement search themes

The corpus (after removing Tiers 1-3) is thin on:

1. **Edge-hardware quantization** — only RP14 (AWQ on Jetson Orin Nano / RPi 4B) and RP15 (AWQ Orin) test on non-data-center hardware. Add ~3 papers on mobile / NPU / Jetson deployment.
2. **Safety post-quantization** — only RP24, RP26, RP32, RP45, RP49 evaluate safety after compression. Add 1-2 papers on jailbreak robustness or refusal-direction stability under quantization.
3. **Energy / power measurement** — only RP08 (survey) and RP12 (LLaMA-2 carbon table) report any energy figures. Add 1-2 papers measuring W or J/token under different bit-widths.
4. **Statistical rigor** — only ~13/50 papers report seeds/CIs/significance tests. Replacements that include error bars improve methodological coverage.

---

## How this audit was produced

- Each paper's full XML was read end-to-end by an extractor agent (see [.slr/system_prompt.txt](.slr/system_prompt.txt)).
- Agents identified `paper_type` and `method_family` from the abstract / title / methodology, not the filename.
- Off-topic and wrong-PDF papers triggered an explicit note in the agent's report.
- Filename / actual-author mismatches were caught by comparing the extracted `first_author` value with the filename author.
- The byte-level duplicate (RP14/RP15) was caught by MD5-hashing every PDF in `papers_pdf/` and grouping by hash.
- The metadata-level duplicate (RP24/RP25) and the wrong-PDF cases (RP33, RP37) were caught by **comparing the extracted PDF title against the ResearchRabbit collection's listed title for the same RP slot**.

To re-run this audit after swapping papers:
- Regenerate extractions for the changed RP IDs.
- Skim `paper_type` + `method_family` columns of [comparative_analysis_table.csv](comparative_analysis_table.csv).
- Re-MD5 all PDFs to check for duplicates.
- Re-open the ResearchRabbit collection and verify each RP slot's PDF title matches the listed entry title.
