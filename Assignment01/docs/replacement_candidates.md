# Replacement Candidates — discovered via ResearchRabbit

This list was harvested through ResearchRabbit's discovery features by capturing the network responses to the **Recently Found** pane (`GET /recent-articles?isSaved=false`) and **Similar Work** for five seed papers (`POST /searches` → returns top-20 graph-similar candidates per seed).

| Seed paper | Why chosen as seed |
|---|---|
| Hong, 2024 — *Decoding Compressed Trust* (RP27) | Core safety+quantization paper |
| Belkhiter, 2024 — *HarmLevelBench* | Safety+quant benchmarking |
| Lin, 2023 — *AWQ: On-Device LLM Compression* (RP15) | Edge hardware seed |
| Fang, 2025 — *Smaller = Weaker?* | Robustness of quantized LLMs |
| Song, 2024 — *PowerInfer* | Consumer-grade GPU inference |

After deduping against the 66 already-saved articles in the project, **79 unique candidates** remained. The 13 below are the recommended additions, scored against the four gap themes flagged in [off_topic_papers.md](off_topic_papers.md):

1. **Edge hardware deployment** (current corpus: 2/50 papers test non-data-center hardware)
2. **Safety post-quantization** (current corpus: 5/50 papers)
3. **Energy / power measurement** (current corpus: 2/50)
4. **Statistical rigor** (current corpus: ~13/50 report seeds/CIs)

---

## Tier A — Direct gap-theme hits (priority adds)

### 1. Quantization Methods × Task Difficulty × Model Size: From Edge to Giant
- **arXiv:** [2409.11055](https://arxiv.org/abs/2409.11055)
- **DOI:** 10.24963/IJCAI.2025/902
- **Author:** Jemin Lee et al.
- **Year:** 2024 (IJCAI 2025)
- **Citations (fwd):** 17
- **Gap theme:** ⭐ Edge HW + safety
- **Abstract excerpt:** "Comprehensive evaluation of instruction-tuned models spanning 1B to 405B parameters, applying four quantization methods across 13 datasets. Findings: (1) quantized models generally surpass smaller FP16 baselines, yet often struggle with instruction-following and hallucination detection; (2) FP8 emerges as most robust across tasks; (3) AWQ tends to..."
- **Why include:** Directly addresses the edge–to-server quantization spectrum; fills the edge-hardware gap and includes hallucination/instruction-following as evaluation axes (extends our safety dimensions).

### 2. Investigating the Impact of Quantization Methods on the Safety and Reliability of LLMs
- **arXiv:** [2502.15799](https://arxiv.org/abs/2502.15799)
- **DOI:** 10.48550/ARXIV.2502.15799
- **Author:** Artyom Kharinaev et al.
- **Year:** 2025
- **Citations (fwd):** 10
- **Gap theme:** ⭐ Safety post-quantization
- **Abstract excerpt:** "We introduce **OpenMiniSafety**, a human-curated safety dataset with 1,067 questions. Public release of safety evaluations for four LLMs (quantized + full-precision), 4,268 annotated Q-A pairs. Assesses 66 quantized variants using four PTQ + two QAT methods across four safety benchmarks..."
- **Why include:** Largest study yet on quant-vs-safety; releases reusable safety dataset; covers PTQ and QAT for direct comparison.

### 3. Beyond Perplexity: Multi-dimensional Safety Evaluation of LLM Compression
- **arXiv:** [2407.04965](https://arxiv.org/abs/2407.04965)
- **DOI:** 10.18653/V1/2024.FINDINGS-EMNLP.901 (EMNLP Findings 2024)
- **Author:** Zhichao Xu et al.
- **Year:** 2024
- **Citations (fwd):** 35
- **Gap theme:** ⭐ Safety post-quantization
- **Abstract excerpt:** "Investigates compression along four dimensions: (1) degeneration harm (bias/toxicity in generation), (2) representational harm (biases in discriminative tasks), (3)..."
- **Why include:** EMNLP venue, four-dimensional safety evaluation framework. Methodological complement to RP27 (Hong 2024).

### 4. LLM in a flash: Efficient LLM Inference with Limited Memory
- **arXiv:** [2312.11514](https://arxiv.org/abs/2312.11514)
- **DOI:** 10.18653/V1/2024.ACL-LONG.678 (ACL 2024)
- **Author:** Keivan Alizadeh-Vahid et al. (Apple)
- **Year:** 2023 (ACL 2024)
- **Citations (fwd):** 288
- **Gap theme:** ⭐ Edge HW deployment
- **Abstract excerpt:** "Tackles running LLMs that exceed available DRAM by storing parameters in flash and bringing them to DRAM on demand. Constructs inference cost model accounting for flash characteristics; optimizes data volume transferred from flash and reads data in larger contiguous chunks..."
- **Why include:** Apple's on-device LLM paper; bridges the edge-deployment narrative with memory-hierarchy considerations that quantization-only papers ignore.

### 5. OstQuant: Refining LLM Quantization with Orthogonal and Scaling Transformations
- **arXiv:** [2501.13987](https://arxiv.org/abs/2501.13987)
- **DOI:** 10.48550/ARXIV.2501.13987
- **Year:** 2025
- **Citations (fwd):** 80
- **Gap theme:** PTQ depth (rotation-based methods alongside QuaRot/SpinQuant)
- **Abstract excerpt:** "PTQ challenge: uneven, heavy-tailed distributions expand quantization range. Introduces Quantization Space Utilization Rate (QSUR), a novel metric that assesses the quantizability of transformed weights/activations. Couples it with orthogonal + scaling transformations..."
- **Why include:** Recent rotation-based PTQ; fits next to QuaRot (RP) and SpinQuant (RP). Strong methodological depth.

---

## Tier B — Strong methodological/benchmarking adds

### 6. DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models
- **arXiv:** [2306.11698](https://arxiv.org/abs/2306.11698)
- **Author:** Boxin Wang et al.
- **Year:** 2023 (NeurIPS 2023)
- **Citations (fwd):** 792
- **Gap theme:** Safety benchmark (used by RP27 — Hong 2024 — as evaluation framework)
- **Why include:** The trust benchmark RP27 cites for its 8 safety dimensions. Including it gives the SLR direct lineage on how safety post-quant studies score models.

### 7. HarmBench: Standardized Evaluation for Automated Red Teaming and Robust Refusal
- **arXiv:** [2402.04249](https://arxiv.org/abs/2402.04249)
- **Author:** Mantas Mazeika et al.
- **Year:** 2024 (ICML 2024)
- **Citations (fwd):** 1,150
- **Gap theme:** Safety methodology (red-teaming benchmark referenced by Belkhiter / HarmLevelBench)
- **Why include:** Foundational red-teaming benchmark; provides the HarmBench evaluation protocol that several quant-safety papers in our corpus build on.

### 8. TrustLLM: Trustworthiness in Large Language Models
- **arXiv:** [2401.05561](https://arxiv.org/abs/2401.05561)
- **Author:** Lichao Sun et al.
- **Year:** 2024
- **Citations (fwd):** 434
- **Gap theme:** Safety / trust survey
- **Why include:** Multi-axis trust framework cited by recent quant-safety papers; useful for the comparative analysis to align safety dimensions across studies.

### 9. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
- **arXiv:** [2208.07339](https://arxiv.org/abs/2208.07339)
- **Author:** Tim Dettmers, M. Lewis, Younes Belkada, Luke Zettlemoyer
- **Year:** 2022 (NeurIPS 2022)
- **Citations (fwd):** 1,193+
- **Gap theme:** PTQ baseline / outlier handling
- **Why include:** **Foundational** Int8 paper — the outlier-feature observation it introduced is cited by virtually every PTQ paper in the corpus. Surprising omission from the current 50.

### 10. ZeroQuant: Efficient and Affordable PTQ for Large-Scale Transformers
- **arXiv:** [2206.01861](https://arxiv.org/abs/2206.01861)
- **Author:** Zhewei Yao et al. (Microsoft DeepSpeed)
- **Year:** 2022 (NeurIPS 2022)
- **Citations (fwd):** 855
- **Gap theme:** PTQ baseline
- **Why include:** Together with LLM.int8(), the PTQ-system baseline for hardware-friendly quantization. Cited by SmoothQuant, GPTQ, AWQ.

### 11. Compressing LLMs: The Truth is Rarely Pure and Never Simple
- **arXiv:** [2310.01382](https://arxiv.org/abs/2310.01382)
- **Author:** Ajay Jaiswal et al.
- **Year:** 2023 (ICLR 2024)
- **Citations (fwd):** 84
- **Gap theme:** Compression analysis (cross-method study)
- **Why include:** Critical re-analysis of compression methods (pruning + quantization); gives the SLR a hedge/critique chapter and supports the "evaluation rigor" narrative.

### 12. QLoRA: Efficient Finetuning of Quantized LLMs
- **arXiv:** [2305.14314](https://arxiv.org/abs/2305.14314)
- **Author:** Tim Dettmers et al.
- **Year:** 2023 (NeurIPS 2023)
- **Citations (fwd):** 5,131
- **Gap theme:** 4-bit + fine-tuning intersection
- **Why include:** Bridges PTQ/QAT and LoRA; relevant for safety-after-quantization fine-tuning interaction (RP-level paper).

### 13. Inference economics of language models
- **arXiv:** [2506.04645](https://arxiv.org/abs/2506.04645)
- **Year:** 2025
- **Citations (fwd):** 11
- **Gap theme:** Energy/cost (partial — economics, not direct power measurement)
- **Why include:** Pareto-frontier analysis of cost-per-token vs. serial speed; helps frame the deployment-economics chapter.

---

---

## Tier C — Second pass (KV-cache + serving systems)

After the four surveys (RP08, RP23, RP44, RP46) were moved to a new `AI503 A1 - QS - survey` folder, a second `RR.discover` round was run with **KIVI**, **KVQuant**, and **FlatQuant** as seeds (focus: KV-cache compression and serving infrastructure). 91 unique candidates accumulated; after deduping against saved + Tier A/B picks, the five additions below are the strongest on-topic fits.

### 14. DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs
- **arXiv:** [2406.01721](https://arxiv.org/abs/2406.01721)
- **Author:** Haokun Lin et al.
- **Year:** 2024 (NeurIPS 2024)
- **Citations (fwd):** 166
- **Gap theme:** PTQ depth — outlier handling
- **Abstract excerpt:** "Quantization of LLMs faces challenges from outlier activations that impede low-bit representation. Traditional approaches address Normal Outliers but struggle with Massive Outliers that show significantly larger values, leading to performance degradation. We introduce DuQuant — a dual transformation (rotation + permutation) that redistributes outliers..."
- **Why include:** Recent NeurIPS PTQ method that complements QuaRot/SpinQuant rotation-based approaches with an explicit dual-transformation analysis; directly comparable to existing rotation methods in our corpus.

### 15. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs
- **arXiv:** [2306.14048](https://arxiv.org/abs/2306.14048)
- **Author:** Zhenyu Zhang et al.
- **Year:** 2023 (NeurIPS 2023)
- **Citations (fwd):** 765
- **Gap theme:** KV-cache compression (composes with quantization for edge serving)
- **Abstract excerpt:** "LLMs are cost-prohibitive to deploy, especially for long-content generation. The KV cache stored in GPU memory scales linearly with sequence length and batch size. We introduce a novel approach for KV-cache compression based on the heavy-hitter hypothesis..."
- **Why include:** Foundational KV-cache compression paper. Quantization compresses *parameters*; H2O compresses *runtime state*. Both are needed for edge inference and the comparative table benefits from the split.

### 16. Efficient Memory Management for LLM Serving with PagedAttention (vLLM)
- **arXiv:** [2309.06180](https://arxiv.org/abs/2309.06180)
- **Author:** Woosuk Kwon et al.
- **Year:** 2023 (SOSP 2023)
- **Citations (fwd):** 6,223
- **Gap theme:** Serving infrastructure (the substrate quantization deploys onto)
- **Abstract excerpt:** "High throughput serving of LLMs requires batching many requests, but KV cache memory grows and shrinks dynamically. We propose PagedAttention, an attention algorithm inspired by classical virtual memory paging..."
- **Why include:** The landmark serving-system paper (SOSP 2023, 6.2k citations). Quantization gains are realized through such systems; including it grounds the deployment chapter.

### 17. Adaptive KV Cache Compression for LLMs (FastGen)
- **arXiv:** [2310.01801](https://arxiv.org/abs/2310.01801)
- **Author:** Suyu Ge et al.
- **Year:** 2023
- **Citations (fwd):** 545
- **Gap theme:** Adaptive KV-cache compression
- **Abstract excerpt:** "Plug-and-play method that reduces memory footprint of generative inference. Conducts targeted profiling of attention modules; constructs KV cache adaptively: evicting long-range contexts on attention heads emphasizing local context, evicting non-special tokens on heads attending to special tokens..."
- **Why include:** Per-head adaptive KV compression — pairs naturally with KV-quantization papers (KIVI, KVQuant) in the corpus. Strengthens the KV-strategies sub-theme.

### 18. DeepSpeed-Inference: Enabling Efficient Inference of Transformer Models at Scale
- **arXiv:** [2207.00032](https://arxiv.org/abs/2207.00032)
- **DOI:** 10.1109/SC41404.2022.00051 (SC '22)
- **Author:** Reza Yazdani Aminabadi et al. (Microsoft)
- **Year:** 2022
- **Citations (fwd):** 776
- **Gap theme:** Inference systems
- **Abstract excerpt:** "Multi-GPU inference solution that minimizes latency while maximizing throughput for both dense and sparse transformers; heterogeneous inference solution leveraging CPU/NVMe memory to enable inference of models that exceed aggregate GPU memory..."
- **Why include:** SC '22 paper; provides the baseline against which AWQ/GPTQ/SmoothQuant report end-to-end latency improvements. Useful for the serving-comparison narrative.

---

## Coverage check (after adding all 13)

| Gap theme | Before | After | Net change |
|---|---:|---:|---:|
| Edge HW deployment | 2 | 4 | +2 (Edge to Giant, LLM in a flash) |
| Safety post-quantization | 5 | 9 | +4 (OpenMiniSafety, Beyond Perplexity, DecodingTrust, HarmBench/TrustLLM as benchmarks) |
| Energy / power measurement | 2 | 3 | +1 (Inference economics — cost not power, partial) |
| Statistical rigor | 13/50 | needs check on each new paper | — |
| Foundational PTQ baselines | — | +2 (LLM.int8, ZeroQuant) | strengthens lineage |
| Recent rotation PTQ | — | +1 (OstQuant) | extends the QuaRot/SpinQuant trio |

**Net corpus after replacements:** 50 - 5 off-topic - 2 wrong-PDF - 2 dups - 1 survey moved out of parent + 13 Tier A/B + 5 Tier C = **58 on-topic papers** (58 ≥ 50 ✓, with margin)

---

## Paste-ready add snippet (for RR DevTools console)

After installing `rr_api.js`, run:

```js
// Save the 13 candidates into the parent collection by their RR articleId
// Note: this assumes RR has indexed each paper. If a paper isn't yet in RR's database, you'd need to search for it via the UI first.
const arxivIds = [
  '2409.11055', '2502.15799', '2407.04965', '2312.11514', '2501.13987',
  '2306.11698', '2402.04249', '2401.05561', '2208.07339', '2206.01861',
  '2310.01382', '2305.14314', '2506.04645'
];
// Look up each by arxiv via /user-articles or search endpoint
// (manual flow: paste arxivId into RR search bar, click + Save)
```

The cleanest path is still to add them through the UI: open `https://app.researchrabbit.ai`, paste each arXiv ID into Search, then **Save → AI503 A1 - Quantization Safety SLR** for each one.

---

## Final RP-ID assignments (after option α gap-fill)

The 18 candidates received the 18 RP IDs that opened up when v2's problematic papers were moved to sibling folders. Sequential gap-fill in numerical order, with candidates sorted by year ascending then title alphabetical:

| RP | Year | Paper | Tier | arXiv |
|---|---|---|---|---|
| RP08 | 2022 | DeepSpeed-Inference | C | 2207.00032 |
| RP11 | 2022 | LLM.int8() | B | 2208.07339 |
| RP14 | 2022 | ZeroQuant | B | 2206.01861 |
| RP19 | 2023 | Compressing LLMs: The Truth is Rarely Pure | B | 2310.01382 |
| RP23 | 2023 | DecodingTrust | B | 2306.11698 |
| RP25 | 2023 | PagedAttention / vLLM | C | 2309.06180 |
| RP31 | 2023 | H2O: Heavy-Hitter Oracle | C | 2306.14048 |
| RP33 | 2023 | FlexGen (High-throughput Generative Inference) | C* | 2303.06865 |
| RP37 | 2023 | LLM in a flash | A | 2312.11514 |
| RP39 | 2023 | FastGen (Adaptive KV Cache) | C | 2310.01801 |
| RP40 | 2023 | QLoRA | B | 2305.14314 |
| RP41 | 2024 | Beyond Perplexity (Multi-dim Safety) | A | 2407.04965 |
| RP42 | 2024 | DuQuant | C | 2406.01721 |
| RP43 | 2024 | Edge to Giant (Lee, IJCAI 2025) | A | 2409.11055 |
| RP44 | 2024 | HarmBench | B | 2402.04249 |
| RP46 | 2024 | TrustLLM | B | 2401.05561 |
| RP47 | 2025 | Inference economics | A | 2506.04645 |
| RP50 | 2025 | OpenMiniSafety (Investigating Quant Safety) | A | 2502.15799 |

C* = FlexGen was added later as the replacement for the OstQuant duplicate that RR had indexed twice (same arXiv 2501.13987 under two different `articleId`s).

OstQuant from the original Tier A pool kept its kept-paper slot RP48 (carried over from v2's numbering), not a gap ID — that's why it doesn't appear in the gap-fill table.

---

## How this list was generated

1. Captured `localStorage.tokens.sessionToken` and `localStorage.projectId` from a logged-in RR session.
2. Patched `window.fetch` to log every response from `api.researchrabbit.ai`.
3. Clicked each seed paper → Similar Work tab; the React app POSTs `/searches` with the seed's `userArticleId` and `articleId` and returns 20 graph-similar candidates per seed.
4. Also called `GET /recent-articles?isSaved=false&projectId=...` for the 7 unsaved suggestions RR auto-generates.
5. Deduped against the 66 saved articles (by `articleId` and `arxivId`).
6. Filtered out generic foundational papers (Attention, BERT, GPT-3 baselines).
7. Scored remaining candidates against the four gap themes from the corpus audit.
8. Persisted full pool to `localStorage.__claude_candidates` (still in the RR tab) — re-readable for future passes.

Endpoints involved (now documented in [.claude/skills/researchrabbit/references/api_endpoints.md](../../../.claude/skills/researchrabbit/references/api_endpoints.md)):
- `POST /searches` — body: `{searchRequest: {type:"singleSet", set:{edgeMode:"both"|"backward"|"forward", userArticleIds:[...], finalArticleIds:[...]}}, page, per}` → returns `{results: "<JSON-encoded {totalCount, items:[{id, score, details:{...}}]}>"}` (results is a stringified JSON inside the outer JSON)
- `GET /recent-articles?isSaved=true|false&projectId=...&page=1&per=300` — recently found / saved papers in the project
- `GET /searches/{searchId}` — refresh a saved search
- `POST /search-sessions` — create a search session linking a paper click to a search
