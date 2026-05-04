# ResearchRabbit collection move — cheat sheet

For each row: in the ResearchRabbit collection page, search the title in the "Find in list..." box, click the paper checkbox, then use the **Save** button (or drag) to move it to the destination collection.

**Existing collection**: `AI503 A1 - Quantization Safety SLR` (parent — keep as-is)  
**Existing subcollection**: `selected` (already created — final included corpus)  
**To create as siblings** (ResearchRabbit collections are flat, not nested):
  - `AI503 A1 - QS - duplicate`
  - `AI503 A1 - QS - wrong PDF`
  - `AI503 A1 - QS - off topic`
  - `AI503 A1 - QS - tangential`

After all moves are done, the parent collection should contain: 33 selected + 2 duplicate + 2 wrong PDF + 5 off topic + 8 tangential = 50 total. (Or the parent stays as a "master" with everything; subcollections just tag-like.)

---

## Tier 1 — Move to `duplicate` (drop)

| RP | Author | Year | Title to search in RR | Note |
|---|---|---:|---|---|
| RP14 | Lin | 2023 | AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration | Duplicate of RP15 (same AWQ paper). Drop RP14, keep RP15. |
| RP25 | Dahmani | 2024 | The Impact of Inference Acceleration on Bias of LLMs (Kirsten 2024 NAACL) | Wrong PDF (math paper) AND duplicate of RP24 (Kirsten). Drop entirely. |

## Tier 2 — Move to `wrong PDF` (re-download)

| RP | RR-listed title (search this) | Actual PDF (wrong, ignore) | Note |
|---|---|---|---|
| RP33 | Plug-and-Play: An Efficient Post-training Pruning Method for Large Language Models (Zhang 2024 ICLR) | Machine learning of discrete field theories with guaranteed ... | Wrong PDF (Offen physics paper). Re-download Zhang 2024 Plug-and-Play. |
| RP37 | Leveraging logit uncertainty for better knowledge distillation (Guo 2024 Scientific Reports) | Generalized Adversarial Code-Suggestions: Exploiting Context... | Wrong PDF (Rubel security paper). Re-download Guo 2024 Logit Uncertainty. |

## Tier 3 — Move to `off topic` (swap with replacements)

| RP | Author | Year | Title to search in RR | Note |
|---|---|---:|---|---|
| RP39 | Yang | 2025 | Qwen3 Technical Report | Off-topic — Qwen3 model release. |
| RP42 | Chen | 2025 | Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling | Off-topic — test-time scaling for reasoning verifiers. |
| RP43 | Kamirul | 2025 | R-Sparse R-CNN: SAR Ship Detection Based on Background-Aware Sparse Learnable Proposals | Off-topic — SAR ship detection (computer vision). |
| RP44 | Aljohani | 2025 | A Comprehensive Survey on the Trustworthiness of Large Language Models in Healthcare | Off-topic — healthcare LLM trustworthiness survey. |
| RP46 | Raza | 2025 | TRiSM for Agentic AI: A Review of Trust, Risk, and Security Management in LLM-based Agentic Multi-Agent Systems | Off-topic — agentic AI multi-agent governance survey. |

## Tier 4 — Move to `tangential` (review against final SLR scope)

| RP | Author | Year | Title to search in RR | Note |
|---|---|---:|---|---|
| RP11 | Frantar | 2023 | SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot | Tangential — SparseGPT pruning, composes with quantization. |
| RP19 | Liu | 2023 | Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time | Tangential — Deja Vu contextual sparsity. |
| RP23 | Li | 2024 | From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge | Tangential — LLM-as-a-Judge evaluation survey. |
| RP31 | Chua | 2024 | Post-Training Statistical Calibration for Higher Activation Sparsity | Tangential — SCAP activation sparsity (tests on quantized models). |
| RP40 | Haziza | 2025 | Accelerating Transformer Inference and Training with 2:4 Activation Sparsity | Tangential — 2:4 activation sparsity (uses FP8 quantization). |
| RP41 | Maximov | 2025 | From 2:4 to 8:16 sparsity patterns in LLMs for Outliers and Weights with Variance Correction | Tangential — 8:16 sparsity (adapts SmoothQuant scaling). |
| RP47 | Shrestha | 2025 | Polar Sparsity: High Throughput Batched LLM Inferencing with Scalable Contextual Sparsity | Tangential — Polar Sparsity (head sparsity for inference). |
| RP50 | Mi | 2025 | ACE: Exploring Activation Cosine Similarity and Variance for Accurate and Calibration-Efficient LLM Pruning | Tangential — ACE calibration-efficient pruning. |

---

## Quick steps in ResearchRabbit

1. Open `https://app.researchrabbit.ai/library/collection/df7b76c5-78b1-4c5f-bcbf-9ffba871ca31`.
2. Click the **+** icon next to "Collections" in the left sidebar (or right-click the parent collection) to create the 4 new sibling collections listed above.
3. For each paper in the tables below, in the **search-in-list** box at the top of the article list type a distinctive part of the title.
4. Click the paper's checkbox, then use **Save → Add to collection** and pick the destination, OR drag the paper onto the destination folder in the sidebar.
5. After all 17 papers are tagged, the remaining 33 stay in `selected`.

## Papers that stay in `selected` (33 total)

| RP | Author | Year | Title |
|---|---|---:|---|
| RP01 | Paperno | 2016 | The LAMBADA dataset: Word prediction requiring a broad discourse context |
| RP02 | Vaswani | 2017 | Attention Is All You Need |
| RP03 | Mihaylov | 2018 | Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question An |
| RP04 | Clark | 2019 | BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions |
| RP05 | Zellers | 2019 | HellaSwag: Can a Machine Really Finish Your Sentence? |
| RP06 | Bisk | 2019 | PIQA: Reasoning about Physical Commonsense in Natural Language |
| RP07 | Nagel | 2020 | Up or Down? Adaptive Rounding for Post-Training Quantization |
| RP08 | Gholami | 2022 | A Survey of Quantization Methods for Efficient Neural Network Inference |
| RP09 | Frantar | 2022 | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformer |
| RP10 | Xiao | 2022 | SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Languag |
| RP12 | Touvron | 2023 | Llama 2: Open Foundation and Fine-Tuned Chat Models |
| RP13 | Chee | 2023 | QuIP: 2-Bit Quantization of Large Language Models With Guarantees |
| RP15 | Lin | 2023 | AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acce |
| RP16 | Kim | 2023 | SqueezeLLM: Dense-and-Sparse Quantization |
| RP17 | Dettmers | 2023 | SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression |
| RP18 | Shao | 2023 | OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models |
| RP20 | Dubey | 2024 | The Llama 3 Herd of Models |
| RP21 | Tseng | 2024 | QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebo |
| RP22 | Hooper | 2024 | KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantizat |
| RP24 | Kirsten | 2024 | The Impact of Inference Acceleration on Bias of LLMs |
| RP26 | Hong | 2024 | Decoding Compressed Trust: Scrutinizing the Trustworthiness of Efficient LLMs Un |
| RP27 | Zhelnin | 2024 | GIFT-SW: Gaussian noise Injected Fine-Tuning of Salient Weights for LLMs |
| RP28 | Ashkboos | 2024 | SliceGPT: Compress Large Language Models by Deleting Rows and Columns |
| RP29 | Ashkboos | 2024 | QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs |
| RP30 | Egiazarian | 2024 | Extreme Compression of Large Language Models via Additive Quantization |
| RP32 | Belkhiter | 2024 | HarmLevelBench: Evaluating Harm-Level Compliance and the Impact of Quantization  |
| RP34 | Song | 2024 | PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU |
| RP35 | Sun | 2024 | FLATQUANT: Flatness Matters for LLM Quantization |
| RP36 | Liu | 2024 | SpinQuant: LLM Quantization with Learned Rotations |
| RP38 | Liu | 2024 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache |
| RP45 | Fang | 2025 | Smaller = Weaker? Benchmarking Robustness of Quantized LLMs in Code Generation |
| RP48 | Hu | 2025 | OSTQuant: Refining Large Language Model Quantization with Orthogonal and Scaling |
| RP49 | Fu | 2025 | Quantized but Deceptive? A Multi-Dimensional Truthfulness Evaluation of Quantize |