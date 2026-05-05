# A2-Ready Research Gap (sharpened from A1 Gap 1)

## Empirical face of the gap (numbers from the corpus)

- Papers using **any quantization method**: **37/50** — RP07, RP09, RP10, RP11, RP13, RP14, RP15, RP16, RP17, RP18, RP19, RP20, RP21, RP22, RP24, RP26, RP27, RP28, RP29, RP30, RP31, RP32, RP33, RP34, RP35, RP36, RP37, RP38, RP40, RP41, RP42, RP43, RP45, RP46, RP48, RP49, RP50
- Papers using **any safety benchmark**: **13/50** — RP12, RP20, RP23, RP24, RP26, RP32, RP38, RP41, RP43, RP44, RP46, RP49, RP50
- Papers that do **both** (co-evaluate quantization × safety): **10/50** — RP20, RP24, RP26, RP32, RP38, RP41, RP43, RP46, RP49, RP50

The intersection set is the empirical evidence for Gap 1: most quantization
papers stop at perplexity/accuracy; most safety papers stop at FP16 models.

## Research question (A2-ready)

> **RQ:** Does post-training quantization to 4-bit and lower degrade transformer
> LLM **safety behaviour** (refusal of harmful prompts, false-refusal of benign
> prompts) more, less, or differently than it degrades **utility** (MMLU,
> perplexity)?

## Hypothesis

> **H1:** At W4A16 and below, refusal rate on adversarial prompts (AdvBench /
> HarmBench) drops disproportionately faster than MMLU accuracy, with the gap
> widening at W3 and W2.

## Why this is A2-feasible

- **Dataset (A2 §3, 20%)**: composite curated benchmark from AdvBench (520
  prompts) + HarmBench subset + XSTest (250 false-refusal prompts) + a small
  manually annotated severity layer. Fits the rubric's "non-trivial" bar.
- **Models (A2 §4)**: 1–3B open models (Llama-3.2-1B/3B, Qwen2.5-1.5B) — fit on
  Colab free.
- **Baselines (mandatory)**: FP16 baseline + the highest-coverage quantization
  methods from the methods inventory (GPTQ, AWQ).
- **Metrics**: refusal rate, attack success rate, MMLU, perplexity — all in the
  metrics inventory.
- **Novelty**: most prior co-eval work uses 7B+ models on server hardware; the
  1–3B/edge regime is under-studied.

## Files this gap depends on

- `inventories/methods_inventory.md` — picks GPTQ/AWQ as A2 baselines.
- `inventories/metrics_inventory.md` — confirms refusal-rate is low-coverage
  (the gap) while perplexity is high-coverage (the comparator).
- `inventories/datasets_inventory.md` — confirms the safety-benchmark gap.
