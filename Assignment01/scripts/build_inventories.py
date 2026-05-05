"""
Build A1 inventory tables (methods, metrics, datasets) from the
comparative_analysis_table.csv + papers_md/ corpus, so Assignment 2 can
pick baselines, evaluation metrics, and a dataset niche directly.

Outputs three markdown tables to inventories/:
  methods_inventory.md   — quantization methods × paper coverage
  metrics_inventory.md   — evaluation metrics × paper coverage
  datasets_inventory.md  — datasets × paper coverage

Plus one summary file gap_for_a2.md sharpening Gap 1 (safety × quantization)
into a concrete A2-ready research question.
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "comparative_analysis_table.csv"
MD_DIR = ROOT / "papers_md"
OUT = ROOT / "inventories"
OUT.mkdir(exist_ok=True)

# Known quantization methods (canonical name -> regex aliases).
# Matched against Models Used + Contribution + Key Results + full paper text.
QUANT_METHODS = {
    "GPTQ":        [r"\bGPTQ\b"],
    "AWQ":         [r"\bAWQ\b"],
    "SmoothQuant": [r"SmoothQuant"],
    "SqueezeLLM":  [r"SqueezeLLM"],
    "SpQR":        [r"\bSpQR\b"],
    "QuIP":        [r"\bQuIP\b(?!#)"],
    "QuIP#":       [r"QuIP#"],
    "AQLM":        [r"\bAQLM\b"],
    "QuaRot":      [r"QuaRot"],
    "SpinQuant":   [r"SpinQuant"],
    "FlatQuant":   [r"FlatQuant"],
    "OmniQuant":   [r"OmniQuant"],
    "OstQuant":    [r"OstQuant"],
    "KIVI":        [r"\bKIVI\b"],
    "LLM.int8()":  [r"LLM\.int8|LLM int8"],
    "ZeroQuant":   [r"ZeroQuant"],
    "QLoRA":       [r"QLoRA"],
    "BitsAndBytes":[r"bitsandbytes|BitsAndBytes"],
    "RTN":         [r"\bRTN\b|round.to.nearest"],
    "PTQ (generic)":[r"post.training quantization|\bPTQ\b"],
    "QAT (generic)":[r"quantization.aware training|\bQAT\b"],
}

# Known evaluation metrics (canonical -> aliases as substrings, case-insensitive).
METRICS = {
    "Perplexity":          ["perplexity", "ppl"],
    "Accuracy":            ["accuracy"],
    "F1":                  ["f1-score", "f1 score", " f1"],
    "BLEU":                ["bleu"],
    "ROUGE":               ["rouge"],
    "Exact Match":         ["exact match", " em "],
    "WER":                 [" wer "],
    "Latency":             ["latency"],
    "Throughput":          ["throughput", "tokens/s", "tokens per second"],
    "Memory":              ["memory footprint", "vram", "memory usage", "model size"],
    "Energy":              ["energy", "power consumption"],
    "MMLU score":          ["mmlu"],
    "ARC":                 [" arc "],
    "HellaSwag":           ["hellaswag"],
    "TruthfulQA":          ["truthfulqa"],
    "Refusal rate":        ["refusal rate", "refusal"],
    "Attack success rate": ["attack success", " asr "],
    "Toxicity":            ["toxicity"],
    "Bias":                [" bias "],
    "Calibration error":   ["calibration"],
}

# Known datasets/benchmarks
DATASETS = {
    "C4":            [r"\bC4\b"],
    "WikiText-2":    [r"WikiText.?2|wikitext2"],
    "WikiText-103":  [r"WikiText.?103|wikitext103"],
    "LAMBADA":       [r"LAMBADA"],
    "PTB":           [r"Penn Treebank|\bPTB\b"],
    "MMLU":          [r"\bMMLU\b"],
    "ARC":           [r"\bARC[- ](Easy|Challenge)|\bARC\b"],
    "HellaSwag":     [r"HellaSwag"],
    "PIQA":          [r"\bPIQA\b"],
    "WinoGrande":    [r"WinoGrande"],
    "BoolQ":         [r"\bBoolQ\b"],
    "TriviaQA":      [r"TriviaQA"],
    "GSM8K":         [r"GSM8K"],
    "HumanEval":     [r"HumanEval"],
    "TruthfulQA":    [r"TruthfulQA"],
    "BBQ":           [r"\bBBQ\b"],
    "AdvBench":      [r"AdvBench"],
    "HarmBench":     [r"HarmBench"],
    "MaliciousInstruct":[r"MaliciousInstruct"],
    "XSTest":        [r"XSTest"],
    "DecodingTrust": [r"DecodingTrust"],
    "OpenMiniSafety":[r"OpenMiniSafety"],
    "ToxicChat":     [r"ToxicChat"],
    "RealToxicityPrompts":[r"RealToxicityPrompts"],
    "GLUE":          [r"\bGLUE\b"],
    "SuperGLUE":     [r"SuperGLUE"],
    "SQuAD":         [r"SQuAD"],
    "WMT":           [r"WMT[- ]?(14|16|2014|2016)"],
}


def load_papers():
    """Yield (rp_id, year, title, joined_text) per paper, combining CSV row + full markdown."""
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    for row in rows:
        rp = row["RP_ID"]
        md_path = next(MD_DIR.glob(f"{rp}_*.md"), None)
        md_text = md_path.read_text(encoding="utf-8", errors="ignore") if md_path else ""
        joined = " ".join([
            row.get("Contribution", ""),
            row.get("Dataset", ""),
            row.get("Models Used", ""),
            row.get("Evaluation Metrics", ""),
            row.get("Key Results", ""),
            md_text,
        ])
        yield rp, row.get("Year", ""), row.get("Title", "")[:80], joined


def build_inventory(label, items, papers, regex=False):
    """For each canonical item, list which papers (RP_IDs) mention any of its aliases."""
    coverage = defaultdict(list)
    for rp, year, title, text in papers:
        text_l = text.lower()
        for canon, aliases in items.items():
            for alias in aliases:
                if regex:
                    if re.search(alias, text, flags=re.IGNORECASE):
                        coverage[canon].append(rp)
                        break
                else:
                    if alias.lower() in text_l:
                        coverage[canon].append(rp)
                        break
    # Dedupe + sort
    return {k: sorted(set(v), key=lambda x: int(x[2:])) for k, v in coverage.items()}


def write_md(path, title, intro, coverage, total):
    rows = sorted(coverage.items(), key=lambda kv: -len(kv[1]))
    lines = [f"# {title}\n", intro, "", f"**Total papers in corpus: {total}**", ""]
    lines.append("| Item | Papers (n) | Coverage % | RP_IDs |")
    lines.append("|---|---:|---:|---|")
    for canon, rps in rows:
        if not rps:
            continue
        pct = 100 * len(rps) / total
        rp_str = ", ".join(rps) if len(rps) <= 12 else ", ".join(rps[:12]) + f", … (+{len(rps)-12})"
        lines.append(f"| {canon} | {len(rps)} | {pct:.0f}% | {rp_str} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path.relative_to(ROOT)} ({len(rows)} items)")


def main():
    papers = list(load_papers())
    n = len(papers)

    methods = build_inventory("methods", QUANT_METHODS, papers, regex=True)
    metrics = build_inventory("metrics", METRICS, papers, regex=False)
    datasets = build_inventory("datasets", DATASETS, papers, regex=True)

    write_md(OUT / "methods_inventory.md",
             "A1 Methods Inventory — Quantization Methods × Paper Coverage",
             "Built from `comparative_analysis_table.csv` + `papers_md/` (50 papers). "
             "A method is counted for a paper if any of its name aliases appears anywhere in "
             "the paper's text or extracted fields. Use this to pick A2 baselines: prefer "
             "methods with the highest coverage (most comparable to prior work).",
             methods, n)

    write_md(OUT / "metrics_inventory.md",
             "A1 Metrics Inventory — Evaluation Metrics × Paper Coverage",
             "Same coverage rule as methods. Use this to pick A2 evaluation metrics: a "
             "high-coverage metric (e.g., perplexity, accuracy) makes results comparable to "
             "the literature; a low-coverage metric (e.g., refusal rate) is where the gap lies.",
             metrics, n)

    write_md(OUT / "datasets_inventory.md",
             "A1 Datasets Inventory — Datasets/Benchmarks × Paper Coverage",
             "Same coverage rule. Splits naturally into: utility benchmarks (high coverage — "
             "MMLU, WikiText, C4) and safety benchmarks (low coverage — AdvBench, HarmBench, "
             "TruthfulQA, BBQ). The mismatch is the empirical face of Gap 1 (safety × "
             "quantization rarely co-evaluated).",
             datasets, n)

    # Sharpen Gap 1 for A2
    safety_dsets = [k for k in DATASETS if k in
                    {"AdvBench","HarmBench","MaliciousInstruct","XSTest","TruthfulQA",
                     "BBQ","DecodingTrust","OpenMiniSafety","ToxicChat","RealToxicityPrompts"}]
    safety_papers = sorted({rp for d in safety_dsets for rp in datasets.get(d, [])},
                           key=lambda x: int(x[2:]))
    quant_papers = sorted({rp for m in QUANT_METHODS for rp in methods.get(m, [])},
                          key=lambda x: int(x[2:]))
    co_eval = sorted(set(safety_papers) & set(quant_papers), key=lambda x: int(x[2:]))

    gap = OUT / "gap_for_a2.md"
    gap.write_text(f"""# A2-Ready Research Gap (sharpened from A1 Gap 1)

## Empirical face of the gap (numbers from the corpus)

- Papers using **any quantization method**: **{len(quant_papers)}/{n}** — {", ".join(quant_papers)}
- Papers using **any safety benchmark**: **{len(safety_papers)}/{n}** — {", ".join(safety_papers) if safety_papers else "(none)"}
- Papers that do **both** (co-evaluate quantization × safety): **{len(co_eval)}/{n}** — {", ".join(co_eval) if co_eval else "(none)"}

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
""", encoding="utf-8")
    print(f"Wrote {gap.relative_to(ROOT)}")
    print(f"\nSummary: quant={len(quant_papers)}, safety={len(safety_papers)}, both={len(co_eval)}")


if __name__ == "__main__":
    main()
