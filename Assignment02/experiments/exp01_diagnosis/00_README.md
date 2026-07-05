# Experiment 1 — Diagnosis (does 4-bit quantization hurt safety?)

Goal: measure what post-training quantization does to a small model's safety and utility.

Naming: `exp01.{in|proc|out}{NN}.name.ext`. Read in number order.

```
exp01_diagnosis/
├── 01_input/     ← what goes in
│   └── exp01.in01.safety_dataset.csv      550 prompts (AdvBench 300 + HarmBench 200 + XSTest 50)
├── 02_processing/ ← what does the work  [needs GPU, Colab]
│   ├── exp01.proc01.quantization.ipynb    quantizes Qwen2.5-1.5B at FP16/INT8/NF4, logs each refusal
│   └── exp01.proc02.bugfix_notes.md       reproducibility notes (pinned versions, padding, regex)
└── 03_output/    ← what comes out
    ├── exp01.out01.gen_fp16.csv           per-prompt answer + refusal label at FP16
    ├── exp01.out02.gen_int8.csv           ... at INT8
    ├── exp01.out03.gen_nf4.csv            ... at 4-bit NF4
    ├── exp01.out04.results.csv            headline metrics (the "results" file)
    ├── exp01.out05.summary.md             same numbers, readable table
    └── exp01.out06.fig_refusal_vs_quant.png   the diagnosis figure
```

**Finding:** NF4 leaves narrow AdvBench refusal flat (0.997) but drops broad HarmBench
refusal 0.915 → 0.765, while utility barely moves. That loss is what Experiment 2 fixes.

**Handoff:** `exp01.out01..03` (the generations) are copied into
`../exp02_saqc/01_input/exp02.in01..03` — they are the input to Experiment 2.
