# Experiment 2 — SAQC (can a cheap router recover the lost safety?)

Goal: recover the HarmBench safety that 4-bit lost, without paying full memory.
Method: run NF4 first; re-run only *suspicious* compliances at INT8, chosen by a learned risk gate.

Naming: `exp02.{in|proc|out}{NN}.name.ext`. Read in number order. **No GPU needed.**

```
exp02_saqc/
├── 01_input/     ← copies of Experiment 1's output (the handoff)
│   ├── exp02.in01.gen_fp16.csv     = exp01.out01
│   ├── exp02.in02.gen_int8.csv     = exp01.out02
│   ├── exp02.in03.gen_nf4.csv      = exp01.out03
│   └── exp02.in04.safety_dataset.csv = exp01.in01
├── 02_processing/ ← the scripts, in run order
│   ├── exp02.proc01.router_core.py     shared helper: loads data, cost model (imported by the others)
│   ├── exp02.proc02.risk_gate.py       explores learned gate vs keyword vs oracle
│   └── exp02.proc03.saqc_experiment.py MAIN — runs the full study → writes 03_output
└── 03_output/    ← results used in the paper (Section 8)
    ├── exp02.out01.policy_comparison.csv   the headline table
    ├── exp02.out02.gate_cv_metrics.csv     gate ROC-AUC / F1 / precision / recall
    ├── exp02.out03.threshold_sweep.csv     safety/cost vs threshold
    ├── exp02.out04.traffic_mix.csv         cost on benign-heavy real traffic
    ├── exp02.out05.fragile_by_category.csv which fragile prompts are recovered
    ├── exp02.out06.fig1_pareto.png         safety vs cost (Figure 8.1)
    ├── exp02.out07.fig2_threshold_sweep.png (Figure 8.2)
    ├── exp02.out08.fig3_gate_roc.png        (Figure 8.3)
    └── exp02.out09.fig4_gap_recovery.png    (Figure 8.4)
```

**To reproduce all outputs** (seconds, no GPU):
```
cd 02_processing
python exp02.proc03.saqc_experiment.py
```

**Finding:** the learned-gate cascade recovers the full HarmBench gap (0.765 → 0.930,
matching INT8) at ~4.7 average bits, escalating only ~9% of prompts.
