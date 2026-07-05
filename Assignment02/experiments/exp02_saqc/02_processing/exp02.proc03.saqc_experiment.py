"""
SAQC — Safety-Aware Quantization Cascade: full experiment (no GPU).

Produces every table and figure for the SAQC study, computed from the diagnosis
run's per-prompt refusal labels (FP16/INT8/NF4) saved in ../01_input, with:

  1. Gate classifier quality under cross-validation, averaged over seeds (mean +- std).
  2. A threshold sweep -> the safety vs. efficiency trade-off curve.
  3. Seed-robustness for the headline cascade numbers.
  4. Per-category breakdown of fragile prompts and recovery.
  5. Figures: Pareto (safety vs avg-bits), threshold sweep, gate ROC, gap-recovery bars.

All results are written to ../03_output/ as exp02.out01..09.
Reproduce: python exp02.proc03.saqc_experiment.py  (self-contained, fixed seed list).
"""
import csv, os
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,
                             roc_curve, confusion_matrix)

# load the shared helper (filename has dots, so import via importlib)
import importlib.util as _u
_s = _u.spec_from_file_location(
    "router_core", os.path.join(os.path.dirname(__file__), "exp02.proc01.router_core.py"))
R = _u.module_from_spec(_s); _s.loader.exec_module(R)

SEEDS = [42, 7, 123, 2024, 31]
N_SPLITS = 5
OUT = os.path.join(os.path.dirname(__file__), "..", "03_output")
os.makedirs(OUT, exist_ok=True)
# all outputs land flat in 03_output with exp02.outNN names (CSV + PNG)
TBL = OUT
FIG = OUT

# ---- chart style --------------------------------------------------------------
plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
BLUE, GREY, RED, GREEN = "#4472C4", "#A5A5A5", "#C00000", "#548235"


# ---- data --------------------------------------------------------------------
def build_rows():
    rows = R.load()
    for d in rows:
        d["harmful"] = 1 if d["expect"] == "refuse" else 0
    return rows


def oof_proba(rows, seed):
    X = np.array([d["prompt"] for d in rows])
    y = np.array([d["harmful"] for d in rows])
    proba = np.zeros(len(rows))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(max_iter=1000, class_weight="balanced", C=4.0),
        )
        clf.fit(X[tr], y[tr])
        proba[te] = clf.predict_proba(X[te])[:, 1]
    return proba


# ---- cascade simulation ------------------------------------------------------
def cascade(rows, fire, esc_to="INT8"):
    by = {"AdvBench": [0, 0], "HarmBench": [0, 0], "XSTest": [0, 0]}
    bits = esc = 0
    for d in rows:
        used = R.BITS["NF4"]
        if d["refused_NF4"]:
            refused = True
        elif fire(d):
            used += R.BITS[esc_to]; refused = d[f"refused_{esc_to}"]; esc += 1
        else:
            refused = False
        s = by[d["source"]]; s[1] += 1; s[0] += 1 if refused else 0
        bits += used
    rate = lambda k: by[k][0] / by[k][1]
    return dict(advbench=rate("AdvBench"), harmbench=rate("HarmBench"),
                xstest=rate("XSTest"), avg_bits=bits / len(rows), esc=esc / len(rows))


def static_metrics(rows, q):
    by = {"AdvBench": [0, 0], "HarmBench": [0, 0], "XSTest": [0, 0]}
    for d in rows:
        s = by[d["source"]]; s[1] += 1; s[0] += 1 if d[f"refused_{q}"] else 0
    rate = lambda k: by[k][0] / by[k][1]
    return dict(advbench=rate("AdvBench"), harmbench=rate("HarmBench"),
                xstest=rate("XSTest"), avg_bits=float(R.BITS[q]), esc=0.0)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


# ---- main --------------------------------------------------------------------
def main():
    rows = build_rows()
    n = len(rows)
    print(f"prompts: {n}  | harmful={sum(d['harmful'] for d in rows)} "
          f"benign={sum(1-d['harmful'] for d in rows)}\n")

    # 1. gate quality across seeds (OOF) ---------------------------------------
    y = np.array([d["harmful"] for d in rows])
    probas = {s: oof_proba(rows, s) for s in SEEDS}
    aucs, f1s, precs, recs = [], [], [], []
    for s in SEEDS:
        p = probas[s]
        aucs.append(roc_auc_score(y, p))
        pred = (p >= 0.30).astype(int)
        f1s.append(f1_score(y, pred)); precs.append(precision_score(y, pred))
        recs.append(recall_score(y, pred))
    def ms(a): return f"{np.mean(a):.3f} +- {np.std(a):.3f}"
    print("Gate quality (5-fold OOF, mean +- std over 5 seeds):")
    print(f"  ROC-AUC={ms(aucs)}  F1={ms(f1s)}  Precision={ms(precs)}  Recall={ms(recs)} (thr 0.30)\n")
    write_csv(os.path.join(TBL, "exp02.out02.gate_cv_metrics.csv"),
              ["seed", "roc_auc", "f1@0.30", "precision@0.30", "recall@0.30"],
              [[s, f"{roc_auc_score(y,probas[s]):.4f}",
                f"{f1_score(y,(probas[s]>=0.3).astype(int)):.4f}",
                f"{precision_score(y,(probas[s]>=0.3).astype(int)):.4f}",
                f"{recall_score(y,(probas[s]>=0.3).astype(int)):.4f}"] for s in SEEDS])

    # 1b. gate confusion matrix + FPR/FNR at thr 0.30 (per seed + mean) ---------
    # y: 1 = harmful (positive), 0 = benign. labels=[0,1] keeps the cell order
    # fixed even if a seed produces no negatives/positives in a class.
    cm_rows, cm_acc = [], np.zeros((2, 2), dtype=float)
    for s in SEEDS:
        pred = (probas[s] >= 0.30).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        cm_acc += np.array([[tn, fp], [fn, tp]])
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        cm_rows.append([s, tp, fp, fn, tn, f"{fpr:.4f}", f"{fnr:.4f}"])
    cm_mean = cm_acc / len(SEEDS)         # [[TN, FP], [FN, TP]]
    m_tn, m_fp = cm_mean[0]; m_fn, m_tp = cm_mean[1]
    m_fpr = m_fp / (m_fp + m_tn) if (m_fp + m_tn) else 0.0
    m_fnr = m_fn / (m_fn + m_tp) if (m_fn + m_tp) else 0.0
    cm_rows.append(["mean", f"{m_tp:.1f}", f"{m_fp:.1f}", f"{m_fn:.1f}", f"{m_tn:.1f}",
                    f"{m_fpr:.4f}", f"{m_fnr:.4f}"])
    write_csv(os.path.join(TBL, "exp02.out11.gate_confusion.csv"),
              ["seed", "tp", "fp", "fn", "tn", "fpr", "fnr"], cm_rows)
    print(f"Gate confusion (mean over seeds, thr 0.30): "
          f"TP={m_tp:.1f} FP={m_fp:.1f} FN={m_fn:.1f} TN={m_tn:.1f} "
          f"| FPR={m_fpr*100:.1f}% FNR={m_fnr*100:.1f}%\n")

    # mean proba across seeds for the reported cascade
    mean_p = np.mean([probas[s] for s in SEEDS], axis=0)
    for d, p in zip(rows, mean_p):
        d["p_harmful"] = p

    # 2. policy comparison table -----------------------------------------------
    policies = []
    for q in ("FP16", "INT8", "NF4"):
        m = static_metrics(rows, q); policies.append((f"static {q}", m))
    kw = cascade(rows, lambda d: bool(R.HARM_KEYWORDS.search(d["prompt"])))
    policies.append(("cascade: keyword", kw))
    learned = cascade(rows, lambda d: d["p_harmful"] >= 0.30)
    policies.append(("cascade: learned (thr 0.30)", learned))
    oracle = cascade(rows, lambda d: d["refused_INT8"])
    policies.append(("cascade: oracle", oracle))

    print(f"{'policy':<30}{'AdvB':>7}{'HarmB':>7}{'XS-FR':>7}{'avgBits':>9}{'%escal':>8}")
    print("-" * 68)
    tbl = []
    for name, m in policies:
        print(f"{name:<30}{m['advbench']*100:6.1f}%{m['harmbench']*100:6.1f}%"
              f"{m['xstest']*100:6.1f}%{m['avg_bits']:7.2f}b{m['esc']*100:7.1f}%")
        tbl.append([name, f"{m['advbench']:.4f}", f"{m['harmbench']:.4f}",
                    f"{m['xstest']:.4f}", f"{m['avg_bits']:.3f}", f"{m['esc']:.4f}"])
    write_csv(os.path.join(TBL, "exp02.out01.policy_comparison.csv"),
              ["policy", "refusal_advbench", "refusal_harmbench",
               "false_refusal_xstest", "avg_bits", "pct_escalated"], tbl)

    # gap recovery headline
    nf4_h = static_metrics(rows, "NF4")["harmbench"]
    int8_h = static_metrics(rows, "INT8")["harmbench"]
    gap = int8_h - nf4_h
    rec = (learned["harmbench"] - nf4_h) / gap if gap else float("nan")
    print(f"\nHarmBench gap (INT8 - NF4) = {gap*100:.1f}pp; learned cascade recovers "
          f"{rec*100:.0f}% at {learned['avg_bits']:.2f} avg bits "
          f"(NF4=4.0, INT8=8.0)\n")

    # 3. threshold sweep (safety vs efficiency) --------------------------------
    sweep = []
    for thr in np.round(np.arange(0.05, 0.96, 0.05), 2):
        m = cascade(rows, lambda d, t=thr: d["p_harmful"] >= t)
        sweep.append((thr, m["harmbench"], m["xstest"], m["avg_bits"], m["esc"]))
    write_csv(os.path.join(TBL, "exp02.out03.threshold_sweep.csv"),
              ["threshold", "harmbench", "xstest_fr", "avg_bits", "pct_escalated"],
              [[t, f"{h:.4f}", f"{x:.4f}", f"{b:.3f}", f"{e:.4f}"]
               for (t, h, x, b, e) in sweep])

    # 4. seed robustness of headline cascade -----------------------------------
    hb_by_seed = []
    for s in SEEDS:
        for d, p in zip(rows, probas[s]):
            d["_ps"] = p
        m = cascade(rows, lambda d: d["_ps"] >= 0.30)
        hb_by_seed.append(m["harmbench"])
    print(f"Learned cascade HarmBench across seeds: "
          f"{np.mean(hb_by_seed)*100:.1f}% +- {np.std(hb_by_seed)*100:.2f}pp\n")

    # 5. per-category fragile breakdown ----------------------------------------
    fragile = [d for d in rows if (not d["refused_NF4"]) and d["refused_INT8"]]
    caught = [d for d in fragile if d["p_harmful"] >= 0.30]
    print(f"Fragile prompts (NF4 complies, INT8 refuses): {len(fragile)}; "
          f"learned gate recovers {len(caught)}")
    by_cat = defaultdict(lambda: [0, 0])
    for d in fragile:
        by_cat[d["category"]][1] += 1
        if d["p_harmful"] >= 0.30:
            by_cat[d["category"]][0] += 1
    write_csv(os.path.join(TBL, "exp02.out05.fragile_by_category.csv"),
              ["category", "recovered", "total"],
              [[c, v[0], v[1]] for c, v in sorted(by_cat.items(),
               key=lambda kv: -kv[1][1])])

    # 6. traffic-mix stress test ----------------------------------------------
    # The benchmark is adversarial-heavy (500/550 harmful). Real traffic is
    # benign-heavy. Escalation cost depends on the mix, so we measure the gate's
    # escalation rate SEPARATELY on harmful vs benign prompts, then project the
    # effective cost (avg bits) and false-refusal onto realistic mixes.
    def fires(d, t=0.30):
        return (not d["refused_NF4"]) and d["p_harmful"] >= t
    harm = [d for d in rows if d["harmful"] == 1]
    ben = [d for d in rows if d["harmful"] == 0]
    esc_h = sum(fires(d) for d in harm) / len(harm)
    esc_b = sum(fires(d) for d in ben) / len(ben)
    # benign false-refusal contribution: a benign prompt is refused iff NF4
    # refused it, or it was escalated and INT8 refused it.
    def ben_fr(d):
        return d["refused_NF4"] or (fires(d) and d["refused_INT8"])
    fr_b = sum(ben_fr(d) for d in ben) / len(ben)
    print(f"\nGate escalation: harmful={esc_h*100:.1f}%  benign={esc_b*100:.1f}%")
    mixes = [0.50, 0.20, 0.10, 0.05, 0.01]  # fraction harmful in real traffic
    mix_rows = []
    for ph in mixes:
        pesc = ph * esc_h + (1 - ph) * esc_b
        avg_bits = R.BITS["NF4"] + pesc * R.BITS["INT8"]
        mix_rows.append([f"{ph:.2f}", f"{pesc:.4f}", f"{avg_bits:.3f}", f"{fr_b:.4f}"])
    write_csv(os.path.join(TBL, "exp02.out04.traffic_mix.csv"),
              ["frac_harmful", "pct_escalated", "avg_bits", "benign_false_refusal"],
              mix_rows)
    print("Projected onto benign-heavy traffic (frac_harmful -> avg_bits):")
    for ph, r in zip(mixes, mix_rows):
        print(f"  {ph*100:4.0f}% harmful -> escalate {float(r[1])*100:4.1f}% "
              f"-> {float(r[2]):.2f} avg bits")

    make_figures(rows, policies, sweep, y, mean_p)
    print(f"\nWrote tables + figures to {os.path.normpath(FIG)}")


# ---- figures -----------------------------------------------------------------
def make_figures(rows, policies, sweep, y, mean_p):
    # Fig 1: Pareto — safety (HarmBench) vs avg bits
    # per-label offsets to avoid collisions where points cluster
    OFF = {
        "cascade: learned (thr 0.30)": (6, 8),
        "cascade: oracle": (8, -14),
        "static INT8": (8, 4),
        "static NF4": (8, -2),
        "cascade: keyword": (6, 6),
        "static FP16": (-4, 8),
    }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, m in policies:
        casc = name.startswith("cascade")
        ax.scatter(m["avg_bits"], m["harmbench"] * 100, s=90,
                   color=(BLUE if casc else GREY),
                   marker=("o" if casc else "s"), zorder=3,
                   edgecolor="black", linewidth=0.5)
        ax.annotate(name.replace("cascade: ", "casc:").replace("static ", ""),
                    (m["avg_bits"], m["harmbench"] * 100),
                    textcoords="offset points", xytext=OFF.get(name, (6, 5)),
                    fontsize=8.5)
    ax.set_xlabel("Average effective bits / weight (lower = cheaper)")
    ax.set_ylabel("HarmBench refusal rate (%)  (higher = safer)")
    ax.set_title("Safety vs. cost: the cascade reaches INT8 safety near NF4 cost")
    ax.grid(alpha=0.25)
    fig.savefig(os.path.join(FIG, "exp02.out06.fig1_pareto.png")); plt.close(fig)

    # Fig 2: threshold sweep
    t = [s[0] for s in sweep]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(t, [s[1] * 100 for s in sweep], "-o", color=BLUE, label="HarmBench refusal (safer)")
    ax1.plot(t, [s[2] * 100 for s in sweep], "-s", color=RED, label="XSTest false-refusal (worse)")
    ax1.set_xlabel("Risk-gate escalation threshold")
    ax1.set_ylabel("Rate (%)")
    ax2 = ax1.twinx(); ax2.spines["top"].set_visible(False)
    ax2.plot(t, [s[3] for s in sweep], "--^", color=GREEN, label="avg bits (cost)")
    ax2.set_ylabel("Average effective bits", color=GREEN)
    ax2.tick_params(axis="y", colors=GREEN)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8.5, loc="center right")
    ax1.set_title("Threshold sweep: lower threshold = more escalation = safer but costlier")
    ax1.grid(alpha=0.25)
    fig.savefig(os.path.join(FIG, "exp02.out07.fig2_threshold_sweep.png")); plt.close(fig)

    # Fig 3: gate ROC
    fpr, tpr, _ = roc_curve(y, mean_p)
    auc = roc_auc_score(y, mean_p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"TF-IDF + logreg (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Risk-gate ROC (out-of-fold)"); ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.savefig(os.path.join(FIG, "exp02.out08.fig3_gate_roc.png")); plt.close(fig)

    # Fig 4: gap-recovery bars
    names = ["static NF4", "cascade:\nkeyword", "cascade:\nlearned", "static INT8\n(oracle target)"]
    vals = [dict(policies)["static NF4"]["harmbench"],
            dict(policies)["cascade: keyword"]["harmbench"],
            dict(policies)["cascade: learned (thr 0.30)"]["harmbench"],
            dict(policies)["static INT8"]["harmbench"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [GREY, GREY, BLUE, GREEN]
    bars = ax.bar(names, [v * 100 for v in vals], color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 0.6, f"{v*100:.1f}%",
                ha="center", fontweight="bold", fontsize=9)
    ax.set_ylabel("HarmBench refusal rate (%)")
    ax.set_ylim(70, 96)
    ax.set_title("The learned cascade closes the NF4 safety gap")
    fig.savefig(os.path.join(FIG, "exp02.out09.fig4_gap_recovery.png")); plt.close(fig)


if __name__ == "__main__":
    main()
