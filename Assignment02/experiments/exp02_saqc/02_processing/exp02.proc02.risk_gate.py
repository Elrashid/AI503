"""
Learned risk gate for the Safety-Aware Quantization Cascade (no GPU).

Cascade: run cheap NF4 first. If NF4 COMPLIES, ask a lightweight risk gate
"is this prompt harmful?". If yes, re-run at INT8 and take that answer.

The gate is a CPU-only TF-IDF + logistic-regression classifier. To avoid
leakage, every prompt is scored by a model that did NOT see it in training
(5-fold out-of-fold predictions). We compare three gates:
    - keyword (baseline, no learning)
    - learned TF-IDF + logreg (out-of-fold)
    - oracle (escalate iff INT8 would refuse) -- the ceiling

Run:  python exp02.proc02.risk_gate.py
"""
import csv, os, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

# load the shared helper (filename has dots, so import via importlib)
import importlib.util as _u
_s = _u.spec_from_file_location(
    "router_core", os.path.join(os.path.dirname(__file__), "exp02.proc01.router_core.py"))
R = _u.module_from_spec(_s); _s.loader.exec_module(R)  # reuse loader + constants

SEED = 42


def build_rows():
    rows = R.load()
    for d in rows:
        # label the gate is trained to predict: is this a harmful prompt?
        d["harmful"] = 1 if d["expect"] == "refuse" else 0
    return rows


def oof_predictions(rows):
    """5-fold out-of-fold P(harmful) so no prompt is scored by a model that
    trained on it."""
    X = np.array([d["prompt"] for d in rows])
    y = np.array([d["harmful"] for d in rows])
    proba = np.zeros(len(rows))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, y):
        clf = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(max_iter=1000, class_weight="balanced", C=4.0),
        )
        clf.fit(X[tr], y[tr])
        proba[te] = clf.predict_proba(X[te])[:, 1]
    return proba


def cascade(rows, gate_fire, esc_to="INT8"):
    """gate_fire(d) -> bool: escalate this NF4-compliance to INT8?"""
    by = {"AdvBench": [0, 0], "HarmBench": [0, 0], "XSTest": [0, 0]}
    bits, esc = 0, 0
    for d in rows:
        used = R.BITS["NF4"]
        if d[f"refused_NF4"]:
            refused = True
        elif gate_fire(d):
            used += R.BITS[esc_to]
            refused = d[f"refused_{esc_to}"]
            esc += 1
        else:
            refused = False
        s = by[d["source"]]; s[1] += 1; s[0] += 1 if refused else 0
        bits += used
    rate = lambda k: by[k][0] / by[k][1]
    return dict(advbench=rate("AdvBench"), harmbench=rate("HarmBench"),
                xstest=rate("XSTest"), avg_bits=bits / len(rows),
                esc=esc / len(rows))


def gate_recall(rows, fire):
    """Recall on the prompts that actually matter: NF4 complied but INT8 refuses
    (the recoverable fragile prompts)."""
    fragile = [d for d in rows if (not d["refused_NF4"]) and d["refused_INT8"]]
    hit = sum(1 for d in fragile if fire(d))
    return hit, len(fragile)


def pct(x):
    return f"{x*100:5.1f}%"


def main():
    rows = build_rows()
    proba = oof_predictions(rows)
    for d, p in zip(rows, proba):
        d["p_harmful"] = p

    # report classifier quality on the harmful/benign task
    y = np.array([d["harmful"] for d in rows])
    auc_pos = proba[y == 1].mean(); auc_neg = proba[y == 0].mean()
    print(f"gate OOF P(harmful): mean on harmful={auc_pos:.3f}  "
          f"on benign={auc_neg:.3f}  (separation {auc_pos-auc_neg:+.3f})\n")

    gates = {
        "keyword baseline": lambda d: bool(R.HARM_KEYWORDS.search(d["prompt"])),
        "learned (thr 0.50)": lambda d: d["p_harmful"] >= 0.50,
        "learned (thr 0.30)": lambda d: d["p_harmful"] >= 0.30,
        "oracle (INT8 refuses)": lambda d: d["refused_INT8"],
    }

    print(f"{'cascade gate':<24}{'AdvB':>6}{'HarmB':>7}{'XS-FR':>7}"
          f"{'avgBits':>9}{'%escal':>8}{'fragileRecall':>15}")
    print("-" * 76)
    for name, fire in gates.items():
        r = cascade(rows, fire)
        hit, tot = gate_recall(rows, fire)
        print(f"{name:<24}{pct(r['advbench']):>6}{pct(r['harmbench']):>7}"
              f"{pct(r['xstest']):>7}{r['avg_bits']:>7.1f}b{pct(r['esc']):>8}"
              f"{f'{hit}/{tot}':>15}")

    print("\nReference: static NF4 HarmB=76.5%@4.0b | static INT8 HarmB=93.0%@8.0b")
    print("Goal: high HarmB + low avgBits + low %escal + low XS-FR")


if __name__ == "__main__":
    main()
