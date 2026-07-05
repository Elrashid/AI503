"""
Figure 5.2 — Study pipeline: Data -> Preprocessing -> Model -> Evaluation.

A styled workflow diagram (not an experiment result), rendered as a PNG so it
embeds like the other figures and survives ODT -> PDF export. Same house style
(Times New Roman + project colours) as exp02.proc04.fig_saqc_flow.py.

Numbered "cards" with coloured header bands; arrows are labelled with the data
artifact that passes between stages.

Run:  python exp02.proc05.fig_pipeline.py
Output: ../03_output/exp02.out12.fig6_pipeline.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

OUT = os.path.join(os.path.dirname(__file__), "..", "03_output")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 11,
    "figure.dpi": 200, "savefig.bbox": "tight",
})
BLUE, GREY, GREEN, INK = "#4472C4", "#7F7F7F", "#548235", "#222222"
SUB = "#555555"

STAGES = [
    ("Data", GREY, [
        "Composite benchmark",
        "AdvBench · HarmBench · XSTest",
        "500 harmful / 50 benign"]),
    ("Preprocessing", GREEN, [
        "Chat-template wrapping",
        "Fragile labelling (NF4 vs INT8)",
        "TF-IDF features"]),
    ("Model", BLUE, [
        "SAQC cascade",
        "NF4 → risk gate → INT8",
        "5-fold cross-validation"]),
    ("Evaluation", GREY, [
        "Refusal & false-refusal",
        "Average effective bits",
        "Confusion matrix · ROC"]),
]
FLOW = ["550\nprompts", "features\n+ labels", "per-prompt\noutcomes"]


def card(ax, cx, cy, w, h, num, title, bullets, color):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.16",
        linewidth=1.7, edgecolor=color, facecolor="white", zorder=2))
    bandh = 0.66
    ax.add_patch(Rectangle(
        (cx - w / 2 + 0.08, cy + h / 2 - bandh - 0.08), w - 0.16, bandh,
        facecolor=color, edgecolor="none", zorder=3))
    ax.text(cx, cy + h / 2 - 0.08 - bandh / 2, title, ha="center",
            va="center", color="white", fontsize=12.5, fontweight="bold", zorder=4)
    # numbered badge, top-left
    ax.add_patch(Circle((cx - w / 2, cy + h / 2), 0.27,
                        facecolor=INK, edgecolor="white", linewidth=1.3, zorder=5))
    ax.text(cx - w / 2, cy + h / 2, str(num), ha="center", va="center",
            color="white", fontsize=11.5, fontweight="bold", zorder=6)
    # bullets
    y0 = cy + h / 2 - bandh - 0.64
    for i, b in enumerate(bullets):
        ax.text(cx - w / 2 + 0.30, y0 - i * 0.72, "•  " + b, ha="left",
                va="center", fontsize=9, color=INK, zorder=4)


def arrow(ax, x1, x2, y, label):
    ax.add_patch(FancyArrowPatch(
        (x1, y), (x2, y), arrowstyle="-|>", mutation_scale=20,
        linewidth=2.2, color=INK, zorder=1))
    ax.text((x1 + x2) / 2, y + 0.72, label, ha="center", va="center",
            fontsize=8.5, style="italic", color=SUB, zorder=4,
            linespacing=0.95)


def main():
    fig, ax = plt.subplots(figsize=(12.5, 3.9))
    ax.set_xlim(0, 24); ax.set_ylim(0, 7); ax.axis("off")

    w, h, cy = 4.7, 4.3, 3.3
    xs = [2.7, 8.9, 15.1, 21.3]
    for i, (cx, (title, color, bullets)) in enumerate(zip(xs, STAGES), start=1):
        card(ax, cx, cy, w, h, i, title, bullets, color)
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + w / 2 + 0.06, xs[i + 1] - w / 2 - 0.06, cy, FLOW[i])

    ax.text(24 / 2, 6.55, "Study pipeline: from data to evaluation",
            ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
    fig.savefig(os.path.join(OUT, "exp02.out12.fig6_pipeline.png"))
    plt.close(fig)
    print("wrote ../03_output/exp02.out12.fig6_pipeline.png")


if __name__ == "__main__":
    main()
