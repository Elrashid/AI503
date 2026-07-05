"""
Figure 5.1 — SAQC cascade control flow (a styled diagram, not an experiment result).

Renders the methodology flowchart as a PNG so it embeds like the other figures and
survives ODT -> PDF export. Same house style (Times New Roman + project colours).

Run:  python exp02.proc04.fig_saqc_flow.py
Output: ../03_output/exp02.out10.fig5_saqc_flow.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

OUT = os.path.join(os.path.dirname(__file__), "..", "03_output")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 11,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
BLUE, GREY, GREEN, INK = "#4472C4", "#A5A5A5", "#548235", "#222222"
LIGHTBLUE, LIGHTGREEN = "#DAE3F3", "#E2EFDA"


def box(ax, x, y, w, h, text, face, edge, bold=False, fs=11):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.4, edgecolor=edge, facecolor=face))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=INK, fontweight="bold" if bold else "normal")


def diamond(ax, x, y, w, h, text, face, edge, fs=10.5):
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
                         closed=True, linewidth=1.4, edgecolor=edge, facecolor=face))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, x1, y1, x2, y2, label=None, lx=0.0, ly=0.0, ha="center"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5,
                                shrinkA=2, shrinkB=2))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label,
                ha=ha, va="center", fontsize=9.5, color=INK,
                style="italic")


def main():
    fig, ax = plt.subplots(figsize=(7.2, 7.8))
    ax.set_xlim(0.3, 9.2); ax.set_ylim(1.8, 11.4); ax.axis("off")

    sx, rx = 3.3, 7.3       # spine x, right-branch x
    y = {"prompt": 10.7, "nf4": 8.9, "d1": 6.95, "gate": 4.85, "int8": 2.7}

    # nodes
    box(ax, sx, y["prompt"], 2.2, 0.7, "Prompt", "white", GREY)
    box(ax, sx, y["nf4"], 2.9, 0.95, "NF4 model (4-bit)\ncheap first pass", LIGHTBLUE, BLUE, bold=True)
    diamond(ax, sx, y["d1"], 3.0, 1.35, "NF4 output\na refusal?", "white", GREY)
    box(ax, sx, y["gate"], 3.0, 0.95, "Risk gate\nflags as risky?", "white", GREY, bold=True)
    box(ax, sx, y["int8"], 2.9, 0.95, "INT8 re-run\nreturn INT8 answer", LIGHTBLUE, BLUE, bold=True)
    box(ax, rx, y["d1"], 2.4, 0.95, "ACCEPT\n(4-bit cost)", LIGHTGREEN, GREEN, bold=True)
    box(ax, rx, y["gate"], 2.4, 0.95, "ACCEPT\n(4-bit cost)", LIGHTGREEN, GREEN, bold=True)

    # edges
    arrow(ax, sx, y["prompt"] - 0.35, sx, y["nf4"] + 0.48)
    arrow(ax, sx, y["nf4"] - 0.48, sx, y["d1"] + 0.68)
    arrow(ax, sx + 1.5, y["d1"], rx - 1.2, y["d1"], "refused", ly=0.28)
    arrow(ax, sx, y["d1"] - 0.68, sx, y["gate"] + 0.48, "complied", lx=-0.95, ha="right")
    arrow(ax, sx + 1.5, y["gate"], rx - 1.2, y["gate"], "no  (low risk)", ly=0.28)
    arrow(ax, sx, y["gate"] - 0.48, sx, y["int8"] + 0.48, "yes  (high risk)", lx=-1.05, ha="right")

    ax.set_title("SAQC cascade: cheap first, precise only when needed",
                 fontsize=12, fontweight="bold", pad=10)
    fig.savefig(os.path.join(OUT, "exp02.out10.fig5_saqc_flow.png"))
    plt.close(fig)
    print("wrote ../03_output/exp02.out10.fig5_saqc_flow.png")


if __name__ == "__main__":
    main()
