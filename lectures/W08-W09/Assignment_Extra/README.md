# AI503 — CNN Image Classification (W08–W09 Extra Assignment)

CNN image-classification "bake-off" on **CIFAR-10**, built in the style of the Week-7
`compare-all-ml-models.ipynb`: one dataset, ~23 models on one leaderboard, every section
mapped to a teaching week + the *Deep Learning: A Comprehensive Guide* PDF page, and a
COVERED / NOT-RUN badge on every assignment task.

**Location:** `AI503_APR26_Machine_Learning/TeachingMaterial/W08-W09/Assignment_Extra/`

## Folder structure
```
Assignment_Extra/
├── README.md                                    ← you are here
├── notebook/                                    ← the model notebook + its exports
│   ├── CNN_Image_Classification_CIFAR10.ipynb   ← THE code deliverable (run on a Colab/Kaggle GPU)
│   ├── CNN_Image_Classification_CIFAR10.html    ← browser-viewable export (all outputs, no Jupyter)
│   ├── CNN_Image_Classification_CIFAR10.pdf     ← 45-page PDF export of the run notebook
│   └── build_notebook.py                        ← regenerates the .ipynb
├── report/                                      ← the written report + its build pipeline
│   ├── REPORT.md                                ← master document
│   ├── appendix.md                              ← numeric matrices + worked calc (from the notebook)
│   ├── draft/   final/                          ← section files (final/ holds the submission ODT + PDF)
│   ├── figures/                                 ← all report figures (+ figures/appendix/ panels)
│   ├── build_report.py                          ← embeds figures, splits REPORT.md → draft/final
│   ├── add_appendix.py                          ← appends Appendix A/B to REPORT.md
│   └── generate_odt.py                          ← builds the BUiD .odt (→ PDF via LibreOffice)
└── records/                                     ← run artefacts (not needed to rebuild)
    ├── run_outputs.md                           ← captured notebook output log
    ├── review_table.md                          ← sentence-by-sentence report review
    ├── cnn_leaderboard_results.csv              ← evaluation results
    └── cnn_report_assets.zip                    ← figure/appendix bundle exported from Colab
```

**Submit:** `report/final/A2_AI503_CNN_Report.pdf` (42-page report) + `notebook/CNN_Image_Classification_CIFAR10.ipynb` (code).

> Rebuild the report: `python report/build_report.py && python report/generate_odt.py`, then LibreOffice `--convert-to pdf`. Regenerate the notebook exports: `python -m nbconvert --to html notebook/CNN_Image_Classification_CIFAR10.ipynb`, then Edge/Chrome `--headless=new --print-to-pdf` on the HTML.

## How to run (Colab or Kaggle)
1. Upload `notebook/CNN_Image_Classification_CIFAR10.ipynb`.
2. Turn on a **GPU**: Colab → *Runtime → Change runtime type → GPU* (A100 if available); Kaggle → *Settings → Accelerator → GPU*.
3. *Runtime → Run all.*

The first run uses `QUICK_MODE = True` (a few epochs) so it finishes in minutes and you can
confirm nothing is broken. Then set `QUICK_MODE = False` for real numbers.

## Speed tiers (flags in the "CONTROL PANEL" cell)
| Flag(s) | Adds | ~A100 time |
|---------|------|:----------:|
| (defaults) `RUN_LADDER/REGULARIZED/MODERN` | required depth ladder + dropout/BN/augment + modern blocks | ~30 min |
| `RUN_TRANSFER = True` | VGG16 / ResNet50 / MobileNetV2 / EfficientNetV2S / ConvNeXtTiny | +~20 min |
| `RUN_FINETUNE = True` | unfreeze + fine-tune ResNet50 | +~20 min |
| `RUN_TUNING = True` | small grid search | +~25 min |
| everything on, `QUICK_MODE=False` | the full A-grade run | ~1.5–2.5 h |

Set `MIXED_PRECISION = True` on A100/H100 for a ~2× speed-up. Raise `IMG_SIZE_TL` (128→224)
and `BATCH` (128→256) for higher transfer-learning accuracy on big GPUs.

## Resumable
Each finished model is checkpointed to `CKPT_DIR/` the moment it completes. If the runtime
disconnects mid-run, just **run the notebook again** — completed models reload from disk instantly
and only the missing ones train. `RESUME = False` forces a clean re-run. QUICK and full runs keep
separate checkpoint folders (a smoke-test never pollutes the real run). **On Colab**, set
`CKPT_DIR = '/content/drive/MyDrive/cnn_ckpts'` (after mounting Drive) so checkpoints survive a
disconnect — the local `/content` disk is wiped when the runtime dies. Checkpoints store each
model's metrics + predictions (~600 KB each), not full weights, so the whole run is only ~15 MB.

## Reuse a trained model (`SAVE_WEIGHTS`)
Set `SAVE_WEIGHTS = True` to also save each model's **full weights** as a `.keras` file. The last
cell then reloads the best saved model and predicts on new images, and you can reload it in any
script with `tf.keras.models.load_model(path, safe_mode=False)`. Heavier (a transfer model is
~90 MB), so it is off by default.

## Report and figures
The report lives in `report/REPORT.md` (master), split into `report/draft/` + `report/final/`
section folders. `python report/build_report.py` embeds the figures and re-splits;
`python report/generate_odt.py` builds the BUiD-formatted `.odt`, then LibreOffice converts it to PDF.
Big/tall charts (F1 bar, metrics heatmap, per-group curves, confusion grid) get their own full page;
the training curves are split into separate accuracy and loss figures.

**Best figures:** after a full run, run the **export cell (Cell 58)** then the **appendix cell (Cell 60)**.
They re-create every figure as a standalone high-resolution PNG (curves pre-split), build the numeric
confusion matrices + the worked metric calculation, save the CSV, and **zip everything to Google Drive**
for one-click download — hand that zip back to rebuild the report with crisp images and appendices.

## Surviving a Colab disconnect (Drive backup/restore)
Checkpoints stay local for speed but are mirrored to Google Drive so a wiped session recovers instantly:
- **Cell 08 — Restore** (near the top): pulls checkpoints + saved `.keras` weights from
  `MyDrive/cnn_backup` back to the local folder *before* training, so finished models reload instead of retraining.
- **Cell 56 — Backup** (near the end): pushes the local checkpoints + weights up to `MyDrive/cnn_backup`.

Typical flow on a fresh session: *Run all* → Cell 08 restores → training skips done models → Cell 56 backs up the new ones.

## Referencing cells
Every cell is tagged with a number + subtitle (e.g. `Cell 20` — *Train the depth ladder*; code cells
show `# ===== Cell 20 — ... =====` as the first line). To request a change, just name the cell —
the banner inside each cell is always authoritative. Numbers are assigned automatically by the
generator, so they re-flow when cells are added or removed (re-run `python build_notebook.py`).

## Assignment coverage
- **T1** dataset · **T2** preprocessing+augmentation · **T3** CNN design · **T4** training settings
- **T5** ≥3 CNNs of increasing depth → done as a 2→3→4→5→6 ladder + full ~23-model leaderboard
- **T6** per-class report + confusion grid + Grad-CAM ("why" classes confuse)
- **T7** every improvement (augmentation, dropout, batch-norm, transfer learning, tuning) — each measured as its own row

The notebook's final cell prints an automatic COVERED / NOT-RUN report based on what actually ran.

## Reference sources used in the notebook
- *Deep Learning: A Comprehensive Guide* (`W05-W06/`), 54 pp — CNNs are **Ch 4, p.17–23**.
- *Ensemble_Learning.pdf* (`W08-W09/`) — voting p.5 / p.13. (Stacking & soft-voting are labelled as standard extras, not from those slides.)
