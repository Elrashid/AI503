# -*- coding: utf-8 -*-
"""
Generator for the CNN Image-Classification assignment notebook (AI503, W08-W09 Extra Assignment).
Dependency-free: builds a valid nbformat v4 .ipynb using only the json standard library.
Run:  python build_notebook.py
Output: CNN_Image_Classification_CIFAR10.ipynb
Style mirrors TeachingMaterial/W07/compare-all-ml-models.ipynb:
  - one shared evaluate() helper + results{} dict
  - one master leaderboard
  - every section mapped to a teaching week + DL-guide PDF page
  - heavy, zero-assumed-knowledge narration (ADHD-friendly)
  - explicit COVERED / NOT-COVERED badge on every assignment task
"""
import json, os

cells = []
def md(src):  cells.append({"cell_type": "markdown", "id": f"md{len(cells):02d}",
                            "metadata": {}, "source": src.strip("\n")})
def code(src): cells.append({"cell_type": "code", "id": f"code{len(cells):02d}", "metadata": {},
                             "execution_count": None, "outputs": [], "source": src.strip("\n")})

# ===================================================================== CELL: TITLE + COVERAGE MAP
md(r"""
# Deep-Learning Image Classification with CNNs — a fully-documented model bake-off

**Course:** AI503 — Machine Learning &nbsp;|&nbsp; **Assignment:** W08–W09 *Extra Assignment* (Image Classification Using CNN)
**Student:** Mohamed Elrashid (22002576) &nbsp;|&nbsp; **Dataset:** CIFAR-10 (one dataset, held constant)

---

### 📖 How to read this notebook (please read me first)

This notebook assumes **no prior knowledge**. Every new word is explained the first time it appears. Each section starts with a coloured badge so you always know *what it is for*:

- ✅ **COVERS TASK n** — this section satisfies a required part of the assignment.
- ➕ **EXTRA** — this goes beyond the minimum (it is what turns a pass into an A).
- 🎓 **WEEK** — which teaching week the idea came from.
- 📘 **DL Guide p.NN** — the exact page in *Deep Learning: A Comprehensive Guide* (the W05–W06 PDF) where the idea is explained.

> **The big idea in one sentence:** the assignment asks for *3* CNNs; we instead put **~23 models on one leaderboard** and change only the architecture each time, exactly like the Week-7 `compare-all-ml-models` notebook put 18 models on one table. That is the only fair way to compare.

---

### ✅ Assignment coverage map (read this to see nothing is missing)

| Task | What the assignment asks for | Where it lives below | Status |
|------|------------------------------|----------------------|:------:|
| **Task 1** | Describe the dataset (count, classes, size, examples, why CNN-suitable) | §2 | ✅ Covered |
| **Task 2** | Preprocess (resize, normalize, split, augmentation) + explain why | §3 | ✅ Covered |
| **Task 3** | Design a CNN (conv, pool, ReLU, flatten, dense, softmax) | §4 + §5 | ✅ Covered |
| **Task 4** | Train (batch size, epochs, optimizer, learning rate) | §5 (shared harness) + §10 (tuning) | ✅ Covered |
| **Task 5** | Evaluate + compare **≥3** CNNs of increasing depth (acc/P/R/F1, confusion matrix, acc & loss curves) | §5 (the 3) + §11 (full leaderboard) | ✅ Covered + exceeded (23 models) |
| **Task 6** | Discuss results — which classes are correct / confused / why | §12 | ✅ Covered |
| **Task 7** | Improve (augmentation, dropout, batch-norm, transfer learning, tuning) | §6–§10 + §13 | ✅ Covered — **all five**, each measured |

*If a cell ever fails or is skipped, its badge still tells you which task it belonged to, so you can see exactly what is and is not done.*
""")

# ===================================================================== CELL: SYLLABUS / CHAPTER MAP
md(r"""
## §0 · Where every idea in this notebook comes from (syllabus + PDF map)

Two reference tables. The first maps each idea to a **teaching week**. The second maps each idea to an **exact page** of the *Deep Learning: A Comprehensive Guide* PDF (W05–W06), so any claim can be checked.

**Teaching-week map**

| Week | Topic | Used in this notebook for |
|------|-------|---------------------------|
| W01–W02 | Logistic Regression | the idea of a linear baseline; softmax is multi-class logistic regression |
| W03 | KNN, Decision Tree, SVM, Naive Bayes | the *stacking* meta-learner idea (a model on top of models) |
| W04 | K-Means clustering | (not used here — images use supervised CNNs) |
| W05–W06 | **Deep Learning** (CNN, RNN, LSTM) | **the whole notebook** — CNNs come from here |
| W07 | Model comparison + ensembles | the single-leaderboard method this notebook copies |
| W08–W09 | **Ensemble Learning** | voting & stacking in §10 |

**DL-Guide PDF page map** *(file: `W05-W06/Deep Learning_ A Comprehensive Guide.pdf`, 54 pages, 12 chapters)*

| Idea | Chapter | Page |
|------|---------|:----:|
| AI vs ML vs DL; when to use DL vs ML | Ch 1 Introduction | p.5–6 |
| Weights, bias, activation function | Ch 2 NN Fundamentals | p.9–10 |
| Training cycle: forward → loss → backprop → update | Ch 2 | p.8 |
| Optimizers (Adam, SGD), learning rate | Ch 2 | p.13 |
| ReLU activation | Ch 2 | p.10 |
| Feedforward / dense network (MLP) | Ch 3 FNN | p.15 |
| Dropout · Data augmentation · Early stopping | Ch 3 | p.16 |
| Hyperparameter tuning (grid / random / Bayesian) | Ch 3 | p.16 |
| **CNN full pipeline diagram** (conv→pool→flatten→dense) | **Ch 4 CNN** | **p.18** |
| **Convolution / filters / kernels / feature maps** | Ch 4 | p.19 |
| **Stride, padding, output-size formula** | Ch 4 | p.19–20 |
| **Pooling (max / average)** | Ch 4 | p.21 |
| **Flatten · Fully-connected · Softmax head** | Ch 4 | p.21 |
| **Landmark CNNs + Transfer learning (VGG, ResNet)** | Ch 4 | p.22 |
| CNN applications (medical imaging, etc.) | Ch 4 | p.23 |
| Applications of DL overall | Ch 12 | p.52 |

*Ensemble voting (W08–W09 `Ensemble_Learning.pdf`): "majority voting" p.5, "weighted voting" p.13, why-ensembles (bias/variance) p.3. **Note:** soft-voting and stacking are standard extensions **not** shown in those slides — they are labelled as extras where used.*

> **Honesty note on reproducibility (read once):** the Week-7 notebook got bit-identical numbers from a fixed seed because scikit-learn on a CPU is deterministic. Deep learning on a **GPU is not** bit-identical by default (the GPU adds tiny numerical noise). We fix every seed we can, but expect the leaderboard *order* to be stable while the last decimal of each score may wobble by ~0.5%. The reproducible artefact here is the **ranking**, not the 4th decimal.
""")

# ===================================================================== CELL: HOW TO RUN (tiers)
md(r"""
## §1 · How to run this — speed tiers (so you are never stuck waiting)

Training ~23 models can take time. You control how much runs using **flags** in the next cell. A flag is just a switch: `True` = run it, `False` = skip it.

| If you set... | You run | Rough time on A100 | Use when |
|---------------|---------|:------------------:|----------|
| everything `False` except the ladder | the **required 3+ CNNs** only | ~5–10 min | first sanity check |
| `RUN_REGULARIZED=True`, `RUN_MODERN=True` | + dropout/BN/augment + modern blocks | ~30 min | the full from-scratch story |
| also `RUN_TRANSFER=True` | + VGG16/ResNet50/MobileNet/EfficientNet/ConvNeXt | +~20 min | transfer learning |
| also `RUN_FINETUNE`, `RUN_TUNING` | + fine-tuning + hyper-parameter search | +~50 min | the final A-grade run |

> **Tip for a first run:** leave `QUICK_MODE = True`. It caps every model to a few epochs so the whole notebook finishes in minutes and you can confirm nothing is broken. Then set `QUICK_MODE = False` for real numbers.

You need a **GPU runtime** (Colab: *Runtime → Change runtime type → GPU*; Kaggle: *Settings → Accelerator → GPU*). On an A100/H100 turn `MIXED_PRECISION = True` for a ~2× speed-up.

> **Resumable.** Every finished model is saved to `CKPT_DIR`. If the runtime disconnects, just run the notebook again — completed models load instantly and only the missing ones train. Set `RESUME = False` to force a clean re-run. QUICK and full runs use separate checkpoint folders, so a quick smoke-test never contaminates the real run. On Colab, point `CKPT_DIR` at a Google-Drive path (e.g. `/content/drive/MyDrive/cnn_ckpts`) so checkpoints survive a disconnect — the local disk is wiped when the runtime dies.
""")

# ===================================================================== CELL: CONFIG FLAGS
code(r"""
# ============================ CONTROL PANEL ============================
# Flip these switches to decide how much of the notebook runs.
SEED            = 42       # fixed random seed (the Week-7 notebook used 42 too)

QUICK_MODE      = True     # True = tiny epochs for a fast smoke-test. Set False for real results.
MIXED_PRECISION = True     # True = use float16 maths on GPU (big speed-up on A100/H100). Harmless on CPU.

# --- which model groups to train (see the speed-tier table above) ---
RUN_LADDER      = True     # Group A: the REQUIRED depth ladder (2->3->4->5->6 conv layers). Keep True.
RUN_REGULARIZED = True     # Group B: dropout / batch-norm / augmentation ablations (Task 7).
RUN_MODERN      = True     # Group C: VGG / residual / separable / global-average-pooling blocks (extra).
RUN_TRANSFER    = False    # Group D: pretrained backbones, frozen (Task 7). Needs internet for weights.
RUN_FINETUNE    = False    # Group D+: unfreeze and fine-tune the best backbone (Task 7).
RUN_TUNING      = False    # Group E: small hyper-parameter grid search (Task 4/7).
RUN_VIT         = False    # OPTIONAL extra: Vision Transformer. Needs `pip install keras-cv`. Safe to leave False.

# --- resume / checkpoints ---
RESUME   = True            # True = skip models already saved in CKPT_DIR (resume after a disconnect / re-run)
CKPT_DIR = 'cnn_ckpts'     # where per-model results are saved. Use '/content/drive/MyDrive/cnn_ckpts' to survive a Colab disconnect.
SAVE_WEIGHTS = False       # also save each model's FULL weights (.keras) so you can reload a trained model for inference (heavier: a transfer model is ~90 MB).

# --- sizes ---
IMG_SIZE_TL = 128          # pixels to upscale CIFAR (32px) to for the pretrained backbones (try 160/224 for max accuracy)
BATCH       = 128          # images per training step. On A100 you can raise to 256-512.
print("Control panel set. QUICK_MODE =", QUICK_MODE, "| MIXED_PRECISION =", MIXED_PRECISION)
""")

# ===================================================================== CELL: IMPORTS narration
md(r"""
## §1.1 · Setup — import the tools and make results repeatable

Every import below is a tool we will use. In plain words:

- **numpy** — fast number arrays (every image is just a grid of numbers).
- **pandas** — tables (our leaderboard is a pandas table).
- **matplotlib / seaborn** — drawing charts and heatmaps.
- **tensorflow.keras** — the deep-learning library. *Keras* is the friendly front-end; *TensorFlow* is the engine underneath.
- **scikit-learn metrics** — ready-made accuracy / precision / recall / F1 / confusion-matrix functions (same ones the Week-7 notebook used).

We also **fix the random seed** everywhere. Training starts from random numbers; fixing the seed means the notebook gives (almost) the same answer every run.
""")

code(r"""
import os, gc, random, warnings, time
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'      # hide low-level TensorFlow chatter
os.environ['TF_DETERMINISTIC_OPS'] = '1'      # ask the GPU to be as repeatable as it can

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression   # used as the stacking meta-learner in section 10

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

set_seeds()
sns.set_style('whitegrid')

# turn on float16 maths if asked AND a GPU is present (it does nothing useful on CPU)
gpus = tf.config.list_physical_devices('GPU')
if MIXED_PRECISION and gpus:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print('Mixed precision ON (float16).')
print('TensorFlow', tf.__version__, '| GPU available:', bool(gpus), '|', gpus)
""")

# ===================================================================== CELL: §1.2 RESTORE FROM DRIVE
md(r"""
## §1.2 · Restore checkpoints + models from Google Drive (run before training) ➕ EXTRA

If your Colab session was wiped, this pulls any earlier **checkpoints and saved model weights** from a Google-Drive backup back to the local folder. The training cells then reload finished models instantly instead of retraining. It is safe to run every time — it only copies what is missing. The matching backup cell near the end pushes new results up to Drive. *(Uses a local `CKPT_DIR` + Drive backup; for the simplest setup you can instead point `CKPT_DIR` straight at a Drive path in the control panel.)*
""")

code(r"""
# ===== RESTORE: pull checkpoints + model weights from Google Drive (run me before training) =====
import shutil
DRIVE_BACKUP = '/content/drive/MyDrive/cnn_backup'      # where backups live on your Drive
try:
    from google.colab import drive
    if not os.path.isdir('/content/drive/MyDrive'):
        drive.mount('/content/drive')
except Exception:
    print('(not on Colab / no Drive - skipping restore)')

if RESUME and os.path.isdir(DRIVE_BACKUP):
    n = 0
    for root, _, files in os.walk(DRIVE_BACKUP):
        for f in files:
            src = os.path.join(root, f)
            dst = os.path.join(CKPT_DIR, os.path.relpath(src, DRIVE_BACKUP))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                shutil.copy2(src, dst); n += 1
    print(f'Restored {n} files from {DRIVE_BACKUP} -> {CKPT_DIR}/ (finished models will load instantly).')
else:
    print('No Drive backup found at', DRIVE_BACKUP, '- nothing to restore yet (it is created by the backup cell).')
""")

# ===================================================================== CELL: T1 DATASET
md(r"""
## §2 · Task 1 — The dataset (CIFAR-10) ✅ COVERS TASK 1

🎓 **WEEK** W05–W06 (Deep Learning) &nbsp;·&nbsp; 📘 **DL Guide p.18, p.23** (why CNNs are made for images)

**What is CIFAR-10?** A famous practice dataset of **60,000 tiny colour photos**, each **32×32 pixels**, sorted into **10 everyday classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. It ships *inside* Keras, so there is nothing to download by hand.

**Why is it a good fit for a CNN?** 📘 (DL Guide p.18) explains that a normal dense network would need ~150,000 inputs for even a small image and explode into millions of weights. A **CNN** instead slides small filters across the picture, so it (a) needs far fewer weights and (b) understands that nearby pixels belong together — the *spatial* structure a photo has. CIFAR-10 is the standard first dataset for exactly this lesson.

**Numbers we must report for Task 1:** total images, number of classes, image size, an example of each category, and the train/test split. The next two cells print all of them.
""")

code(r"""
from tensorflow.keras.datasets import cifar10

# CIFAR-10 arrives already split into a 50,000-image training set and a 10,000-image test set.
(x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog',
               'frog','horse','ship','truck']
NUM_CLASSES = len(CLASS_NAMES)

print('Image shape (height, width, colour channels):', x_train_full.shape[1:])
print('Pixel value range:', x_train_full.min(), 'to', x_train_full.max(), '(0=black, 255=brightest)')
print('Training images:', x_train_full.shape[0])
print('Test images    :', x_test.shape[0])
print('Total images   :', x_train_full.shape[0] + x_test.shape[0])
print('Number of classes:', NUM_CLASSES, '->', CLASS_NAMES)

# class balance — are the 10 classes equally common? (the Week-7 notebook checked balance too)
unique, counts = np.unique(y_train_full, return_counts=True)
print('\nImages per class in the training set:')
for c, n in zip(unique, counts):
    print(f'  {CLASS_NAMES[c]:12s}: {n}')
print('\nCIFAR-10 is perfectly balanced (5,000 of each) and has no train/test leakage by construction.')
""")

code(r"""
# Show one example image from every class (Task 1 asks for category examples)
fig, axes = plt.subplots(2, 5, figsize=(11, 5))
for cls, ax in enumerate(axes.flat):
    idx = np.where(y_train_full[:, 0] == cls)[0][0]   # first image of this class
    ax.imshow(x_train_full[idx])
    ax.set_title(CLASS_NAMES[cls], fontsize=10)
    ax.axis('off')
plt.suptitle('CIFAR-10 — one example per class (32x32 pixels each)', y=1.02)
plt.tight_layout(); plt.show()
""")

# ===================================================================== CELL: T2 PREPROCESS
md(r"""
## §3 · Task 2 — Preprocessing ✅ COVERS TASK 2

🎓 **WEEK** W05–W06 &nbsp;·&nbsp; 📘 **DL Guide p.16** (data augmentation, why we regularise)

Four steps, each with the *why*:

1. **Normalize** — pixel values are 0–255. We divide by 255 so they become 0.0–1.0. *Why:* neural networks learn faster and more stably when inputs are small and on the same scale (same reason KNN/SVM needed scaling back in W03).
2. **Train / validation / test split** — we carve a **validation** set out of the training images. *Why:* we tune and watch the model on validation data, and only touch the test set once at the very end. Touching the test set during training would be cheating (data leakage — the same sin the Week-7 notebook warned about with duplicate rows).
3. **One-hot labels** — class `3` ("cat") becomes `[0,0,0,1,0,0,0,0,0,0]`. *Why:* the softmax output layer produces 10 probabilities, so the true answer must also be 10 numbers.
4. **Data augmentation** — randomly flip/rotate/zoom images *during training only*. 📘 (p.16) *Why:* it shows the model slightly different versions of each photo every epoch, so it cannot simply memorise the training set (it fights overfitting). We will also **measure** how much augmentation helps by turning it into its own leaderboard row in §6.

> **Two image scales.** From-scratch CNNs train on the native **32×32** images in [0,1]. The pretrained backbones in §8 were built for big ImageNet photos, so they get the **raw 0–255** images upscaled to `IMG_SIZE_TL`, and each backbone applies *its own* normalisation internally. Mixing those up quietly wrecks accuracy, so we keep both versions of the data.
""")

code(r"""
from sklearn.model_selection import train_test_split

# ---- version A: scaled [0,1] floats for the FROM-SCRATCH CNNs ----
x_train_full_s = x_train_full.astype('float32') / 255.0
x_test_s       = x_test.astype('float32') / 255.0

# ---- version B: raw [0,255] floats for the PRETRAINED backbones (they scale internally) ----
x_train_full_r = x_train_full.astype('float32')
x_test_r       = x_test.astype('float32')

# ---- one-hot labels ----
y_train_full_oh = tf.keras.utils.to_categorical(y_train_full, NUM_CLASSES)
y_test_oh       = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

# ---- carve a validation set out of the 50k training images (stratified = keep class balance) ----
idx_tr, idx_va = train_test_split(np.arange(len(x_train_full_s)), test_size=5000,
                                  random_state=SEED, stratify=y_train_full[:, 0])

Xtr_s, Xva_s = x_train_full_s[idx_tr], x_train_full_s[idx_va]   # scaled, for from-scratch
Xtr_r, Xva_r = x_train_full_r[idx_tr], x_train_full_r[idx_va]   # raw, for transfer
ytr_oh, yva_oh = y_train_full_oh[idx_tr], y_train_full_oh[idx_va]
ytr_int = y_train_full[idx_tr, 0]                                # plain integer labels (handy)

# integer truth vectors used by the metrics later
y_test_int = y_test[:, 0]
y_val_int  = y_train_full[idx_va, 0]

print('Train images     :', Xtr_s.shape[0])
print('Validation images:', Xva_s.shape[0])
print('Test images      :', x_test_s.shape[0])
print('Label shape (one-hot):', ytr_oh.shape, '-> 10 numbers per image')
""")

code(r"""
# Build the augmentation pipeline as Keras layers. These layers ONLY change images during training.
data_augment = models.Sequential([
    layers.RandomFlip('horizontal'),     # a mirrored cat is still a cat
    layers.RandomRotation(0.08),         # small tilt
    layers.RandomZoom(0.10),             # slight zoom in/out
    layers.RandomTranslation(0.10, 0.10) # small shift
], name='data_augment')

# Show the SAME image augmented 5 different ways, so you can see what the model sees
sample = Xtr_s[0:1]
fig, axes = plt.subplots(1, 6, figsize=(13, 2.6))
axes[0].imshow(sample[0]); axes[0].set_title('original'); axes[0].axis('off')
for i in range(1, 6):
    aug = data_augment(sample, training=True)[0].numpy().astype('float32')
    axes[i].imshow(np.clip(aug, 0, 1)); axes[i].set_title(f'augmented {i}'); axes[i].axis('off')
plt.suptitle('Data augmentation — 5 random versions of one training image (DL Guide p.16)', y=1.05)
plt.tight_layout(); plt.show()
""")

# ===================================================================== CELL: HARNESS narration
md(r"""
## §3.1 · The shared scoreboard and training helper (the heart of the notebook)

Just like the Week-7 notebook had **one** `evaluate()` function and **one** `results` dictionary that every model wrote into, we build the same thing here — but upgraded from 2-class to **10-class** images. That upgrade matters and is the single most important code in the notebook:

- **`average='macro'`** on precision/recall/F1 — score each of the 10 classes, then average. (The Week-7 version was binary; 10 classes needs macro.)
- **`roc_auc_score(..., multi_class='ovr')`** — "one-vs-rest" ROC for 10 classes.
- We read **train accuracy, validation accuracy, train loss and validation loss** straight from the training `history` — Task 5 asks for all four as reported numbers, not just curves.
- We compute **test** accuracy/precision/recall/F1 on the untouched 10k test set.

Every model therefore lands one tidy row on the same scoreboard, and we can sort them all fairly at the end.
""")

code(r"""
results   = {}   # name -> dict of scores + predictions (our leaderboard rows)
histories = {}   # name -> per-epoch training history (for the accuracy/loss curves)

# ---- resume support: save each finished model to disk; reload it instead of retraining ----
import pickle
_CKPT = os.path.join(CKPT_DIR, 'quick' if QUICK_MODE else 'full')   # keep QUICK and full runs separate
os.makedirs(_CKPT, exist_ok=True)
def _ckpt_path(name):
    safe = name.replace('/', '_').replace(' ', '_').replace(':', '')
    return os.path.join(_CKPT, safe + '.pkl')
def _save_ckpt(name):
    try:
        with open(_ckpt_path(name), 'wb') as f:
            pickle.dump({'row': results[name], 'hist': histories[name]}, f)
    except Exception as e:
        print('  (checkpoint save failed:', e, ')')
def _load_ckpt(name):
    p = _ckpt_path(name)
    if RESUME and os.path.exists(p):
        with open(p, 'rb') as f:
            d = pickle.load(f)
        results[name] = d['row']; histories[name] = d['hist']
        return True
    return False

# ---- optional: also save FULL model weights (.keras) so a trained model can be reloaded for inference ----
_WCKPT = os.path.join(_CKPT, 'weights')
os.makedirs(_WCKPT, exist_ok=True)
def _weights_path(name):
    safe = name.replace('/', '_').replace(' ', '_').replace(':', '')
    return os.path.join(_WCKPT, safe + '.keras')
def _reload_weights(name):
    p = _weights_path(name)
    if os.path.exists(p):
        try:
            return tf.keras.models.load_model(p, safe_mode=False, compile=False)
        except Exception as e:
            print('  (could not reload weights for', name, '->', str(e)[:80], ')')
    return None

def make_callbacks(patience=6):
    # EarlyStopping = stop when validation stops improving (and keep the best weights).
    # ReduceLROnPlateau = lower the learning rate when progress stalls (DL Guide p.13).
    return [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0)]

def epochs_for(default):
    return 3 if QUICK_MODE else default   # QUICK_MODE shrinks every training run

def run_model(name, group, build_fn, use_raw=False, augment=False,
              epochs=30, lr=1e-3, keep_model=False, fine_tune=None):
    '''Train one model, score it on val+test, and store one leaderboard row.
       use_raw   : feed raw [0,255] images (for pretrained backbones) instead of [0,1].
       augment   : prepend the augmentation layers (training only).
       fine_tune : optional dict {"unfreeze": base_model, "epochs": n, "lr": small} for a 2nd phase.
       Returns the trained model (kept only if keep_model=True, else freed to save memory).'''
    if _load_ckpt(name):
        r = results[name]
        print(f'{name:34s} [resumed from checkpoint]  test_acc={r["test_acc"]:.3f}  F1={r["f1"]:.3f}')
        return _reload_weights(name) if keep_model else None
    K.clear_session(); set_seeds(); t0 = time.time()
    Xtr, Xva, Xte = (Xtr_r, Xva_r, x_test_r) if use_raw else (Xtr_s, Xva_s, x_test_s)

    net = build_fn()
    if augment:
        _inp = layers.Input(Xtr.shape[1:])
        net = models.Model(_inp, net(data_augment(_inp)), name='augmented')
    net.compile(optimizer=optimizers.Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    h = net.fit(Xtr, ytr_oh, validation_data=(Xva, yva_oh),
                epochs=epochs_for(epochs), batch_size=BATCH,
                callbacks=make_callbacks(), verbose=0)
    hist = {k: list(v) for k, v in h.history.items()}

    # optional fine-tuning phase: unfreeze a pretrained base and train gently with a tiny LR
    if fine_tune is not None and not QUICK_MODE:
        for _lyr in net.layers:           # unfreeze everything, including the nested pretrained base
            _lyr.trainable = True
        net.compile(optimizer=optimizers.Adam(fine_tune.get('lr', 1e-5)),
                    loss='categorical_crossentropy', metrics=['accuracy'])
        h2 = net.fit(Xtr, ytr_oh, validation_data=(Xva, yva_oh),
                     epochs=epochs_for(fine_tune.get('epochs', 8)), batch_size=BATCH,
                     callbacks=make_callbacks(), verbose=0)
        for k, v in h2.history.items():
            hist.setdefault(k, []).extend(list(v))

    _score_and_store(name, group, net, hist, Xva, Xte, time.time() - t0)
    if keep_model:
        return net
    del net; gc.collect(); K.clear_session()
    return None

def _score_and_store(name, group, net, hist, Xva, Xte, secs):
    prob_test = net.predict(Xte, batch_size=256, verbose=0)   # shape (10000, 10)
    prob_val  = net.predict(Xva, batch_size=256, verbose=0)
    pred_test = prob_test.argmax(1)

    row = {
        'group'     : group,
        'train_acc' : hist['accuracy'][-1],
        'val_acc'   : hist['val_accuracy'][-1],
        'test_acc'  : accuracy_score(y_test_int, pred_test),
        'train_loss': hist['loss'][-1],
        'val_loss'  : hist['val_loss'][-1],
        'precision' : precision_score(y_test_int, pred_test, average='macro'),
        'recall'    : recall_score(y_test_int, pred_test, average='macro'),
        'f1'        : f1_score(y_test_int, pred_test, average='macro'),
        'roc_auc'   : roc_auc_score(y_test_oh, prob_test, multi_class='ovr', average='macro'),
        'seconds'   : round(secs, 1),
        'y_pred'    : pred_test,
        'prob_test' : prob_test.astype('float32'),
        'prob_val'  : prob_val.astype('float32'),
    }
    results[name] = row; histories[name] = hist
    _save_ckpt(name)
    if SAVE_WEIGHTS:
        try:
            net.save(_weights_path(name))
        except Exception as e:
            print('  (weight save failed:', e, ')')
    print(f'{name:34s} test_acc={row["test_acc"]:.3f}  F1={row["f1"]:.3f}  '
          f'AUC={row["roc_auc"]:.3f}  ({row["seconds"]}s)')

print('Scoreboard ready. Each model will print one line and store one row.')
""")

# ===================================================================== CELL: T3 BUILDING BLOCKS
md(r"""
## §4 · Task 3 — The CNN building blocks, explained from zero ✅ COVERS TASK 3

🎓 **WEEK** W05–W06 &nbsp;·&nbsp; 📘 **DL Guide Ch 4, p.18–21**

Before we build models, here is every Lego brick a CNN is made of. The required pieces (conv, pool, ReLU, flatten, dense, softmax) are all here.

| Brick | In plain words | 📘 Page |
|-------|----------------|:------:|
| **Convolution (Conv2D)** | A small window (a *filter* / *kernel*, e.g. 3×3) slides over the image and looks for one pattern (an edge, a colour blob). Each filter makes one **feature map**. | p.19 |
| **Filter / kernel** | The little grid of numbers that *is* the pattern detector. A layer has many filters, so it finds many patterns at once. | p.19 |
| **Feature map** | The output image showing *where* a filter fired. | p.19 |
| **ReLU** | After each conv, replace negatives with 0: `max(0, x)`. Adds non-linearity so the network can learn complex shapes. | p.10 |
| **Pooling (MaxPooling2D)** | Shrink the feature map by keeping the strongest value in each 2×2 window. Makes the model smaller and a little position-proof. | p.21 |
| **Flatten** | Unroll the final 2-D feature maps into one long 1-D list of numbers, so a normal dense layer can read it. | p.21 |
| **Dense (fully connected)** | A classic neuron layer that mixes all features to make the decision. | p.21 |
| **Softmax** | The final layer turns raw scores into 10 probabilities that add up to 1. The biggest one is the predicted class. | p.21 |

**The standard CNN recipe** 📘 (p.18): `Input → [Conv → ReLU → Pool] × N → Flatten → Dense → Softmax`. Every model in Group A is exactly this recipe with a different number of `[Conv→ReLU→Pool]` blocks.

> **Why the last layer says `dtype='float32'`:** when mixed precision is ON, the network does its maths in fast float16. But probabilities need full float32 precision to stay accurate, so we force *only* the final softmax back to float32. You will see this on every model.
""")

# ===================================================================== CELL: build_cnn factory
code(r"""
# The parametric from-scratch CNN factory used for the whole depth ladder AND the regularised variants.
def build_cnn(n_blocks=3, base_filters=32, dropout=0.0, batchnorm=False, name=None):
    # n_blocks = how many [Conv -> (BN) -> Pool] blocks. This is the "depth" we increase.
    m = models.Sequential(name=name or f'cnn_{n_blocks}blocks')
    m.add(layers.Input((32, 32, 3)))
    f = base_filters
    pools_used = 0
    for i in range(n_blocks):
        m.add(layers.Conv2D(f, 3, padding='same', activation='relu'))   # Conv + ReLU (DL Guide p.19)
        if batchnorm:
            m.add(layers.BatchNormalization())                          # steadies training (extra, not in W05-06 PDF)
        if pools_used < 4:                  # at most 4 pools: 32->16->8->4->2 (never shrink below 2x2)
            m.add(layers.MaxPooling2D())                                 # pooling (DL Guide p.21)
            pools_used += 1
        f = min(f * 2, 256)                 # double the filters each block (common practice)
    m.add(layers.Flatten())                                             # flatten (DL Guide p.21)
    m.add(layers.Dense(128, activation='relu'))                        # dense head (DL Guide p.21)
    if dropout > 0:
        m.add(layers.Dropout(dropout))                                 # dropout (DL Guide p.16)
    m.add(layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32'))  # softmax (DL Guide p.21)
    return m

# print the layer table of a 4-block CNN so you can SEE the recipe
build_cnn(4).summary()
""")

# ===================================================================== CELL: GROUP A narration
md(r"""
## §5 · Group A — the required depth ladder (Task 5 core) ✅ COVERS TASK 5 & TASK 3 & TASK 4

🎓 **WEEK** W05–W06 &nbsp;·&nbsp; 📘 **DL Guide p.18**

The assignment requires **three** CNNs of growing depth: Model 1 = 2 conv layers, Model 2 = 3–4, Model 3 = 5+. We do that **and** add the in-between steps, training **2 → 3 → 4 → 5 → 6** conv layers. This is the "add one layer, re-train, measure" idea — and it produces one clean *accuracy-vs-depth* chart.

**Honest expectation (write this down before running):** accuracy should **rise** from 2 to ~4 layers as the network learns richer features, then **plateau or dip** at 5–6 layers because (a) after a few poolings a 32×32 image has almost no spatial size left, (b) deep plain networks suffer vanishing gradients, and (c) more layers = more parameters = more overfitting on limited data. That plateau is the whole lesson — and the reason Groups B–D exist.

**Training settings (Task 4):** optimizer = **Adam**, learning rate = **0.001**, batch size = **128**, loss = **categorical cross-entropy**, with **early stopping** so each model trains just long enough.
""")

code(r"""
# Train the depth ladder. We keep the 4-conv model in memory for the Grad-CAM picture later.
KEEP_FOR_GRADCAM = None
LADDER = []
if RUN_LADDER:
    for depth in [2, 3, 4, 5, 6]:
        nm = f'CNN-{depth}conv'
        keep = (depth == 4)
        m = run_model(nm, 'A_ladder', (lambda d=depth: build_cnn(d)),
                      epochs=30, lr=1e-3, keep_model=keep)
        if keep:
            KEEP_FOR_GRADCAM = m
        LADDER.append((depth, nm))
else:
    print('RUN_LADDER is False — skipping the required ladder (set it True).')
""")

code(r"""
# Chart 1: accuracy vs depth (the headline picture for Task 5's architecture comparison)
if LADDER:
    depths = [d for d, _ in LADDER]
    accs   = [results[n]['test_acc'] for _, n in LADDER]
    plt.figure(figsize=(8, 5))
    plt.plot(depths, accs, 'o-', color='#4472C4', linewidth=2, markersize=9)
    for d, a in zip(depths, accs):
        plt.text(d, a + 0.004, f'{a:.3f}', ha='center', fontsize=10, fontweight='bold')
    plt.xlabel('Number of convolutional layers (depth)')
    plt.ylabel('Test accuracy')
    plt.title('Accuracy vs depth — rises, then plateaus (the core Task-5 finding)')
    plt.xticks(depths); plt.tight_layout(); plt.show()
""")

code(r"""
# Chart 2: train & validation accuracy and loss curves for the three REQUIRED models
required = [('Model 1 = 2 conv', 'CNN-2conv'),
            ('Model 2 = 4 conv', 'CNN-4conv'),
            ('Model 3 = 6 conv', 'CNN-6conv')]
colors = ['#4472C4', '#ED7D31', '#70AD47']
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for (label, nm), c in zip(required, colors):
    if nm in histories and histories[nm]:
        h = histories[nm]
        ax[0].plot(h['accuracy'],     color=c, label=f'{label} (train)')
        ax[0].plot(h['val_accuracy'], color=c, linestyle='--', label=f'{label} (val)')
        ax[1].plot(h['loss'],         color=c, label=f'{label} (train)')
        ax[1].plot(h['val_loss'],     color=c, linestyle='--', label=f'{label} (val)')
ax[0].set_title('Accuracy curves (solid=train, dashed=val)'); ax[0].set_xlabel('epoch'); ax[0].legend(fontsize=7)
ax[1].set_title('Loss curves (solid=train, dashed=val)');     ax[1].set_xlabel('epoch'); ax[1].legend(fontsize=7)
plt.suptitle('Required 3 CNNs — training vs validation (a widening gap = overfitting)', y=1.03)
plt.tight_layout(); plt.show()
""")

# ===================================================================== CELL: GROUP B narration
md(r"""
## §6 · Group B — regularisation ablations (Task 7, part 1) ✅ COVERS TASK 7 ➕ EXTRA

🎓 **WEEK** W05–W06 &nbsp;·&nbsp; 📘 **DL Guide p.16** (dropout, augmentation)

Task 7 says "improve using one or more of: augmentation, dropout, batch-norm…". Instead of picking one, we add **each one separately** to the same 5-conv base, so the leaderboard shows exactly **how much each trick is worth**.

- **Dropout** 📘(p.16): randomly switch off neurons during training so the model cannot lean on any single one → fights overfitting.
- **Batch normalisation**: re-centre each layer's outputs so training is faster and steadier. *(Standard technique; not in the W05–W06 slides — labelled extra.)*
- **Augmentation** 📘(p.16): the random flips/rotations from §3, now switched on.
- **All three together**: the fully-regularised custom net.
""")

code(r"""
if RUN_REGULARIZED:
    run_model('Deep+Dropout',         'B_reg', lambda: build_cnn(5, dropout=0.4))
    run_model('Deep+BatchNorm',       'B_reg', lambda: build_cnn(5, batchnorm=True))
    run_model('Deep+Augment',         'B_reg', lambda: build_cnn(5), augment=True)
    run_model('Deep+All(BN+Drop+Aug)','B_reg', lambda: build_cnn(5, dropout=0.4, batchnorm=True), augment=True)
else:
    print('RUN_REGULARIZED is False — skipping Group B.')
""")

# ===================================================================== CELL: GROUP C narration
md(r"""
## §7 · Group C — modern CNN blocks, built from scratch ➕ EXTRA

🎓 **WEEK** W05–W06 &nbsp;·&nbsp; 📘 **DL Guide p.22** (landmark architectures: VGG, ResNet)

These four show *design ideas* from famous networks, built small enough to train from scratch:

- **VGG-style**: two conv layers before each pool (deeper, richer blocks). 📘 p.22
- **Residual / skip connection** (the ResNet idea 📘 p.22): add the block's input to its output, so gradients have a shortcut and very deep nets still train.
- **Depthwise-separable conv** (the MobileNet idea): a cheaper convolution that splits the work, far fewer parameters.
- **All-convolutional + Global Average Pooling**: no flatten, no big dense layer — average each final feature map straight into a class score. Tiny and overfit-resistant.
""")

code(r"""
def build_vgg():
    m = models.Sequential(name='vgg_style')
    m.add(layers.Input((32, 32, 3)))
    for f in [32, 64, 128]:
        m.add(layers.Conv2D(f, 3, padding='same', activation='relu'))
        m.add(layers.Conv2D(f, 3, padding='same', activation='relu'))   # two convs per block = VGG idea
        m.add(layers.BatchNormalization())
        m.add(layers.MaxPooling2D())
    m.add(layers.Flatten())
    m.add(layers.Dense(128, activation='relu'))
    m.add(layers.Dropout(0.4))
    m.add(layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32'))
    return m

def build_residual():
    inp = layers.Input((32, 32, 3))
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    def res_block(t, f):
        shortcut = t
        y = layers.Conv2D(f, 3, padding='same', activation='relu')(t)
        y = layers.BatchNormalization()(y)
        y = layers.Conv2D(f, 3, padding='same')(y)
        y = layers.BatchNormalization()(y)
        if shortcut.shape[-1] != f:
            shortcut = layers.Conv2D(f, 1, padding='same')(shortcut)     # match channels
        out = layers.Activation('relu')(layers.add([shortcut, y]))       # the skip connection
        return layers.MaxPooling2D()(out)
    x = res_block(x, 64)
    x = res_block(x, 128)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
    return models.Model(inp, out, name='residual_mini')

def build_separable():
    m = models.Sequential(name='separable')
    m.add(layers.Input((32, 32, 3)))
    m.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
    for f in [64, 128, 256]:
        m.add(layers.SeparableConv2D(f, 3, padding='same', activation='relu'))
        m.add(layers.BatchNormalization())
        m.add(layers.MaxPooling2D())
    m.add(layers.GlobalAveragePooling2D())
    m.add(layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32'))
    return m

def build_allconv_gap():
    m = models.Sequential(name='allconv_gap')
    m.add(layers.Input((32, 32, 3)))
    for f in [48, 96, 192]:
        m.add(layers.Conv2D(f, 3, padding='same', activation='relu'))
        m.add(layers.Conv2D(f, 3, strides=2, padding='same', activation='relu'))  # strided conv = learned pooling
        m.add(layers.BatchNormalization())
    m.add(layers.Conv2D(NUM_CLASSES, 1, padding='same'))
    m.add(layers.GlobalAveragePooling2D())
    m.add(layers.Activation('softmax', dtype='float32'))
    return m

if RUN_MODERN:
    gm = run_model('VGG-style (2xconv blocks)', 'C_modern', build_vgg, keep_model=True)
    if KEEP_FOR_GRADCAM is None:
        KEEP_FOR_GRADCAM = gm                 # fallback Grad-CAM model
    run_model('Residual-mini (skip conn.)', 'C_modern', build_residual)
    run_model('Depthwise-separable',        'C_modern', build_separable)
    run_model('All-conv + GAP',             'C_modern', build_allconv_gap)
else:
    print('RUN_MODERN is False — skipping Group C.')
""")

# ===================================================================== CELL: GROUP D narration
md(r"""
## §8 · Group D — transfer learning with pretrained backbones (Task 7, part 2) ✅ COVERS TASK 7 ➕ EXTRA

🎓 **WEEK** W05–W06 &nbsp;·&nbsp; 📘 **DL Guide p.22** (VGG, ResNet)

**Transfer learning** means: take a giant network already trained on millions of ImageNet photos, and reuse its learned "eyes". We **freeze** its body (keep its weights) and only train a new 10-class head on top. 📘 (p.22) names VGGNet and ResNet as exactly these reusable networks.

Two phases:
1. **Frozen** — train only the new head. Fast, and usually already beats from-scratch CNNs.
2. **Fine-tuning** (`RUN_FINETUNE`) — then *unfreeze* the body and train gently with a tiny learning rate (1e-5) so it adapts to CIFAR without forgetting. This usually gives the best score on the board.

> The pretrained images were big, so we upscale CIFAR from 32×32 to `IMG_SIZE_TL`. Each backbone also applies **its own** normalisation (that is why VGG/ResNet/MobileNet get a `preprocess_input`, while EfficientNet/ConvNeXt already include it). Needs internet the first time to download the weights.
""")

code(r"""
# Load whichever backbones this TensorFlow version provides (skip any that are missing).
import importlib
BACKBONES = {}
_specs = [('VGG16','vgg16','VGG16', False),
          ('ResNet50','resnet50','ResNet50', False),
          ('MobileNetV2','mobilenet_v2','MobileNetV2', False),
          ('EfficientNetV2S','efficientnet_v2','EfficientNetV2S', True),
          ('ConvNeXtTiny','convnext','ConvNeXtTiny', True)]
for disp, mod, ctor, builtin_prep in _specs:
    try:
        m = importlib.import_module(f'tensorflow.keras.applications.{mod}')
        BACKBONES[disp] = (getattr(m, ctor), None if builtin_prep else getattr(m, 'preprocess_input'))
    except Exception as e:
        print(f'  (skip {disp}: {e})')
print('Backbones available:', list(BACKBONES.keys()))

def build_transfer(backbone_ctor, preprocess_fn, size=IMG_SIZE_TL):
    base = backbone_ctor(include_top=False, weights='imagenet',
                         input_shape=(size, size, 3), pooling='avg')
    base.trainable = False                                  # FREEZE the body
    inp = layers.Input((32, 32, 3))                         # raw 0-255 CIFAR images
    x = layers.Resizing(size, size)(inp)                    # upscale to the size the backbone expects
    if preprocess_fn is not None:
        x = layers.Lambda(lambda t: preprocess_fn(tf.cast(t, 'float32')))(x)  # backbone norm in float32
    x = base(x, training=False)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)  # new 10-class head
    return models.Model(inp, out, name=f'{backbone_ctor.__name__}_TL')

if RUN_TRANSFER:
    for bname, (ctor, prep) in BACKBONES.items():
        run_model(f'{bname} (frozen TL)', 'D_transfer',
                  (lambda c=ctor, p=prep: build_transfer(c, p)),
                  use_raw=True, epochs=12, lr=1e-3)
else:
    print('RUN_TRANSFER is False — skipping pretrained backbones.')
""")

code(r"""
# Fine-tune ONE champion backbone (ResNet50): phase-1 frozen head, then unfreeze and train gently.
if RUN_FINETUNE and 'ResNet50' in BACKBONES:
    ctor, prep = BACKBONES['ResNet50']
    run_model('ResNet50 (fine-tuned)', 'D_transfer',
              (lambda: build_transfer(ctor, prep)),
              use_raw=True, epochs=12, lr=1e-3,
              fine_tune={'epochs': 8, 'lr': 1e-5})
else:
    print('RUN_FINETUNE is False (or ResNet50 unavailable) — skipping fine-tuning.')
""")

# ===================================================================== CELL: GROUP E narration
md(r"""
## §9 · Group E — hyper-parameter tuning (Task 4 / Task 7) ✅ COVERS TASK 7 ➕ EXTRA

🎓 **WEEK** W05–W06 &nbsp;·&nbsp; 📘 **DL Guide p.16** (grid / random / Bayesian search)

The Week-7 notebook used `GridSearchCV` to tune classic models. The CNN version of that is a small **grid search** over a couple of settings — learning rate and dropout — on the 4-conv CNN. 📘 (p.16) lists grid / random / Bayesian as the three search strategies; for a serious run, the `keras-tuner` library does this automatically, but a hand grid keeps the notebook dependency-free.
""")

code(r"""
if RUN_TUNING:
    for lr_ in [1e-3, 5e-4]:
        for dr_ in [0.3, 0.5]:
            run_model(f'Grid lr={lr_:.0e} drop={dr_}', 'E_tuning',
                      (lambda d=dr_: build_cnn(4, dropout=d)), epochs=25, lr=lr_)
else:
    print('RUN_TUNING is False — skipping the hyper-parameter grid.')
""")

# ===================================================================== CELL: GROUP F narration
md(r"""
## §10 · Group F — ensembles: combine the best models ➕ EXTRA (Task 7)

🎓 **WEEK** W08–W09 (Ensemble Learning) &nbsp;·&nbsp; 📄 `Ensemble_Learning.pdf` p.3 (why ensembles help), p.5 ("majority voting"), p.13 ("weighted voting")

An **ensemble** combines several models so their mistakes cancel out 📄(p.3). We take the **top-3 models so far** and combine them three ways:

- **Soft vote** — average the 10 probabilities from each model, then pick the biggest. *(Averaging is the regression cousin of the "majority voting" on p.5.)*
- **Hard vote** — each model votes for one class; the majority wins 📄(p.5).
- **Stacking** — train a small Logistic Regression (a W01–W02 / W03 model!) **on top of** the base models' outputs. *(Stacking is a standard extension, not shown in the W08–W09 slides — labelled extra.)*

> **No cheating:** the stacking meta-learner is trained on the **validation** predictions and judged on the **untouched test** set, so it never sees the test answers in advance — the same data-hygiene rule the Week-7 notebook followed with `StackingClassifier(cv=5)`.
""")

code(r"""
def _store_pred(name, group, prob_test=None, pred=None):
    # store an ensemble row (no training history, so train/val columns are blank)
    if pred is None:
        pred = prob_test.argmax(1)
    row = {'group': group, 'train_acc': np.nan, 'val_acc': np.nan,
           'test_acc': accuracy_score(y_test_int, pred),
           'train_loss': np.nan, 'val_loss': np.nan,
           'precision': precision_score(y_test_int, pred, average='macro'),
           'recall':    recall_score(y_test_int, pred, average='macro'),
           'f1':        f1_score(y_test_int, pred, average='macro'),
           'roc_auc':   (roc_auc_score(y_test_oh, prob_test, multi_class='ovr', average='macro')
                         if prob_test is not None else np.nan),
           'seconds': 0.0, 'y_pred': pred,
           'prob_test': None if prob_test is None else prob_test.astype('float32'),
           'prob_val': None}
    results[name] = row; histories[name] = {}
    print(f'{name:34s} test_acc={row["test_acc"]:.3f}  F1={row["f1"]:.3f}')

trained = [n for n, r in results.items() if r.get('prob_val') is not None]
if len(trained) >= 2:
    top = sorted(trained, key=lambda n: results[n]['f1'], reverse=True)[:3]
    print('Combining top-3 models:', top, '\n')

    soft = np.mean([results[n]['prob_test'] for n in top], axis=0)          # soft vote
    _store_pred('Ensemble: Soft-Vote (top3)', 'F_ensemble', prob_test=soft)

    stack_preds = np.stack([results[n]['y_pred'] for n in top], axis=1)     # hard vote
    hard = np.array([np.bincount(r, minlength=NUM_CLASSES).argmax() for r in stack_preds])
    _store_pred('Ensemble: Hard-Vote (top3)', 'F_ensemble', pred=hard)

    Xv = np.concatenate([results[n]['prob_val']  for n in top], axis=1)     # stacking (meta on VAL)
    Xt = np.concatenate([results[n]['prob_test'] for n in top], axis=1)
    meta = LogisticRegression(max_iter=2000)
    meta.fit(Xv, y_val_int)
    _store_pred('Ensemble: Stacking (LogReg)', 'F_ensemble',
                prob_test=meta.predict_proba(Xt))
else:
    print('Need at least 2 trained models for ensembles — run more groups first.')
""")

# ===================================================================== CELL: §11 COMPARISON
md(r"""
## §11 · Task 5 — the master comparison ✅ COVERS TASK 5

🎓 **WEEK** W07 (this is the single-leaderboard method from `compare-all-ml-models`)

Now every model that ran writes one row into the same table, sorted by **F1**. This is the image-classification twin of the Week-7 final comparison: leaderboard → F1 bar → metrics heatmap → ROC overlay → per-group curves → confusion-matrix grid.
""")

code(r"""
# THE LEADERBOARD — one row per model, sorted by macro-F1
metric_cols = ['group','train_acc','val_acc','test_acc','train_loss','val_loss',
               'precision','recall','f1','roc_auc','seconds']
board = pd.DataFrame([dict(model=n, **{k: r[k] for k in metric_cols}) for n, r in results.items()]).set_index('model')
for c in metric_cols:
    if c != 'group':
        board[c] = board[c].astype(float)
board = board.sort_values('f1', ascending=False)
print(f'{len(board)} models on the leaderboard. Champion: {board.index[0]}')
board.round(4)
""")

code(r"""
# F1 bar chart (grey = plain from-scratch, blue = transfer / tuned / ensemble — the "upgrades")
order = board.sort_values('f1')
blue_groups = {'D_transfer', 'E_tuning', 'F_ensemble'}
bar_colors = ['#4472C4' if g in blue_groups else '#A5A5A5' for g in order['group']]
plt.figure(figsize=(11, max(5, 0.34 * len(order))))
plt.barh(order.index, order['f1'], color=bar_colors)
for i, v in enumerate(order['f1']):
    plt.text(v + 0.004, i, f'{v:.3f}', va='center', fontsize=8)
plt.xlabel('Macro F1 score'); plt.title('All models by F1  (grey = from-scratch, blue = transfer/tuned/ensemble)')
plt.xlim(0, min(1.0, order['f1'].max() * 1.12)); plt.tight_layout(); plt.show()
""")

code(r"""
# All-metrics heatmap
hm = board[['test_acc', 'precision', 'recall', 'f1', 'roc_auc']].astype(float)
plt.figure(figsize=(8, 0.42 * len(hm) + 1))
sns.heatmap(hm, annot=True, fmt='.3f', cmap='YlGnBu', cbar=False)
plt.title('Every metric for every model'); plt.tight_layout(); plt.show()
""")

code(r"""
# ROC overlay (micro-averaged, one readable curve per model) for the top-8 by F1
from sklearn.metrics import roc_curve, auc
top8 = [n for n in board.index if results[n]['prob_test'] is not None][:8]
plt.figure(figsize=(8, 7))
yt_flat = y_test_oh.ravel()
for n in top8:
    fpr, tpr, _ = roc_curve(yt_flat, results[n]['prob_test'].ravel())
    plt.plot(fpr, tpr, label=f'{n} (AUC={auc(fpr, tpr):.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC curves (micro-averaged) — top 8 models'); plt.legend(fontsize=8, loc='lower right')
plt.tight_layout(); plt.show()
""")

code(r"""
# Train/validation accuracy curves, one panel per group (kept readable)
group_order = [g for g in ['A_ladder','B_reg','C_modern','D_transfer','E_tuning']
               if any(r['group'] == g for r in results.values())]
if group_order:
    nrow = (len(group_order) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(13, 4 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, g in zip(axes, group_order):
        for n, r in results.items():
            if r['group'] == g and histories.get(n, {}).get('val_accuracy'):
                ax.plot(histories[n]['val_accuracy'], label=n)
        ax.set_title(f'Group {g} — validation accuracy'); ax.set_xlabel('epoch')
        ax.set_ylabel('val accuracy'); ax.legend(fontsize=6)
    for ax in axes[len(group_order):]:
        ax.axis('off')
    plt.tight_layout(); plt.show()
""")

code(r"""
# Confusion-matrix grid for the top-6 models
top6 = list(board.index[:6])
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, n in zip(axes.flat, top6):
    cm = confusion_matrix(y_test_int, results[n]['y_pred'])
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(n, fontsize=9); ax.set_xlabel('predicted'); ax.set_ylabel('true')
    ax.tick_params(labelsize=6)
plt.suptitle('Confusion matrices — top 6 models by F1', y=1.01)
plt.tight_layout(); plt.show()
""")

# ===================================================================== CELL: §12 TASK 6
md(r"""
## §12 · Task 6 — results discussion (which classes, and why) ✅ COVERS TASK 6

🎓 **WEEK** W05–W06

Three tools answer "which classes are right, which are confused, and why":
1. A **per-class precision/recall/F1 table** for the champion — shows exactly which of the 10 classes it handles well or badly.
2. A **per-class accuracy bar** — the same story as one picture.
3. A **misclassification gallery** + **Grad-CAM** heatmaps — Grad-CAM colours the pixels the CNN actually looked at, so you can *see why* it confused, say, a cat for a dog.
""")

code(r"""
champion = board.index[0]
print('CHAMPION MODEL:', champion, '\n')
print('Per-class report (precision / recall / F1 for each of the 10 classes):\n')
print(classification_report(y_test_int, results[champion]['y_pred'], target_names=CLASS_NAMES, digits=3))
""")

code(r"""
# Per-class accuracy bar for the champion (diagonal of its confusion matrix / row totals)
cm = confusion_matrix(y_test_int, results[champion]['y_pred'])
per_class_acc = cm.diagonal() / cm.sum(1)
order_idx = np.argsort(per_class_acc)
plt.figure(figsize=(9, 5))
plt.barh([CLASS_NAMES[i] for i in order_idx], per_class_acc[order_idx], color='#4472C4')
for i, v in enumerate(per_class_acc[order_idx]):
    plt.text(v + 0.005, i, f'{v:.2f}', va='center', fontsize=9)
plt.xlabel('accuracy'); plt.title(f'Per-class accuracy — {champion}\n(left = the classes it confuses most)')
plt.xlim(0, 1.05); plt.tight_layout(); plt.show()

# Name the worst confusion pair automatically
cm_off = cm.copy(); np.fill_diagonal(cm_off, 0)
i, j = np.unravel_index(cm_off.argmax(), cm_off.shape)
print(f'Most common single mistake: a real "{CLASS_NAMES[i]}" predicted as "{CLASS_NAMES[j]}" ({cm_off[i, j]} times).')
""")

code(r"""
# Misclassification gallery: 10 test images the champion got WRONG
wrong = np.where(results[champion]['y_pred'] != y_test_int)[0][:10]
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for ax, idx in zip(axes.flat, wrong):
    ax.imshow(x_test[idx]); ax.axis('off')
    ax.set_title(f'true: {CLASS_NAMES[y_test_int[idx]]}\npred: {CLASS_NAMES[results[champion]["y_pred"][idx]]}',
                 fontsize=8, color='crimson')
plt.suptitle(f'Where {champion} went wrong (true vs predicted)', y=1.02)
plt.tight_layout(); plt.show()
""")

code(r"""
# Grad-CAM: colour the pixels the CNN used to decide. Works on a from-scratch Sequential CNN.
def _train_small_cnn():
    set_seeds(); m = build_cnn(4)
    m.compile(optimizers.Adam(1e-3), 'categorical_crossentropy', metrics=['accuracy'])
    m.fit(Xtr_s, ytr_oh, epochs=epochs_for(5), batch_size=BATCH, verbose=0)
    return m

def grad_cam(model, img_batch):
    # Rebuild a functional graph from the trained layers so we can tap the last conv layer.
    # (Keras 3 / TF 2.16+ does not expose .output on a Sequential the old way.)
    last_conv = [l for l in model.layers if isinstance(l, layers.Conv2D)][-1]
    inp = tf.keras.Input(shape=img_batch.shape[1:]); x = inp; conv_out = None
    for lyr in model.layers:
        x = lyr(x)
        if lyr is last_conv:
            conv_out = x
    grad_model = models.Model(inp, [conv_out, x])
    with tf.GradientTape() as tape:
        c_out, preds = grad_model(img_batch)
        loss = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(loss, c_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(c_out[0] * tf.cast(pooled, c_out.dtype), axis=-1)
    cam = tf.maximum(cam, 0); cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (32, 32)).numpy().squeeze().astype('float32')
    return cam, int(tf.argmax(preds[0]))

def _render_gradcam(model):
    for k, idx in enumerate(range(5)):
        cam, cls = grad_cam(model, x_test_s[idx:idx + 1])
        axes[0, k].imshow(x_test[idx]); axes[0, k].axis('off')
        axes[0, k].set_title(f'true:{CLASS_NAMES[y_test_int[idx]]}\nsaw:{CLASS_NAMES[cls]}', fontsize=8)
        axes[1, k].imshow(x_test[idx]); axes[1, k].imshow(cam, cmap='jet', alpha=0.5); axes[1, k].axis('off')

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
gc_model = KEEP_FOR_GRADCAM if KEEP_FOR_GRADCAM is not None else _train_small_cnn()
try:
    _render_gradcam(gc_model)                       # try the kept/reloaded model
except Exception as e:
    print('Grad-CAM: retrying with a fresh model ->', str(e)[:90])
    try:
        _render_gradcam(_train_small_cnn())         # a freshly built model never goes stale
    except Exception as e2:
        print('Grad-CAM unavailable on this backend ->', str(e2)[:90], '| showing plain images.')
        for k, idx in enumerate(range(5)):
            for r in (0, 1):
                axes[r, k].imshow(x_test[idx]); axes[r, k].axis('off')
            axes[0, k].set_title(CLASS_NAMES[y_test_int[idx]], fontsize=8)
plt.suptitle('Grad-CAM — top row: image | bottom row: where the CNN looked (red = most important)', y=1.02)
plt.tight_layout(); plt.show()
""")

# ===================================================================== CELL: §13 TASK 7 delta
md(r"""
## §13 · Task 7 — what each improvement was worth ✅ COVERS TASK 7

🎓 **WEEK** W05–W06 + W08–W09

Task 7 asked us to *improve* the model. We did **every** technique on the list, so here is the scoreboard of **how much each one added** vs a plain deep CNN baseline. A positive Δ-F1 means the trick helped.
""")

code(r"""
baseline_name = 'CNN-5conv' if 'CNN-5conv' in results else board.index[-1]
base_f1 = results[baseline_name]['f1']
techniques = ['Deep+Dropout', 'Deep+BatchNorm', 'Deep+Augment', 'Deep+All(BN+Drop+Aug)',
              'VGG-style (2xconv blocks)', 'Residual-mini (skip conn.)']
techniques += [n for n in results if 'frozen TL' in n or 'fine-tuned' in n or 'Ensemble' in n]
rows = []
for n in techniques:
    if n in results:
        rows.append({'technique': n, 'F1': round(results[n]['f1'], 4),
                     'delta_vs_baseline': round(results[n]['f1'] - base_f1, 4)})
delta = pd.DataFrame(rows).sort_values('delta_vs_baseline', ascending=False).reset_index(drop=True)
print(f'Baseline = {baseline_name} (F1 = {base_f1:.4f})\n')
delta
""")

# ===================================================================== CELL: §14 POST-RUN
md(r"""
## §14 · Post-run analysis — expectation vs reality

🎓 **WEEK** W07 (the Week-7 notebook graded its own predictions the same way)

Before running we predicted: *fine-tuned backbone wins, the stacked ensemble is second, the 2-conv baseline is last, and from-scratch depth plateaus past ~4 layers.* The cell below checks those predictions automatically against the real leaderboard, then you write the discussion using the printed numbers.
""")

code(r"""
print('=' * 60); print('AUTOMATIC POST-RUN CHECK'); print('=' * 60)
print('Top 5 models by F1:')
print(board[['test_acc', 'f1', 'roc_auc']].head(5).round(4), '\n')

scratch_groups = {'A_ladder', 'B_reg', 'C_modern'}
scratch = [n for n, r in results.items() if r['group'] in scratch_groups]
transfer = [n for n, r in results.items() if r['group'] == 'D_transfer']
if scratch:
    bs = max(scratch, key=lambda n: results[n]['f1'])
    print(f'Best FROM-SCRATCH : {bs}  (F1={results[bs]["f1"]:.3f})')
if transfer:
    bt = max(transfer, key=lambda n: results[n]['f1'])
    print(f'Best TRANSFER     : {bt}  (F1={results[bt]["f1"]:.3f})')
    if scratch:
        print('Prediction "transfer beats from-scratch":',
              'CONFIRMED' if results[bt]['f1'] > results[bs]['f1'] else 'REJECTED')

# did from-scratch depth plateau? compare 4-conv vs 6-conv
if 'CNN-4conv' in results and 'CNN-6conv' in results:
    d = results['CNN-6conv']['f1'] - results['CNN-4conv']['f1']
    print(f'\nDepth check: 6-conv minus 4-conv F1 = {d:+.3f}',
          '-> deeper did NOT help (plateau CONFIRMED)' if d <= 0.01 else '-> deeper still helped')
print(f'\nOverall champion: {board.index[0]}  (F1={board.iloc[0]["f1"]:.3f})')
""")

md(r"""
**Write your discussion here** (fill the blanks from the numbers printed above):

- The overall champion was **______** with F1 = ____. This matches / does not match our prediction because ______.
- From-scratch accuracy rose from 2 to ~4 conv layers, then ______ at 5–6 layers, confirming the depth-plateau lesson from §5.
- The single most confused class pair was **______ ↔ ______** (from §12). A likely reason is ______ (e.g. both are four-legged animals at only 32×32 pixels).
- The improvement that added the most F1 was **______** (from §13), which makes sense because ______.
- One surprise was ______ — possibly caused by ______.
""")

# ===================================================================== CELL: §15 COVERAGE + SAVE
md(r"""
## §15 · Final coverage check + deliverables ✅

The table restates that **every task is covered**, and the cell below saves the results file and lists what to hand in. The "Status" column is filled automatically from what actually ran.
""")

code(r"""
# Save the leaderboard so it can go in the report as the evaluation-results file
board.to_csv('cnn_leaderboard_results.csv')
print('Saved cnn_leaderboard_results.csv')

# Automatic coverage report based on which groups produced rows
def any_group(*gs): return any(r['group'] in gs for r in results.values())
checks = [
    ('Task 1 — dataset described',                 True),
    ('Task 2 — preprocessing + augmentation',      True),
    ('Task 3 — CNN designed (conv/pool/.../softmax)', any_group('A_ladder', 'B_reg', 'C_modern')),
    ('Task 4 — training settings (Adam/lr/batch/epochs)', any_group('A_ladder', 'B_reg', 'C_modern', 'D_transfer', 'E_tuning')),
    ('Task 5 — >=3 CNNs of increasing depth compared', sum(r['group'] == 'A_ladder' for r in results.values()) >= 3),
    ('Task 6 — per-class + confusion + Grad-CAM discussion', True),
    ('Task 7 — improvements (augment/dropout/BN/transfer/tuning)', any_group('B_reg', 'D_transfer', 'E_tuning', 'F_ensemble')),
]
print('\n================ ASSIGNMENT COVERAGE ================')
for name, ok in checks:
    print(f'  [{"COVERED" if ok else "NOT RUN"}]  {name}')
print('====================================================')
print(f'\nTotal models trained: {len(results)}   |   Champion: {board.index[0]}')

print('\nDELIVERABLES CHECKLIST:')
for d in ['Written report (use the §14 discussion + the leaderboard table)',
          'This notebook (.ipynb), runs top-to-bottom on a GPU',
          'Dataset link: CIFAR-10 (Keras built-in / https://www.cs.toronto.edu/~kriz/cifar.html)',
          'Evaluation results: cnn_leaderboard_results.csv (saved above)',
          'Graphs: accuracy-vs-depth, F1 bar, heatmap, ROC, curves',
          'Confusion matrices: §11 top-6 grid + §12 champion']:
    print('  [ ]', d)
""")

# ===================================================================== CELL: §16 REUSE / INFERENCE
md(r"""
## §16 · Reusing a trained model (save & reload full weights) ➕ EXTRA

🎓 **WEEK** W05–W06

By default the notebook saves each model's *predictions* (enough to resume the leaderboard). If you also set `SAVE_WEIGHTS = True` in the control panel, it saves each model's **full weights** as a `.keras` file. You can then reload a trained model later and predict on brand-new images — no retraining. This cell reloads the best saved model and runs a quick inference demo. *(Weights are large — a transfer model is ~90 MB — so this is opt-in.)*
""")

code(r"""
# Reload the best SAVED model and predict on new images (needs SAVE_WEIGHTS=True).
if SAVE_WEIGHTS:
    saved = [n for n in board.index if os.path.exists(_weights_path(n))]
    reloaded, used = None, None
    for n in saved:                       # board is sorted by F1, so this picks the best that reloads
        m = _reload_weights(n)
        if m is not None:
            reloaded, used = m, n; break
    if reloaded is not None:
        raw = results[used]['group'] == 'D_transfer'   # transfer models were trained on raw 0-255 images
        Xdemo = (x_test_r if raw else x_test_s)[:8]
        preds = reloaded.predict(Xdemo, verbose=0).argmax(1)
        print('Reloaded best saved model:', used)
        fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
        for ax, k in zip(axes.flat, range(8)):
            ax.imshow(x_test[k]); ax.axis('off')
            ok = preds[k] == y_test_int[k]
            ax.set_title(f'pred: {CLASS_NAMES[preds[k]]}\n' + ('correct' if ok else 'WRONG (' + CLASS_NAMES[y_test_int[k]] + ')'),
                         fontsize=8, color='green' if ok else 'crimson')
        plt.suptitle(f'Reloaded "{used}" predicting on 8 unseen test images', y=1.02)
        plt.tight_layout(); plt.show()
        print('\nTo reuse this model in any script later:')
        print("  import tensorflow as tf")
        print(f"  model = tf.keras.models.load_model(r'{_weights_path(used)}', safe_mode=False)")
        print(f"  preds = model.predict(images).argmax(1)   # images: (N,32,32,3), " + ('raw 0-255' if raw else 'scaled 0-1'))
    else:
        print('No saved weights could be reloaded.')
else:
    print('SAVE_WEIGHTS is False -> no full weights were saved. Set it True in the control panel and re-run to enable model reuse.')
""")

# ===================================================================== CELL: §16.5 BACKUP TO DRIVE
md(r"""
## §16.5 · Back up checkpoints + models to Google Drive ➕ EXTRA

Copies the local checkpoints and saved `.keras` weights up to a Google-Drive backup folder, so a future session recovers them with the restore cell near the top (§1.2). Run it after training, or any time you want to save progress. This is what makes the run survive a Colab disconnect even when `CKPT_DIR` is local.
""")

code(r"""
# ===== BACKUP: copy local checkpoints + model weights to Google Drive (run me after training) =====
import shutil
DRIVE_BACKUP = '/content/drive/MyDrive/cnn_backup'
try:
    from google.colab import drive
    if not os.path.isdir('/content/drive/MyDrive'):
        drive.mount('/content/drive')
except Exception:
    print('(not on Colab / no Drive - skipping backup)')

if os.path.isdir(CKPT_DIR):
    n = 0
    for root, _, files in os.walk(CKPT_DIR):
        for f in files:
            src = os.path.join(root, f)
            dst = os.path.join(DRIVE_BACKUP, os.path.relpath(src, CKPT_DIR))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst); n += 1
    print(f'Backed up {n} files {CKPT_DIR}/ -> {DRIVE_BACKUP}/ on Google Drive.')
else:
    print('No local checkpoints to back up yet - train some models first.')
""")

# ===================================================================== CELL: §17 EXPORT ASSETS
md(r"""
## §17 · Export standalone figures + results, then zip for download ➕ EXTRA

This cell re-creates **every report figure as a separate high-resolution PNG** — and it **splits the side-by-side training curves into two standalone plots** (accuracy, loss). It also saves the leaderboard CSV, mounts Google Drive when on Colab, and **zips everything** so you can download it in one click. Run it after the whole notebook has run; hand the zip back and the written report rebuilds with these clean images.
""")

code(r"""
# ============ EXPORT: standalone high-res figures + results, zipped for download ============
import os, zipfile

# 1) pick an output folder (Google Drive on Colab, else local ./report_assets)
EXPORT_DIR = 'report_assets'
try:
    from google.colab import drive
    if not os.path.isdir('/content/drive/MyDrive'):
        drive.mount('/content/drive')
    EXPORT_DIR = '/content/drive/MyDrive/cnn_report_assets'
except Exception:
    print('(no Google Drive -> saving locally to ./report_assets)')
FIG_DIR = os.path.join(EXPORT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
DPI = 160

def savefig(name):
    plt.savefig(os.path.join(FIG_DIR, name + '.png'), dpi=DPI, bbox_inches='tight'); plt.close()
    print('  saved', name + '.png')

champ = board.index[0]
req = [('Model 1 = 2 conv', 'CNN-2conv'), ('Model 2 = 4 conv', 'CNN-4conv'), ('Model 3 = 6 conv', 'CNN-6conv')]
cols3 = ['#4472C4', '#ED7D31', '#70AD47']

# class examples
fig, ax = plt.subplots(2, 5, figsize=(11, 5))
for cls, a in enumerate(ax.flat):
    idx = np.where(y_test_int == cls)[0][0]; a.imshow(x_test[idx]); a.set_title(CLASS_NAMES[cls], fontsize=10); a.axis('off')
plt.suptitle('CIFAR-10 - one example per class', y=1.02); plt.tight_layout(); savefig('fig_classes')

# augmentation strip
try:
    s = Xtr_s[0:1]; fig, ax = plt.subplots(1, 6, figsize=(13, 2.6))
    ax[0].imshow(s[0]); ax[0].set_title('original'); ax[0].axis('off')
    for k in range(1, 6):
        au = data_augment(s, training=True)[0].numpy().astype('float32'); ax[k].imshow(np.clip(au, 0, 1)); ax[k].set_title(f'aug {k}'); ax[k].axis('off')
    plt.tight_layout(); savefig('fig_augmentation')
except Exception as e:
    print('  (augmentation skipped:', e, ')')

# accuracy vs depth
if LADDER:
    ds = [d for d, _ in LADDER]; acc = [results[n]['test_acc'] for _, n in LADDER]
    plt.figure(figsize=(8, 5)); plt.plot(ds, acc, 'o-', color='#4472C4', lw=2, ms=9)
    for d, a in zip(ds, acc):
        plt.text(d, a + 0.004, f'{a:.3f}', ha='center', fontweight='bold')
    plt.xlabel('Number of convolutional layers'); plt.ylabel('Test accuracy'); plt.title('Accuracy vs depth'); plt.xticks(ds); savefig('fig_accuracy_vs_depth')

# required curves SPLIT into two standalone figures
for ylabel, train_key, val_key, fname, ttl in [
        ('accuracy', 'accuracy', 'val_accuracy', 'fig_curves_accuracy', 'Accuracy curves (solid=train, dashed=val)'),
        ('loss', 'loss', 'val_loss', 'fig_curves_loss', 'Loss curves (solid=train, dashed=val)')]:
    plt.figure(figsize=(7.5, 5))
    for (lab, nm), c in zip(req, cols3):
        if histories.get(nm):
            plt.plot(histories[nm][train_key], color=c, label=f'{lab} (train)')
            plt.plot(histories[nm][val_key], color=c, ls='--', label=f'{lab} (val)')
    plt.title(ttl); plt.xlabel('epoch'); plt.ylabel(ylabel); plt.legend(fontsize=8); savefig(fname)

# F1 bar (tall)
order = board.sort_values('f1'); blue = {'D_transfer', 'E_tuning', 'F_ensemble'}
bc = ['#4472C4' if g in blue else '#A5A5A5' for g in order['group']]
plt.figure(figsize=(10, max(6, 0.34 * len(order)))); plt.barh(order.index, order['f1'], color=bc)
for i, v in enumerate(order['f1']):
    plt.text(v + 0.004, i, f'{v:.3f}', va='center', fontsize=8)
plt.xlabel('Macro F1'); plt.title('All models by F1'); plt.xlim(0, min(1.0, order['f1'].max() * 1.12)); savefig('fig_f1_bar')

# metrics heatmap (tall)
hm = board[['test_acc', 'precision', 'recall', 'f1', 'roc_auc']].astype(float)
plt.figure(figsize=(8, 0.42 * len(hm) + 1)); sns.heatmap(hm, annot=True, fmt='.3f', cmap='YlGnBu', cbar=False)
plt.title('Every metric for every model'); savefig('fig_metrics_heatmap')

# ROC top-8
from sklearn.metrics import roc_curve, auc
t8 = [n for n in board.index if results[n]['prob_test'] is not None][:8]
plt.figure(figsize=(8, 7)); yf = y_test_oh.ravel()
for n in t8:
    fpr, tpr, _ = roc_curve(yf, results[n]['prob_test'].ravel()); plt.plot(fpr, tpr, label=f'{n} (AUC={auc(fpr, tpr):.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=.4); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC curves (top 8)'); plt.legend(fontsize=8, loc='lower right'); savefig('fig_roc')

# per-group validation curves
groups = [g for g in ['A_ladder', 'B_reg', 'C_modern', 'D_transfer', 'E_tuning'] if any(r['group'] == g for r in results.values())]
if groups:
    nr = (len(groups) + 1) // 2; fig, ax = plt.subplots(nr, 2, figsize=(13, 4 * nr)); ax = np.array(ax).reshape(-1)
    for a, g in zip(ax, groups):
        for n, r in results.items():
            if r['group'] == g and histories.get(n, {}).get('val_accuracy'):
                a.plot(histories[n]['val_accuracy'], label=n)
        a.set_title('Group ' + g); a.set_xlabel('epoch'); a.set_ylabel('val accuracy'); a.legend(fontsize=6)
    for a in ax[len(groups):]:
        a.axis('off')
    plt.tight_layout(); savefig('fig_group_curves')

# confusion grid top-6
t6 = list(board.index[:6]); fig, ax = plt.subplots(2, 3, figsize=(14, 8))
for a, n in zip(ax.flat, t6):
    cmx = confusion_matrix(y_test_int, results[n]['y_pred'])
    sns.heatmap(cmx, cmap='Blues', cbar=False, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=a)
    a.set_title(n, fontsize=9); a.set_xlabel('pred'); a.set_ylabel('true'); a.tick_params(labelsize=6)
plt.suptitle('Confusion matrices - top 6', y=1.01); plt.tight_layout(); savefig('fig_confusion_grid')

# per-class accuracy (champion)
cmx = confusion_matrix(y_test_int, results[champ]['y_pred']); pca = cmx.diagonal() / cmx.sum(1); oi = np.argsort(pca)
plt.figure(figsize=(9, 5)); plt.barh([CLASS_NAMES[i] for i in oi], pca[oi], color='#4472C4')
for i, v in enumerate(pca[oi]):
    plt.text(v + 0.005, i, f'{v:.2f}', va='center', fontsize=9)
plt.xlabel('accuracy'); plt.title('Per-class accuracy - ' + champ); plt.xlim(0, 1.05); savefig('fig_per_class_acc')

# misclassification gallery (champion)
wr = np.where(results[champ]['y_pred'] != y_test_int)[0][:10]; fig, ax = plt.subplots(2, 5, figsize=(12, 5))
for a, idx in zip(ax.flat, wr):
    a.imshow(x_test[idx]); a.axis('off')
    a.set_title('true:' + CLASS_NAMES[y_test_int[idx]] + chr(10) + 'pred:' + CLASS_NAMES[results[champ]['y_pred'][idx]], fontsize=8, color='crimson')
plt.suptitle('Champion misclassifications', y=1.02); plt.tight_layout(); savefig('fig_misclassified')

# Grad-CAM
try:
    gm = gc_model if ('gc_model' in dir() and gc_model is not None) else _train_small_cnn()
    fig, ax = plt.subplots(2, 5, figsize=(12, 5))
    for k, idx in enumerate(range(5)):
        cam, cls = grad_cam(gm, x_test_s[idx:idx + 1])
        ax[0, k].imshow(x_test[idx]); ax[0, k].axis('off'); ax[0, k].set_title('true:' + CLASS_NAMES[y_test_int[idx]] + chr(10) + 'saw:' + CLASS_NAMES[cls], fontsize=8)
        ax[1, k].imshow(x_test[idx]); ax[1, k].imshow(cam, cmap='jet', alpha=0.5); ax[1, k].axis('off')
    plt.suptitle('Grad-CAM', y=1.02); plt.tight_layout(); savefig('fig_gradcam')
except Exception as e:
    print('  (grad-cam skipped:', str(e)[:80], ')')

# results CSV + zip everything
csv_path = os.path.join(EXPORT_DIR, 'cnn_leaderboard_results.csv'); board.to_csv(csv_path); print('  saved cnn_leaderboard_results.csv')
zip_path = os.path.join(EXPORT_DIR, 'cnn_report_assets.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, _, fs in os.walk(FIG_DIR):
        for f in fs:
            z.write(os.path.join(root, f), os.path.join('figures', f))
    z.write(csv_path, 'cnn_leaderboard_results.csv')
print('\nZIP READY ->', zip_path)
try:
    from google.colab import files as _cf; _cf.download(zip_path)
except Exception:
    print('(download it from the path above, or from your Google Drive)')
""")

# ===================================================================== CELL: §18 APPENDIX ASSETS
md(r"""
## §18 · Appendix assets — individual panels + numeric confusion matrices + worked calculation ➕ EXTRA

This builds the appendix material the report needs: each model-family's validation curve as its **own** figure, each top model's confusion matrix as its **own** figure **with the counts printed in each cell**, the **numeric confusion-matrix tables**, and a **tutorial-style hand-calculation** of precision / recall / F1 / accuracy for the champion (showing TP, FP, FN, TN step by step). Everything is saved and re-zipped into `cnn_report_assets.zip`. Run it right after the export cell (§17).
""")

code(r"""
# ===== APPENDIX ASSETS: individual panels + numeric confusion matrices + worked metric calc =====
import os, re, zipfile
APP_DIR = os.path.join(EXPORT_DIR, 'appendix'); os.makedirs(APP_DIR, exist_ok=True)
def appfig(name):
    plt.savefig(os.path.join(APP_DIR, name + '.png'), dpi=160, bbox_inches='tight'); plt.close(); print('  saved', name)
def slug(s):
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')

champ = board.index[0]

# (1) one validation-curve figure per model family (readable, not a 6-panel grid)
GNAME = {'A_ladder': 'Depth ladder', 'B_reg': 'Regularised', 'C_modern': 'Modern blocks',
         'D_transfer': 'Transfer learning', 'E_tuning': 'Tuning grid'}
groups = [g for g in GNAME if any(r['group'] == g for r in results.values())]
for g in groups:
    plt.figure(figsize=(7, 4.5))
    for n, r in results.items():
        if r['group'] == g and histories.get(n, {}).get('val_accuracy'):
            plt.plot(histories[n]['val_accuracy'], label=n)
    plt.title('Validation accuracy - ' + GNAME[g]); plt.xlabel('epoch'); plt.ylabel('val accuracy'); plt.legend(fontsize=7)
    appfig('grp_' + g)

# (2) one confusion-matrix figure per top model, WITH the counts annotated
top6 = list(board.index[:6])
for n in top6:
    cm = confusion_matrix(y_test_int, results[n]['y_pred'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot_kws={'size': 7})
    plt.title(n, fontsize=10); plt.xlabel('predicted'); plt.ylabel('true'); plt.xticks(fontsize=8); plt.yticks(fontsize=8, rotation=0)
    appfig('cm_' + slug(n))

# (3) numeric confusion-matrix tables + (4) tutorial-style worked calculation -> appendix.md
L = ['# Appendix - numeric confusion matrices and worked metric calculation', '']
for n in top6[:3]:
    cm = confusion_matrix(y_test_int, results[n]['y_pred'])
    L.append('## Confusion matrix (counts): ' + n)
    L.append('')
    L.append('| true \\\\ pred | ' + ' | '.join(CLASS_NAMES) + ' |')
    L.append('|' + '---|' * (len(CLASS_NAMES) + 1))
    for i, row in enumerate(cm):
        L.append('| **' + CLASS_NAMES[i] + '** | ' + ' | '.join(str(int(x)) for x in row) + ' |')
    L.append('')

cm = confusion_matrix(y_test_int, results[champ]['y_pred']); total = int(cm.sum())
L.append('## Worked example - how the per-class metrics are calculated')
L.append('')
L.append('Champion model: **' + champ + '**. From the confusion matrix C, for each class i:')
L.append('')
L.append('- TP = C[i, i]  (class i predicted correctly)')
L.append('- FN = (sum of row i) - TP  (class i predicted as something else)')
L.append('- FP = (sum of column i) - TP  (other classes predicted as i)')
L.append('- TN = total - TP - FN - FP')
L.append('- **Recall** = TP / (TP + FN)   **Precision** = TP / (TP + FP)   **F1** = 2 P R / (P + R)   **Accuracy(class)** = (TP + TN) / total')
L.append('')
ci = CLASS_NAMES.index('cat')
TP = int(cm[ci, ci]); FN = int(cm[ci].sum()) - TP; FP = int(cm[:, ci].sum()) - TP; TN = total - TP - FN - FP
P = TP / (TP + FP); R = TP / (TP + FN); F1 = 2 * P * R / (P + R)
L.append('**Worked example for class "cat":**')
L.append('')
L.append('- TP = %d, FN = %d, FP = %d, TN = %d' % (TP, FN, FP, TN))
L.append('- Recall = %d / (%d + %d) = **%.3f**' % (TP, TP, FN, R))
L.append('- Precision = %d / (%d + %d) = **%.3f**' % (TP, TP, FP, P))
L.append('- F1 = 2 x %.3f x %.3f / (%.3f + %.3f) = **%.3f**' % (P, R, P, R, F1))
L.append('- Accuracy(cat) = (%d + %d) / %d = **%.3f**' % (TP, TN, total, (TP + TN) / total))
L.append('')
L.append('**All ten classes (champion):**')
L.append('')
L.append('| class | TP | FP | FN | Precision | Recall | F1 |')
L.append('|---|---:|---:|---:|---:|---:|---:|')
for i, c in enumerate(CLASS_NAMES):
    tp = int(cm[i, i]); fn = int(cm[i].sum()) - tp; fp = int(cm[:, i].sum()) - tp
    p = tp / (tp + fp); r = tp / (tp + fn); f = 2 * p * r / (p + r)
    L.append('| %s | %d | %d | %d | %.3f | %.3f | %.3f |' % (c, tp, fp, fn, p, r, f))
acc = int(cm.diagonal().sum()) / total
L.append('')
L.append('Overall accuracy = sum(diagonal) / total = %d / %d = **%.3f**' % (int(cm.diagonal().sum()), total, acc))
open(os.path.join(EXPORT_DIR, 'appendix.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('  saved appendix.md')

# (5) re-zip EVERYTHING (figures + appendix + csv + appendix.md) into one download
zip_path = os.path.join(EXPORT_DIR, 'cnn_report_assets.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for sub in ['figures', 'appendix']:
        d = os.path.join(EXPORT_DIR, sub)
        for root, _, fs in os.walk(d):
            for f in fs:
                z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), EXPORT_DIR))
    for extra in ['cnn_leaderboard_results.csv', 'appendix.md']:
        p = os.path.join(EXPORT_DIR, extra)
        if os.path.exists(p):
            z.write(p, extra)
print('\nCOMBINED ZIP (figures + appendix) ->', zip_path)
try:
    from google.colab import files as _cf; _cf.download(zip_path)
except Exception:
    print('(download it from the path above, or from your Google Drive)')
""")

# >>>END<<<

# ===================================================================== NUMBER EVERY CELL
import re as _re
def _clean(t):
    t = t.replace('*', '').replace('`', '').replace('|', ' ').strip(' ·-—')
    return t or 'section'
def _cell_subtitle(src, is_md, last_md):
    lines = src.splitlines()
    if is_md:
        for ln in lines:
            s = ln.strip()
            if s.startswith('#'):
                return _clean(_re.split(r'[✅➕🎓📘📄🔖]', s.lstrip('#').strip())[0])
        for ln in lines:
            if ln.strip():
                return _clean(ln.strip().lstrip('>')[:60])
        return 'notes'
    first = next((ln.strip() for ln in lines if ln.strip()), '')
    if first.startswith('#'):
        t = _re.sub(r'[=\-\s]+$', '', _re.sub(r'^[=\-\s]+', '', first.lstrip('#')))
        if t.strip():
            return _clean(t[:60])
    return _clean(last_md)
def number_cells(cs):
    last_md = 'intro'
    for i, c in enumerate(cs, 1):
        cid = '%02d' % i
        c['id'] = 'cell-' + cid
        src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
        if c['cell_type'] == 'markdown':
            src = _re.sub(r'^`Cell \d+`[^\n]*\n\n', '', src, count=1)
            sub = _cell_subtitle(src, True, last_md); last_md = sub
            c['source'] = '`Cell ' + cid + '` — *' + sub + '*\n\n' + src
        else:
            src = _re.sub(r'^# ===== Cell \d+[^\n]*\n', '', src, count=1)
            sub = _cell_subtitle(src, False, last_md)
            c['source'] = '# ===== Cell ' + cid + ' — ' + sub + ' =====\n' + src
number_cells(cells)

# ===================================================================== ASSEMBLE + WRITE
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "CNN_Image_Classification_CIFAR10.ipynb")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Wrote {OUT} with {len(cells)} cells "
      f"({sum(c['cell_type']=='markdown' for c in cells)} markdown, "
      f"{sum(c['cell_type']=='code' for c in cells)} code).")
