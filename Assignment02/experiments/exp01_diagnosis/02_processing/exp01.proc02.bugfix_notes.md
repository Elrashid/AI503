# A2 Notebook — Bug Fix Log

A running log of fixes to `A2_quantization_safety.ipynb`. Each entry is structured so it can be cited verbatim in the paper's Methodology / Limitations section.

---

## Index

| # | Date | Title | Severity | Cells touched |
|---|---|---|---|---|
| 1 | 2026-05-05 | `bitsandbytes==0.44.1` incompatible with CUDA 12.8 / py3.12 | **blocker** | install cell |
| 2 | 2026-05-05 | `HF_TOKEN` import-time `UserWarning` from `huggingface_hub` | cosmetic | imports cell |
| 3 | 2026-05-05 | `walledai/AdvBench` is now a gated HF dataset | **blocker** | dataset cell |
| 4 | 2026-05-05 | `NameError: rng is not defined` in MMLU on cached-dataset path | **blocker** | dataset cell (regression from #3) |
| 5 | 2026-05-05 | `padding_side='left'` required for decoder-only generation | **blocker** (silent corruption) | loader cell |
| 6 | 2026-05-05 | Sampling-with-greedy-decoding UserWarnings (×3 per batch) | cosmetic | loader + generate_batch |

> Note: a concurrent Fix #6 in the SLR `paper_reader.html` (`ttsStop` null-deref on first paper load) is documented inline in `Assignment01/scripts/generate_paper_reader.py`, not here, because it's outside the A2 notebook scope. The numbering above restarts at 6 for the A2 context.

---

## 🐛 Fix #1 — `bitsandbytes==0.44.1` incompatible with Colab CUDA 12.8 + Python 3.12

**Date:** 2026-05-05
**Cells touched:** install cell (cell 3)
**Severity:** blocker — without this fix, INT8 and NF4 conditions cannot run

### 1. Original (broken) state

**Pin in install cell:**
```python
!pip -q install --upgrade transformers==4.46.0 accelerate==1.0.1 \
    bitsandbytes==0.44.1 datasets==3.0.1 sentencepiece pandas matplotlib
```

**Symptom (verbatim from cell 7 stdout on Colab L4 / 2026-05):**
```
WARNING:bitsandbytes.cextension:Could not find the bitsandbytes CUDA binary
  at PosixPath('/usr/local/lib/python3.12/dist-packages/bitsandbytes/libbitsandbytes_cuda128.so')
WARNING:bitsandbytes.cextension:The installed version of bitsandbytes was
  compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and
  GPU quantization are unavailable.
bitsandbytes:   not importable — No module named 'triton.ops'
```

### 2. Root cause

Two independent kernel-update breakages, both *environmental* (not our code):

1. **CUDA-binary mismatch.** Colab now ships CUDA 12.8 (`libbitsandbytes_cuda128.so` is what bnb 0.44 looks for at import time). bitsandbytes 0.44.1 was released before CUDA 12.8 existed; it ships precompiled `.so` files only for CUDA 11.8 / 12.1 / 12.4. With no matching binary, bnb falls back to a CPU-only stub.
2. **Triton API removal.** The Colab kernel updated to `torch 2.10` + modern `triton`. In modern triton, `triton.ops` was removed in favour of `triton.runtime.jit`. bitsandbytes 0.44.1 still does `from triton.ops import ...`, so the whole `bitsandbytes` import crashes before reaching the cextension fallback.

These are both "library too old for kernel" problems. We didn't write any quantization code — the fix is just a version bump.

### 3. Fix

**Patched install cell:**
```python
!pip -q install --upgrade \
    transformers==4.46.3 \
    'accelerate>=1.0.1' \
    'bitsandbytes>=0.46.0' \
    datasets==3.0.1 \
    sentencepiece \
    'pandas==2.2.2' \
    matplotlib

# Sanity check — fail fast if bnb still can't load its CUDA binary
import importlib, sys, torch
for mod in ['transformers', 'bitsandbytes']:
    try:
        m = importlib.import_module(mod)
        print(f'  {mod:15} {m.__version__}')
    except ImportError as e:
        print(f'  {mod:15} FAILED — {e}'); sys.exit(1)
import bitsandbytes as bnb
from bitsandbytes import cextension as _cx
if torch.cuda.is_available() and hasattr(_cx, 'lib') and _cx.lib is None:
    print('  ⚠ bitsandbytes loaded but its CUDA binary did NOT load — restart and retry')
else:
    print('  ✓ bitsandbytes CUDA binary loaded successfully')
```

Three pin-level changes:

| Package | Was | Now | Reason |
|---|---|---|---|
| `transformers` | `4.46.0` (yanked) | `4.46.3` | 4.46.0 was yanked from PyPI; 4.46.3 is the patched re-release |
| `bitsandbytes` | `0.44.1` | `>=0.46.0` | 0.46.0 ships CUDA 12.8 binary + uses post-`triton.ops` triton API |
| `pandas` | unpinned (resolved to `3.0.2`) | `==2.2.2` | matches Colab's preinstalled pandas; avoids breaking `google.colab.drive` and gradio |

A post-install sanity check was added so a bad install fails *here* with a readable message, not 30 minutes later inside the run loop.

### 4. Steps to verify

1. **Restart runtime** in Colab (`Runtime → Restart session`). This is **required** — the broken `bitsandbytes 0.44.1` is still resident in memory; restarting drops it.
2. Re-run **cell 3** (the install cell). Expected stdout:
   ```
   transformers    4.46.3
   bitsandbytes    0.46.x
   ✓ bitsandbytes CUDA binary loaded successfully
   ```
   No yanked-package warning, no pandas dependency-resolver error.
3. Re-run **cell 5** (imports). The `bitsandbytes was compiled without GPU support` warning should be **gone**.
4. Re-run **cell 7** (Drive + `log_active_config()`). Look for:
   ```
   bitsandbytes:   0.46.x  (4-bit/8-bit support: yes)
   ```
5. Once verified, proceed to cell 9 (config) and onwards. The fix is correct iff cell 18 (model loader) can produce all three quant variants without `RuntimeError: bitsandbytes ... GPU support unavailable`.

### 5. Citations

- **Library:** Dettmers, T., Lewis, M., Belkada, Y. and Zettlemoyer, L. (2022) 'LLM.int8(): 8-bit matrix multiplication for transformers at scale', *NeurIPS*. **[RP11]** — defines INT8 path that requires bnb to function.
- **Library:** Dettmers, T., Pagnoni, A., Holtzman, A. and Zettlemoyer, L. (2023) 'QLoRA: efficient finetuning of quantized LLMs', *NeurIPS*. **[RP17]** — defines NF4 datatype, requires bnb 0.39+; we need 0.46+ for CUDA 12.8.
- **Release notes:** bitsandbytes 0.46.0 changelog (CUDA 12.8 binary added; triton import migrated). https://github.com/TimDettmers/bitsandbytes/releases/tag/0.46.0
- **Methodology paper:** the bnb library is the reference implementation cited as the "practical 4-bit baseline" in the paper's Section 5.3 — see `Assignment02/draft/05_methodology.md` and `07_experimental_setup.md`.

### 6. Goal alignment

This fix unblocks **two of the six experimental conditions** (INT8 and NF4 on both models) — i.e. it unblocks the entire H1 hypothesis test. Without it the run loop only produces FP16 baselines, leaving no quantization axis to test. Specifically:

- **RQ1** (does PTQ to ≤4-bit degrade safety differently from utility?) — requires INT8 + NF4 conditions.
- **H1** (refusal drops faster than MMLU at NF4) — directly impossible without the NF4 condition.
- **Rubric §4 (Experimental Setup, 15%)** — requires reproducible quantization configs; locked-in pinned versions make the experiment replayable.
- **Rubric §6 (Dataset, 20%) / §8 (Results, 20%)** — unaffected by this fix but unrunnable until quantization works.

This is the kind of fix that goes into the paper's **Section 7.6 (Threats to validity)** as: *"Pipeline tested on Colab L4 with CUDA 12.8, torch 2.10, Python 3.12, bitsandbytes 0.46.0. Earlier bnb versions (0.44.x) lack a CUDA-12.8 binary and import a removed `triton.ops` symbol, breaking on the current Colab kernel."*

---

## 🐛 Fix #2 — `HF_TOKEN secret timed out` UserWarning at import

**Date:** 2026-05-05
**Cells touched:** imports cell (cell 7)
**Severity:** cosmetic — no functional impact, but noisy in lecture/published output

### 1. Original (broken) state

**Symptom (verbatim from cell 7 stdout):**
```
/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:104: UserWarning:
Error while fetching `HF_TOKEN` secret value from your vault:
'Requesting secret HF_TOKEN timed out. Secrets can only be fetched
when running from the Colab UI.'.
You are not authenticated with the Hugging Face Hub in this notebook.
```

### 2. Root cause

`huggingface_hub ≥ 0.20` probes `from google.colab import userdata; userdata.get("HF_TOKEN")` *at module import time*. The RPC has a 5-second timeout. If the user has not added an `HF_TOKEN` Colab secret via the 🔑 sidebar (we don't — our flow uses Drive cache, loaded later in §1.5), the probe times out and emits a UserWarning.

The warning is **cosmetic only** — token loading in cell 13 still works correctly because §1.5 explicitly calls `huggingface_hub.login(token, ...)` which bypasses the probe.

### 3. Fix

Three `warnings.filterwarnings(...)` calls at the top of the imports cell, *before* `transformers` is imported:

```python
import warnings
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*",                                    category=UserWarning)
warnings.filterwarnings("ignore", message=".*not authenticated with the Hugging Face Hub.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Requesting secret.*",                           category=UserWarning)
```

Each pattern targets one specific warning *message*. Other unrelated UserWarnings (e.g. legitimate deprecation notices) still surface — we suppress only the three known-cosmetic auth ones.

### 4. Steps to verify

1. Re-run cell 7 (imports).
2. Expected stdout: `Device: cuda | GPU: NVIDIA L4` and **no** yellow `UserWarning` block from `_auth.py:104`.
3. Sanity check that other warnings still work: any package emitting a *real* `UserWarning` not matching our three regex patterns will print normally.

### 5. Citations

- `huggingface_hub` token-resolution order (file → env var → Colab userdata): https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/utils/_auth.py
- Our token-loading flow (§1.5 / cell 13) takes precedence over whatever the import-time probe finds, so suppressing the probe warning is safe.

### 6. Goal alignment

Pure UX / lecture-quality fix. Zero impact on:

- H1 hypothesis test
- RQ1, RQ2, RQ3
- Any experimental condition

Documented anyway because (a) when we publish the notebook with results, leaving an unexplained `UserWarning` in the output makes future readers ask *"is this broken?"* and (b) the bug-fix-log rule (per `feedback_bug_fix_documentation.md`) requires a paper trail for *every* change to imports, not just functional bugs.

The Methodology paragraph in §7.4 of the paper covers this implicitly via the pinned-version table — no separate sentence needed in the paper.

---

## 🐛 Fix #3 — `walledai/AdvBench` is a gated HuggingFace dataset

**Date:** 2026-05-05
**Cells touched:** dataset construction cell (cell 19)
**Severity:** blocker — without this fix, no safety prompts can be loaded → entire experiment cannot run

### 1. Original (broken) state

**Symptom (verbatim):**
```
DatasetNotFoundError: Dataset 'walledai/AdvBench' is a gated dataset on the Hub.
Visit the dataset page at https://huggingface.co/datasets/walledai/AdvBench
to ask for access.
```

### 2. Root cause

Environmental, not our code. `walledai/AdvBench` and `walledai/HarmBench` are HuggingFace mirrors of the original AdvBench (Zou et al., 2023) and HarmBench ([RP44]) datasets. The maintainer recently flipped both to *gated* — meaning even with our HF token (which has "read gated repos" permission), the user must additionally accept each dataset's individual licence on the dataset page. Even after acceptance the propagation can take time.

The author-controlled GitHub repos remain freely downloadable.

### 3. Fix

Two-tier loader: try HF Hub first, fall back to authors' GitHub CSVs on `DatasetNotFoundError` / `GatedRepoError`. The GitHub CSVs are byte-identical to the HF mirrors:

| Dataset | HF Hub (gated) | GitHub fallback (open) |
|---|---|---|
| AdvBench | `walledai/AdvBench` | `llm-attacks/llm-attacks/.../harmful_behaviors.csv` |
| HarmBench | `walledai/HarmBench` | `centerforaisafety/HarmBench/.../harmbench_behaviors_text_all.csv` |
| XSTest | `natolambert/xstest-v2-copy` | (not gated, no fallback needed) |

### 4. Steps to verify

1. Re-run the dataset cell. Expected stdout depends on which licences the user accepted:
   - HF accepted → `AdvBench: loaded N from HF (walledai/AdvBench)`
   - HF not accepted → `AdvBench: HF gated (...); falling back to GitHub` then `loaded N from harmful_behaviors.csv`
2. Either way, `safety_dataset.csv` is written and synced to Drive.
3. `df_prompts.groupby(['source','expect']).size()` should show all three sources with their expected counts.

### 5. Citations

- Zou, A., Wang, Z., Carlini, N., Nasr, M., Kolter, J.Z. and Fredrikson, M. (2023) 'Universal and transferable adversarial attacks on aligned language models'. arXiv:2307.15043. AdvBench paper. Original repo: https://github.com/llm-attacks/llm-attacks
- Mazeika, M. et al. (2024) 'HarmBench: A standardized evaluation framework for automated red teaming and robust refusal', *ICML*. **[RP44]**. Original repo: https://github.com/centerforaisafety/HarmBench
- Röttger, P. et al. (2024) 'XSTest...', *NAACL*. (Not affected — XSTest mirror is open.)

### 6. Goal alignment

This fix unblocks the **entire experiment** — no safety prompts means no run loop. Specifically:

- **RQ1 / H1** — depend on AdvBench refusal-rate measurement.
- **RQ3** — depends on the AdvBench × XSTest contrast (genuine vs over-cautious refusal).
- **Rubric §6 (Dataset, 20%)** — gated.
- **Rubric §8 (Results, 20%)** — downstream.

The fallback architecture is also a methodological strength to claim explicitly in the paper's Section 6.1:

> *"Prompts are loaded with a two-tier strategy: HuggingFace mirror first (`walledai/AdvBench`, `walledai/HarmBench`), with fallback to the authors' open GitHub CSVs to ensure reproducibility independent of HF Hub gating decisions. The two sources are byte-identical."*

This converts a host-dependent dependency into a hardened, mirror-tolerant pipeline.

---

## 🐛 Fix #4 — `NameError: rng is not defined` in MMLU cell on cached-dataset path

**Date:** 2026-05-05
**Cells touched:** dataset construction cell (cell 19) — regression from Fix #3
**Severity:** blocker — MMLU utility evaluation cannot run without it

### 1. Original (broken) state

**Symptom (verbatim from cell 31 stderr):**
```
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_17054/2351800925.py in <cell line: 0>()
      1 # MMLU: 'cais/mmlu' all subjects; we sample N_MMLU questions across subjects.
      2 mmlu = load_dataset('cais/mmlu', 'all', split='test')
----> 3 idx = rng.choice(len(mmlu), size=N_MMLU, replace=False)
      4 mmlu_qs = [mmlu[int(i)] for i in idx]
      5 print('MMLU sample size:', len(mmlu_qs))

NameError: name 'rng' is not defined
```

### 2. Root cause

Logic regression introduced in Fix #3 (dataset two-tier loader). The dataset cell originally had `rng = np.random.default_rng(SEED)` at the top, before any branching. When Fix #3 added the `if (OUT/"safety_dataset.csv").exists(): ... else: ...` cache check, the `rng` line was moved *inside* the `else:` branch — so when the cached path is taken (which is normal after the first run), `rng` is never created. Cells 31 and 33 then reference `rng` at module scope and crash.

This is a self-inflicted code regression, not an environmental issue. Caught only by running end-to-end past cell 19.

### 3. Fix

Hoist `rng = np.random.default_rng(SEED)` out of the conditional, back to the top of the dataset cell:

```python
# Fix #4: rng must exist whether or not we hit the cache, since cells 31/33 use it.
rng = np.random.default_rng(SEED)

if (OUT/"safety_dataset.csv").exists():
    df_prompts = pd.read_csv(OUT/"safety_dataset.csv")
    ...
else:
    # ... build fresh and save ...
```

Cheap (microseconds), deterministic via `SEED=42`, idempotent across re-runs.

### 4. Steps to verify

1. Re-run cell 19 (dataset). Should still print `Loaded cached dataset: 550 rows` — the cache is preserved.
2. Re-run cell 31 (MMLU). Expected: `MMLU sample size: 1000` (or whatever `N_MMLU` is). No `NameError`.
3. Cell 33 (perplexity) should also succeed with the same `rng` available.

### 5. Citations

None — this is a self-inflicted regression. Reproducibility seed (`SEED=42`) is the only "external" reference, and it's already in the notebook.

### 6. Goal alignment

Unblocks the **MMLU utility evaluation** path:

- Without `rng`, the run loop crashes at the first MMLU call.
- No MMLU accuracy numbers → cannot compute `MMLU-drop%`.
- **H1** (refusal drops faster than MMLU at NF4) cannot be tested without the MMLU comparator.
- **Rubric §5 (Empirical Results, 10–20%)** — gated.

This fix also illustrates *why* the bug-fix-log discipline matters: the cross-reference to Fix #3 makes the regression cause obvious in the audit trail. Without that, the same bug could resurface every time we rewrite the dataset cell.

---

## 🐛 Fix #5 — `padding_side='left'` required for decoder-only generation

**Date:** 2026-05-05
**Cells touched:** model loader cell (cell 25)
**Severity:** **blocker (silent corruption)** — right-padding produces unreliable generations on shorter prompts in a batch, invalidating all refusal-rate numbers

### 1. Original (broken) state

**Symptom (warning during run loop):**
```
A decoder-only architecture is being used, but right-padding was detected!
For correct generation results, please set `padding_side='left'` when
initializing the tokenizer.
```

### 2. Root cause

Decoder-only LLMs (Llama, Qwen, GPT-style) generate at the rightmost token position. HuggingFace tokenizers default to `padding_side='right'`, which is correct for encoders (BERT, T5) but wrong for causal LMs.

| Padding | Effect on shorter prompt in a padded batch |
|---|---|
| `right` (default) | `[Hello world <pad> <pad>]` → generation starts at `<pad>`, model emits gibberish |
| `left` (correct) | `[<pad> <pad> Hello world]` → generation starts at `world`, model emits real continuation |

Our `load_model()` only set `tok.pad_token = tok.eos_token`; it never touched `padding_side`. Result: every shorter prompt in a batch produced unreliable text. The refusal regex would match or miss based on noise rather than model behaviour, and downstream refusal-rate numbers would be wrong.

This is **silent corruption**: no exception, no crash. Caught only by reading the warning and reasoning about its consequence.

### 3. Fix

One line added to `load_model()`:

```python
tok.padding_side = "left"
```

Inserted right after `tok.pad_token = tok.eos_token` so it always runs.

### 4. Steps to verify

1. Re-run cell 25 (loader definition).
2. **Delete any cached partial run** that used right-padding:
   ```python
   from pathlib import Path
   for f in OUT.glob('gen_*.csv'): f.unlink()
   for f in OUT.glob('res_*.json'): f.unlink()
   if DRIVE_OUT:
       for f in DRIVE_OUT.glob('gen_*.csv'): f.unlink()
       for f in DRIVE_OUT.glob('res_*.json'): f.unlink()
   ```
3. Re-run cell 37 (run loop). The padding warning should not appear.
4. Spot-check first generation: `df_run.iloc[0]['generation']` should be coherent (refusal or compliance), not gibberish or empty.

### 5. Citations

- HuggingFace `transformers` LLM tutorial, *padding side*: https://huggingface.co/docs/transformers/main/llm_tutorial
- Not a paper-level decision; library default-correction. But the consequence (invalid refusal rates) directly impacts comparability with [RP24] (Kirsten et al., 2024) and [RP26] (Hong et al., 2024).

### 6. Goal alignment

Without left-padding:

- **H1** (refusal drops faster than MMLU) — undefined; refusal axis is noise.
- **RQ3** (selective vs uniform anxiety) — both AdvBench and XSTest rates corrupted.
- **Rubric §8 (Results, 20%)** — invalid until rerun.

This fix is what makes the difference between *real* refusal-rate measurements and plausible-looking gibberish. **Any partial run prior to this fix must be discarded** — the verification step includes the cache-clear command. Documented in the paper's Methodology as: *"All generations produced with `padding_side='left'` to ensure decoder-only batched generation conditions on real prompt tokens rather than padding artefacts."*

---

## 🐛 Fix #6 — Sampling-param UserWarnings on every batch with `do_sample=False`

**Date:** 2026-05-05
**Cells touched:** loader cell (cell 25), generate_batch cell (cell 30)
**Severity:** cosmetic — output noise only, behaviour unchanged

### 1. Original (broken) state

**Symptom (verbatim, fires once per `model.generate(...)` call → ~3 warnings × 6 conditions × N batches):**
```
UserWarning: `do_sample` is set to `False`. However, `temperature` is set to
  `0.7` -- this flag is only used in sample-based generation modes.
UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.8` ...
UserWarning: `do_sample` is set to `False`. However, `top_k` is set to `20` ...
```

### 2. Root cause

Qwen2.5-1.5B-Instruct ships a `generation_config.json` on the Hub with chat-style sampling defaults (`temperature=0.7, top_p=0.8, top_k=20`). When the model loads, those become attributes of `model.generation_config`. Our `generate_batch()` correctly forces greedy decoding via `do_sample=False`, but the config validator notices the leftover sampling params and emits a UserWarning per param per batch.

Generation IS greedy (the warned params are unused), but the noise floods stdout and looks like a config bug to anyone reading the run-loop log.

### 3. Fix

Two coordinated edits, both belt-and-braces:

1. **In `load_model()`** — clear sampling defaults right after loading:
   ```python
   model.generation_config.do_sample = False
   model.generation_config.temperature = None
   model.generation_config.top_p = None
   model.generation_config.top_k = None
   ```
2. **In `generate_batch()`** — pass `temperature=None, top_p=None, top_k=None` to `model.generate(...)` so the call-site is self-documenting.

### 4. Steps to verify

1. Re-run cell 25 (loader def) and cell 30 (generate_batch def).
2. Re-run cell 37 (run loop). The three `UserWarning` lines should not appear at the start of any condition.
3. Spot-check determinism: `df_run.iloc[0]['generation']` should be byte-identical across two consecutive runs of the same condition (greedy means deterministic).

### 5. Citations

- HuggingFace `transformers` GenerationConfig docs: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
- Qwen2.5 model card: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct (lists chat-recommended sampling defaults).
- Convention of greedy decoding for evaluation: [RP09] GPTQ (Frantar et al., 2022), [RP15] AWQ (Lin et al., 2023) — both report greedy or beam=1 for perplexity/refusal evaluation, never sampling.

### 6. Goal alignment

Pure cosmetic + reproducibility-clarity fix:

- **No numeric impact** on H1 / RQ1 / refusal rate / MMLU / perplexity — generation was already greedy.
- **Reproducibility:** the paper's Section 7.3 ("Generation settings") claims `do_sample=False` — silencing the contradictory warnings makes that claim visibly true in the run log.
- **Lecture quality:** the run loop output for 6 conditions stays readable. Without the fix, ~50+ warning lines clutter the slide.

The fix also illustrates a wider lesson worth one sentence in the paper's Methodology: *"Default `generation_config.json` parameters from each model's Hub card are explicitly cleared at load time to prevent unintended interaction with greedy decoding."*
