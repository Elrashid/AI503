# -*- coding: utf-8 -*-
"""Embed figures into REPORT.md, then split it into draft/ section files and mirror to final/."""
import os, re, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, 'REPORT.md')

# ---- 1. Figure insertions: (unique anchor already in REPORT.md, [(png_name, fig_number, caption), ...]) ----
INSERTS = [
    ("airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck",
     [("fig_classes", "2.1", "Example images, one per CIFAR-10 class.")]),
    ("so it cannot simply memorise the training set",
     [("fig_augmentation", "3.1", "One training image shown with five random augmentations.")]),
    ("the six-layer F1 minus the four-layer F1 was",
     [("fig_accuracy_vs_depth", "4.1", "Test accuracy against the number of convolutional layers.")]),
    ("the gap between training and validation accuracy widened with depth",
     [("fig_curves_accuracy", "4.2", "Training and validation accuracy for the three required CNNs (Models 1-3)."),
      ("fig_curves_loss", "4.3", "Training and validation loss for the three required CNNs (Models 1-3).")]),
    ("features learned on the large ImageNet dataset",
     [("fig_f1_bar", "4.4", "Macro F1 of all 26 models (grey = from-scratch, blue = transfer / tuned / ensemble).")]),
    ("so it never saw the test answers in advance",
     [("fig_metrics_heatmap", "4.5", "Heatmap of every metric for every model."),
      ("fig_roc", "4.6", "Micro-averaged ROC curves for the top eight models."),
      ("fig_group_curves", "4.7", "Validation-accuracy curves grouped by model family.")]),
    ("followed by dog at 0.902",
     [("fig_per_class_acc", "5.1", "Per-class accuracy of the champion ensemble.")]),
    ("it reflects the data, not the model size",
     [("fig_confusion_grid", "5.2", "Confusion matrices for the top six models.")]),
    ("the network groups images by appearance",
     [("fig_misclassified", "5.3", "Ten test images the champion misclassified (true vs predicted).")]),
    ("whether the network looks at the object or at the background",
     [("fig_gradcam", "6.1", "Grad-CAM heatmaps showing the pixels that drove each prediction.")]),
]

def fig_block(figs, prefix='figures'):
    out = []
    for name, num, cap in figs:
        out.append('![%s](%s/%s.png)\n\n> **Figure %s:** %s' % (cap, prefix, name, num, cap))
    return '\n\n'.join(out)

text = open(REPORT, encoding='utf-8').read()
# strip any previously-inserted figure blocks so the script is idempotent
text = re.sub(r'\n!\[[^\]]*\]\(figures/fig_[^)]+\)\n\n> \*\*Figure [^\n]+', '', text)  # main figs only (not figures/appendix/)

for anchor, figs in INSERTS:
    idx = text.find(anchor)
    if idx == -1:
        raise SystemExit('ANCHOR NOT FOUND: ' + anchor[:60])
    para_end = text.find('\n\n', idx)
    if para_end == -1:
        para_end = len(text)
    block = fig_block(figs)
    text = text[:para_end] + '\n\n' + block + text[para_end:]

open(REPORT, 'w', encoding='utf-8').write(text)
print('Figures embedded into REPORT.md:', sum(len(f) for _, f in INSERTS))

# ---- 2. Split into section files ----
SECTIONS = [
    ('## Abstract',                          '01_abstract/01_abstract.md',          'Abstract'),
    ('## 1. Introduction',                   '02_introduction/01_introduction.md',  '1. Introduction and Objectives'),
    ('## 2. Dataset',                        '03_dataset/01_dataset.md',            '2. Dataset'),
    ('## 3. Methodology',                    '04_methodology/01_methodology.md',    '3. Methodology'),
    ('## 4. Results',                        '05_results/01_results.md',            '4. Results'),
    ('## 5. Discussion',                     '06_discussion/01_discussion.md',      '5. Discussion'),
    ('## 6. Improvement',                    '07_improvement/01_improvement.md',    '6. Improvement Analysis'),
    ('## 7. Limitations',                    '08_limitations/01_limitations.md',    '7. Limitations and Threats to Validity'),
    ('## 8. Conclusion',                     '09_conclusion/01_conclusion.md',      '8. Conclusion'),
    ('## References',                        '10_references/01_references.md',      'References'),
    ('## Appendix A',                        '11_appendix_a/01_appendix_a.md',     'Appendix A - Figures'),
    ('## Appendix B',                        '12_appendix_b/01_appendix_b.md',     'Appendix B - Numeric matrices and worked calculation'),
    ('## Appendix C',                        '13_appendix_c/01_appendix_c.md',     'Appendix C - Assignment brief'),
]

# locate each section start
positions = []
for prefix, path, title in SECTIONS:
    p = text.find('\n' + prefix)
    if p == -1 and text.startswith(prefix):
        p = 0
    if p == -1:
        raise SystemExit('SECTION NOT FOUND: ' + prefix)
    positions.append(p)
positions.append(len(text))

draft = os.path.join(HERE, 'draft')
final = os.path.join(HERE, 'final')

# Write section files into BOTH draft/ and final/ (overwrite in place; no rmtree -> avoids OneDrive locks).
# Note: re-running this overwrites final/ too, so finish manual edits to final/ AFTER the last build.
toc = ['# Table of Contents\n']
for i, (prefix, path, title) in enumerate(SECTIONS):
    chunk = text[positions[i]:positions[i + 1]].strip()
    chunk = chunk.replace('](figures/', '](../../figures/')   # fix relative path for nested files
    chunk = re.sub(r'\n-{3,}\n', '\n', chunk)                  # drop horizontal rules
    for folder in (draft, final):
        fp = os.path.join(folder, path)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        open(fp, 'w', encoding='utf-8').write(chunk + '\n')
    toc.append('- [%s](%s)' % (title, path))
for folder in (draft, final):
    open(os.path.join(folder, 'TOC.md'), 'w', encoding='utf-8').write('\n'.join(toc) + '\n')
print('Wrote %d section files + TOC.md into draft/ and final/' % len(SECTIONS))
