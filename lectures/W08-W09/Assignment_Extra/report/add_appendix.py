# -*- coding: utf-8 -*-
"""Append Appendix A (figures) and Appendix B (numeric matrices + worked calc) to REPORT.md.
Idempotent: strips any previous appendix block first. Reads numbers from appendix.md."""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, 'REPORT.md')
APP_MD = os.path.join(HERE, 'appendix.md')

# ---- Appendix A: figure references ----
GRP = [('grp_A_ladder', 'the from-scratch depth ladder (2 to 6 conv layers)'),
       ('grp_B_reg', 'the regularised variants (dropout, batch-norm, augmentation)'),
       ('grp_C_modern', 'the modern blocks (VGG-style, residual, separable, GAP)'),
       ('grp_D_transfer', 'the transfer-learning backbones'),
       ('grp_E_tuning', 'the hyper-parameter tuning grid')]
CM = [('cm_ensemble_stacking_logreg', 'Stacking ensemble (champion)'),
      ('cm_ensemble_soft_vote_top3', 'Soft-vote ensemble'),
      ('cm_ensemble_hard_vote_top3', 'Hard-vote ensemble'),
      ('cm_resnet50_fine_tuned', 'ResNet50 (fine-tuned)'),
      ('cm_convnexttiny_frozen_tl', 'ConvNeXt-Tiny (frozen)'),
      ('cm_resnet50_frozen_tl', 'ResNet50 (frozen)')]

A = ['## Appendix A - Per-Family and Per-Model Figures', '',
     'This appendix gives readable, full-size versions of the multi-panel figures from the main text.', '',
     '### A.1 Validation-Accuracy Curves by Model Family', '']
n = 1
for fn, desc in GRP:
    A += ['![Validation curves](figures/appendix/%s.png)' % fn, '',
          '> **Figure A.%d:** Validation accuracy of %s.' % (n, desc), '']
    n += 1
A += ['### A.2 Confusion Matrices for the Top Six Models', '']
for fn, desc in CM:
    A += ['![Confusion matrix](figures/appendix/%s.png)' % fn, '',
          '> **Figure A.%d:** Confusion matrix (counts) - %s.' % (n, desc), '']
    n += 1

# ---- Appendix B: numeric matrices (abbreviated headers) + tutorial, from appendix.md ----
ABBR = {'airplane': 'plane', 'automobile': 'auto'}
def abbr(name): return ABBR.get(name, name)

raw = open(APP_MD, encoding='utf-8').read()
B = ['## Appendix B - Numeric Confusion Matrices and Worked Metric Calculation', '',
     '### B.1 Numeric Confusion Matrices (Counts)', '',
     'Each row is the true class; each column is the predicted class. The diagonal holds the correct predictions. Class names are abbreviated (plane = airplane, auto = automobile).', '']
# pull each "## Confusion matrix (counts): NAME" block and rebuild with abbreviated headers
for m in re.finditer(r'## Confusion matrix \(counts\): (.+?)\n\n(\| true.*?)(?=\n\n##|\n\n\Z|\Z)', raw, re.DOTALL):
    title, table = m.group(1).strip(), m.group(2).strip()
    lines = table.split('\n')
    hdr = lines[0]
    # abbreviate header + corner
    cells = [c.strip() for c in hdr.split('|')[1:-1]]
    cells = ['true/pred'] + [abbr(c) for c in cells[1:]]
    new_hdr = '| ' + ' | '.join(cells) + ' |'
    body = []
    for ln in lines[2:]:
        cc = [c.strip() for c in ln.split('|')[1:-1]]
        cc[0] = '**%s**' % abbr(cc[0].replace('**', ''))
        body.append('| ' + ' | '.join(cc) + ' |')
    B += ['**%s:**' % title, '', new_hdr, lines[1], *body, '']

# worked-example section (everything from "## Worked example" onward), re-headed as B.2
we = raw[raw.index('## Worked example'):]
we = we.replace('## Worked example - how the per-class metrics are calculated',
                '### B.2 Worked Example - Manual Calculation of Per-Class Metrics')
B += [we.strip(), '']

# ---- Appendix C: the assignment brief (verbatim wording, reproduced for reference) ----
C = [
    '## Appendix C - Assignment Brief', '',
    '*The original assignment instructions, reproduced for reference. The notebook and report above implement all seven tasks (T1-T7).*', '',
    '### Assignment: Image Classification Using CNN on a Large Dataset', '',
    '**Title:** Deep Learning-Based Image Classification Using Convolutional Neural Networks.', '',
    '**Objective:** The aim of this assignment is to design, train, and evaluate a Convolutional Neural Network (CNN) for classifying images from a large image dataset. Students will explore image preprocessing, CNN architecture design, model training, evaluation, and performance analysis.', '',
    '### Dataset', '', 'Use one large image dataset such as:', '',
    '- CIFAR-10', '- CIFAR-100', '- Fashion-MNIST', '- PlantVillage Dataset', '- Chest X-ray Dataset',
    '- Kaggle Animals-10 Dataset', '- Food-101 Dataset', '- Brain Tumor MRI Dataset', '- Traffic Signs Dataset', '',
    'The dataset should include:', '',
    '- At least 10,000 images', '- Multiple image classes', '- Training and testing sets', '',
    '### Task 1: Dataset Description', '',
    'Describe the selected image dataset, including the number of images, number of classes, image size, and examples of categories. Explain why this dataset is suitable for CNN classification.', '',
    '### Task 2: Image Preprocessing', '',
    'Apply suitable preprocessing techniques such as image resizing, normalization, train-test split, and data augmentation. Explain why preprocessing is important for CNN performance.', '',
    '### Task 3: CNN Model Design', '', 'Build a CNN model that includes:', '',
    '- Convolutional layers', '- Pooling layers', '- Activation function such as ReLU',
    '- Flatten layer', '- Dense fully connected layers', '- Output layer with Softmax activation', '',
    '### Task 4: Model Training', '',
    'Train the CNN model using the training dataset. Select suitable hyperparameters such as batch size, number of epochs, optimizer, and learning rate.', '',
    '### Task 5: Model Evaluation and Architecture Comparison', '', 'Evaluate the CNN model using:', '',
    '- Accuracy', '- Precision', '- Recall', '- F1-score', '- Confusion matrix',
    '- Training and validation accuracy curves', '- Training and validation loss curves', '',
    'In addition, students must perform an architecture comparison by gradually increasing the number of hidden layers in the CNN model and evaluating the performance each time. Students should train and compare at least three CNN architectures:', '',
    '| Model | CNN Architecture |',
    '|-------|------------------|',
    '| Model 1 | Basic CNN with 2 convolutional layers |',
    '| Model 2 | CNN with 3-4 convolutional layers |',
    '| Model 3 | Deeper CNN with 5 or more convolutional layers |', '',
    'For each model, students should report: training accuracy, validation accuracy, test accuracy, training loss, validation loss, precision, recall, and F1-score.', '',
    '### Task 6: Results Discussion', '',
    'Discuss the model performance. Identify which classes were classified correctly and which classes caused confusion. Explain possible reasons for misclassification.', '',
    '### Task 7: Improvement', '', 'Improve the CNN model using one or more of the following:', '',
    '- Data augmentation', '- Dropout', '- Batch normalization',
    '- Transfer learning using VGG16, ResNet50, or MobileNet', '- Hyperparameter tuning', '',
    '### Final Deliverables', '', 'Students must submit:', '',
    '- A written report.', '- Python code or Jupyter Notebook.', '- Dataset source link.',
    '- Model evaluation results.', '- Graphs and confusion matrix.', '',
]

block = '\n'.join(['', '<!-- APPENDIX START -->', ''] + A + B + C + ['<!-- APPENDIX END -->'])

text = open(REPORT, encoding='utf-8').read()
text = re.sub(r'\n<!-- APPENDIX START -->.*?<!-- APPENDIX END -->', '', text, flags=re.DOTALL)  # idempotent
open(REPORT, 'w', encoding='utf-8').write(text.rstrip() + '\n' + block + '\n')
print('Appended Appendix A (%d figures) + Appendix B (numeric + worked calc) to REPORT.md' % (len(GRP) + len(CM)))
