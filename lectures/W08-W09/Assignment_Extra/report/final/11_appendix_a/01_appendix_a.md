## Appendix A - Per-Family and Per-Model Figures

This appendix gives readable, full-size versions of the multi-panel figures from the main text.

### A.1 Validation-Accuracy Curves by Model Family

![Validation curves](../../figures/appendix/grp_A_ladder.png)

> **Figure A.1:** Validation accuracy of the from-scratch depth ladder (2 to 6 conv layers).

![Validation curves](../../figures/appendix/grp_B_reg.png)

> **Figure A.2:** Validation accuracy of the regularised variants (dropout, batch-norm, augmentation).

![Validation curves](../../figures/appendix/grp_C_modern.png)

> **Figure A.3:** Validation accuracy of the modern blocks (VGG-style, residual, separable, GAP).

![Validation curves](../../figures/appendix/grp_D_transfer.png)

> **Figure A.4:** Validation accuracy of the transfer-learning backbones.

![Validation curves](../../figures/appendix/grp_E_tuning.png)

> **Figure A.5:** Validation accuracy of the hyper-parameter tuning grid.

### A.2 Confusion Matrices for the Top Six Models

![Confusion matrix](../../figures/appendix/cm_ensemble_stacking_logreg.png)

> **Figure A.6:** Confusion matrix (counts) - Stacking ensemble (champion).

![Confusion matrix](../../figures/appendix/cm_ensemble_soft_vote_top3.png)

> **Figure A.7:** Confusion matrix (counts) - Soft-vote ensemble.

![Confusion matrix](../../figures/appendix/cm_ensemble_hard_vote_top3.png)

> **Figure A.8:** Confusion matrix (counts) - Hard-vote ensemble.

![Confusion matrix](../../figures/appendix/cm_resnet50_fine_tuned.png)

> **Figure A.9:** Confusion matrix (counts) - ResNet50 (fine-tuned).

![Confusion matrix](../../figures/appendix/cm_convnexttiny_frozen_tl.png)

> **Figure A.10:** Confusion matrix (counts) - ConvNeXt-Tiny (frozen).

![Confusion matrix](../../figures/appendix/cm_resnet50_frozen_tl.png)

> **Figure A.11:** Confusion matrix (counts) - ResNet50 (frozen).
