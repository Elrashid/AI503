## Abstract

This report studies image classification with Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset. The assignment asks for three CNNs of growing depth. This study goes further. It places twenty-six models on one leaderboard and changes only the architecture each time. The models range from a two-layer baseline to fine-tuned ImageNet backbones and stacked ensembles. A shared evaluation routine scores every model with the same metrics, so the comparison is fair. The best single network was a fine-tuned ResNet50 at 0.922 macro F1. A stacking ensemble of the three strongest backbones reached 0.940 macro F1 and 0.997 ROC-AUC. The from-scratch depth study showed accuracy rising to four convolutional layers, then falling. This confirms that raw depth alone does not help on small images. Transfer learning gave the largest gains, and class "cat" stayed the hardest category throughout.

---
