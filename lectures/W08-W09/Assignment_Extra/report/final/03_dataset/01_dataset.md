## 2. Dataset (Task 1)

CIFAR-10 is a standard benchmark of 60,000 colour images (Krizhevsky, 2009 [R7]). Each image is 32×32 pixels with three colour channels. The data splits into 50,000 training images and 10,000 test images. There are ten classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck (Krizhevsky, 2009 [R7]).

![Example images, one per CIFAR-10 class.](../../figures/fig_classes.png)

> **Figure 2.1:** Example images, one per CIFAR-10 class.








The classes are perfectly balanced. The run confirmed 5,000 training images for each of the ten classes. The set also has a fixed train and test split by design, so there is no leakage between them.

Why does this dataset suit a CNN? The pictures are small and natural, and nearby pixels belong together. A CNN reads exactly this kind of local structure (Course Notes, W05–W06, p.18 [R16]). The dataset is large enough to train deep models, yet small enough to run many experiments in one session. It is also a well-known benchmark, so the results are easy to place against published work.

This study carved a validation set from the training data. The final split was 45,000 training, 5,000 validation, and 10,000 test images. The validation set guided early stopping and tuning. The test set stayed untouched until the final score.

---
