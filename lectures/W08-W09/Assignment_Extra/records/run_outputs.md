# Run outputs - CNN_Image_Classification_CIFAR10.ipynb

**31 code cells executed | 0 errors | 12 figures**

## Cell 04 — CONTROL PANEL  (exec 1)

```
Control panel set. QUICK_MODE = False | MIXED_PRECISION = True
```

## Cell 06 — §1.1 · Setup — import the tools and make results repeatable  (exec 2)

```
Mixed precision ON (float16).
TensorFlow 2.20.0 | GPU available: True | [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Cell 08 — §2 · Task 1 — The dataset (CIFAR-10)  (exec 3)

```
Image shape (height, width, colour channels): (32, 32, 3)
Pixel value range: 0 to 255 (0=black, 255=brightest)
Training images: 50000
Test images    : 10000
Total images   : 60000
Number of classes: 10 -> ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

Images per class in the training set:
  airplane    : 5000
  automobile  : 5000
  bird        : 5000
  cat         : 5000
  deer        : 5000
  dog         : 5000
  frog        : 5000
  horse       : 5000
  ship        : 5000
  truck       : 5000

CIFAR-10 is perfectly balanced (5,000 of each) and has no train/test leakage by construction.
```

## Cell 09 — Show one example image from every class (Task 1 asks for cat  (exec 4)

```
<Figure size 1100x500 with 10 Axes>
```

*[figure rendered]*

## Cell 11 — §3 · Task 2 — Preprocessing  (exec 5)

```
Train images     : 45000
Validation images: 5000
Test images      : 10000
Label shape (one-hot): (45000, 10) -> 10 numbers per image
```

## Cell 12 — Build the augmentation pipeline as Keras layers. These layer  (exec 6)

```
<Figure size 1300x260 with 6 Axes>
```

*[figure rendered]*

## Cell 14 — §3.1 · The shared scoreboard and training helper (the heart of the notebook)  (exec 7)

```
Scoreboard ready. Each model will print one line and store one row.
```

## Cell 16 — The parametric from-scratch CNN factory used for the whole d  (exec 8)

```
Model: "cnn_4blocks"
```

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 32, 32, 32)     │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 16, 16, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 16, 16, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 8, 8, 64)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 8, 8, 128)      │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 4, 4, 128)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (Conv2D)               │ (None, 4, 4, 256)      │       295,168 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_3 (MaxPooling2D)  │ (None, 2, 2, 256)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 1024)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 128)            │       131,200 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 10)             │         1,290 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
```

```
 Total params: 520,906 (1.99 MB)
```

```
 Trainable params: 520,906 (1.99 MB)
```

```
 Non-trainable params: 0 (0.00 B)
```

## Cell 18 — Train the depth ladder. We keep the 4-conv model in memory f  (exec 9)

```
CNN-2conv                          test_acc=0.704  F1=0.706  AUC=0.956  (32.1s)
CNN-3conv                          test_acc=0.732  F1=0.730  AUC=0.962  (27.2s)
CNN-4conv                          test_acc=0.735  F1=0.732  AUC=0.964  (27.5s)
CNN-5conv                          test_acc=0.700  F1=0.695  AUC=0.956  (27.7s)
CNN-6conv                          test_acc=0.704  F1=0.705  AUC=0.957  (29.7s)
```

## Cell 19 — Chart 1: accuracy vs depth (the headline picture for Task 5'  (exec 10)

```
<Figure size 800x500 with 1 Axes>
```

*[figure rendered]*

## Cell 20 — Chart 2: train & validation accuracy and loss curves for the  (exec 11)

```
<Figure size 1300x460 with 2 Axes>
```

*[figure rendered]*

## Cell 22 — §6 · Group B — regularisation ablations (Task 7, part 1)  (exec 12)

```
Deep+Dropout                       test_acc=0.741  F1=0.742  AUC=0.964  (35.1s)
Deep+BatchNorm                     test_acc=0.707  F1=0.707  AUC=0.955  (46.4s)
Deep+Augment                       test_acc=0.770  F1=0.766  AUC=0.975  (82.6s)
Deep+All(BN+Drop+Aug)              test_acc=0.801  F1=0.801  AUC=0.979  (127.2s)
```

## Cell 24 — §7 · Group C — modern CNN blocks, built from scratch  (exec 13)

```
VGG-style (2xconv blocks)          test_acc=0.765  F1=0.766  AUC=0.970  (56.2s)
Residual-mini (skip conn.)         test_acc=0.754  F1=0.755  AUC=0.972  (96.0s)
Depthwise-separable                test_acc=0.732  F1=0.727  AUC=0.964  (66.9s)
All-conv + GAP                     test_acc=0.771  F1=0.772  AUC=0.970  (56.3s)
```

## Cell 26 — Load whichever backbones this TensorFlow version provides (s  (exec 14)

```
Backbones available: ['VGG16', 'ResNet50', 'MobileNetV2', 'EfficientNetV2S', 'ConvNeXtTiny']
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
58889256/58889256 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step
VGG16 (frozen TL)                  test_acc=0.842  F1=0.840  AUC=0.987  (111.9s)
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
94765736/94765736 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step
ResNet50 (frozen TL)               test_acc=0.889  F1=0.889  AUC=0.993  (126.1s)
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5
9406464/9406464 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step
MobileNetV2 (frozen TL)            test_acc=0.858  F1=0.858  AUC=0.989  (74.2s)
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/efficientnet_v2/efficientnetv2-s_notop.h5
82420632/82420632 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step
EfficientNetV2S (frozen TL)        test_acc=0.875  F1=0.875  AUC=0.991  (195.0s)
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/convnext/convnext_tiny_notop.h5
111650432/111650432 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step
ConvNeXtTiny (frozen TL)           test_acc=0.915  F1=0.915  AUC=0.996  (337.4s)
```

## Cell 27 — Fine-tune ONE champion backbone (ResNet50): phase-1 frozen h  (exec 15)

```
ResNet50 (fine-tuned)              test_acc=0.922  F1=0.922  AUC=0.996  (518.2s)
```

## Cell 29 — §9 · Group E — hyper-parameter tuning (Task 4 / Task 7)  (exec 16)

```
Grid lr=1e-03 drop=0.3             test_acc=0.748  F1=0.747  AUC=0.967  (38.0s)
Grid lr=1e-03 drop=0.5             test_acc=0.746  F1=0.748  AUC=0.967  (35.4s)
Grid lr=5e-04 drop=0.3             test_acc=0.733  F1=0.729  AUC=0.963  (35.1s)
Grid lr=5e-04 drop=0.5             test_acc=0.754  F1=0.753  AUC=0.968  (43.9s)
```

## Cell 31 — §10 · Group F — ensembles: combine the best models  (exec 17)

```
Combining top-3 models: ['ResNet50 (fine-tuned)', 'ConvNeXtTiny (frozen TL)', 'ResNet50 (frozen TL)'] 

Ensemble: Soft-Vote (top3)         test_acc=0.934  F1=0.934
Ensemble: Hard-Vote (top3)         test_acc=0.930  F1=0.930
Ensemble: Stacking (LogReg)        test_acc=0.940  F1=0.940
```

## Cell 33 — THE LEADERBOARD — one row per model, sorted by macro-F1  (exec 18)

```
26 models on the leaderboard. Champion: Ensemble: Stacking (LogReg)
```

```
                                  group  train_acc  val_acc  test_acc  \
model                                                                   
Ensemble: Stacking (LogReg)  F_ensemble        NaN      NaN    0.9398   
Ensemble: Soft-Vote (top3)   F_ensemble        NaN      NaN    0.9341   
Ensemble: Hard-Vote (top3)   F_ensemble        NaN      NaN    0.9300   
ResNet50 (fine-tuned)        D_transfer     0.9993   0.9348    0.9220   
ConvNeXtTiny (frozen TL)     D_transfer     0.9180   0.9194    0.9150   
ResNet50 (frozen TL)         D_transfer     0.8976   0.8940    0.8895   
EfficientNetV2S (frozen TL)  D_transfer     0.8630   0.8806    0.8752   
MobileNetV2 (frozen TL)      D_transfer     0.8581   0.8636    0.8581   
VGG16 (frozen TL)            D_transfer     0.8126   0.8428    0.8415   
Deep+All(BN+Drop+Aug)             B_reg     0.8566   0.8094    0.8012   
All-conv + GAP                 C_modern     1.0000   0.7768    0.7710   
VGG-style (2xconv blocks)      C_modern     0.9663   0.8072    0.7654   
Deep+Augment                      B_reg     0.8223   0.7820    0.7698   
Residual-mini (skip conn.)     C_modern     0.9913   0.7800    0.7540   
Grid lr=5e-04 drop=0.5         E_tuning     0.8959   0.7688    0.7538   
Grid lr=1e-03 drop=0.5         E_tuning     0.9219   0.7694    0.7456   
Grid lr=1e-03 drop=0.3         E_tuning     0.9622   0.7656    0.7482   
Deep+Dropout                      B_reg     0.9686   0.7538    0.7413   
CNN-4conv                      A_ladder     0.9521   0.7330    0.7349   
CNN-3conv                      A_ladder     0.8913   0.7384    0.7320   
Grid lr=5e-04 drop=0.3         E_tuning     0.8693   0.7618    0.7327   
Depthwise-separable            C_modern     0.9336   0.7484    0.7317   
Deep+BatchNorm                    B_reg     0.9998   0.7842    0.7071   
CNN-2conv                      A_ladder     0.8817   0.7144    0.7041   
CNN-6conv                      A_ladder     0.9487   0.7198    0.7044   
CNN-5conv                      A_ladder     0.9564   0.7530    0.7001   

                             train_loss  val_loss  precision  recall      f1  \
model                                                                          
Ensemble: Stacking (LogReg)         NaN       NaN     0.9399  0.9398  0.9398   
Ensemble: Soft-Vote (top3)          NaN       NaN     0.9344  0.9341  0.9341   
Ensemble: Hard-Vote (top3)          NaN       NaN     0.9306  0.9300  0.9300   
ResNet50 (fine-tuned)            0.0071    0.2336     0.9221  0.9220  0.9220   
ConvNeXtTiny (frozen TL)         0.2460    0.2423     0.9152  0.9150  0.9149   
ResNet50 (frozen TL)             0.2907    0.3167     0.8909  0.8895  0.8894   
EfficientNetV2S (frozen TL)      0.4038    0.3598     0.8753  0.8752  0.8750   
MobileNetV2 (frozen TL)          0.4065    0.3968     0.8593  0.8581  0.8579   
VGG16 (frozen TL)                0.5418    0.4565     0.8425  0.8415  0.8403   
Deep+All(BN+Drop+Aug)            0.4151    0.5863     0.8042  0.8012  0.8010   
All-conv + GAP                   0.0025    0.9178     0.7744  0.7710  0.7721   
VGG-style (2xconv blocks)        0.0955    0.8742     0.7751  0.7654  0.7663   
Deep+Augment                     0.5006    0.6733     0.7772  0.7698  0.7656   
Residual-mini (skip conn.)       0.0592    0.9110     0.7694  0.7540  0.7545   
Grid lr=5e-04 drop=0.5           0.3035    0.7654     0.7539  0.7538  0.7530   
Grid lr=1e-03 drop=0.5           0.2273    0.9148     0.7530  0.7456  0.7477   
Grid lr=1e-03 drop=0.3           0.1169    1.0025     0.7557  0.7482  0.7475   
Deep+Dropout                     0.1001    1.2144     0.7485  0.7413  0.7420   
CNN-4conv                        0.1591    1.1338     0.7353  0.7349  0.7319   
CNN-3conv                        0.3313    0.8521     0.7336  0.7320  0.7304   
Grid lr=5e-04 drop=0.3           0.3772    0.7516     0.7312  0.7327  0.7289   
Depthwise-separable              0.2358    0.8363     0.7379  0.7317  0.7267   
Deep+BatchNorm                   0.0034    1.0681     0.7219  0.7071  0.7073   
CNN-2conv                        0.3651    0.9432     0.7135  0.7041  0.7059   
CNN-6conv                        0.1530    1.2896     0.7236  0.7044  0.7047   
CNN-5conv                        0.1340    1.1952     0.7054  0.7001  0.6947   

                             roc_auc  seconds  
model                                          
Ensemble: Stacking (LogReg)   0.9968      0.0  
Ensemble: Soft-Vote (top3)    0.9973      0.0  
Ensemble: Hard-Vote (top3)       NaN      0.0  
ResNet50 (fine-tuned)         0.9961    518.2  
ConvNeXtTiny (frozen TL)      0.9956    337.4  
ResNet50 (frozen TL)          0.9933    126.1  
EfficientNetV2S (frozen TL)   0.9911    195.0  
MobileNetV2 (frozen TL)       0.9894     74.2  
VGG16 (frozen TL)             0.9865    111.9  
Deep+All(BN+Drop+Aug)         0.9794    127.2  
All-conv + GAP                0.9698     56.3  
VGG-style (2xconv blocks)     0.9704     56.2  
Deep+Augment                  0.9746     82.6  
Residual-mini (skip conn.)    0.9717     96.0  
Grid lr=5e-04 drop=0.5        0.9681     43.9  
Grid lr=1e-03 drop=0.5        0.9670     35.4  
Grid lr=1e-03 drop=0.3        0.9670     38.0  
Deep+Dropout                  0.9635     35.1  
CNN-4conv                     0.9644     27.5  
CNN-3conv                     0.9624     27.2  
Grid lr=5e-04 drop=0.3        0.9632     35.1  
Depthwise-separable           0.9642     66.9  
Deep+BatchNorm                0.9551     46.4  
CNN-2conv                     0.9558     32.1  
CNN-6conv                     0.9569     29.7  
CNN-5conv                     0.9559     27.7
```

## Cell 34 — F1 bar chart (grey = plain from-scratch, blue = transfer / t  (exec 19)

```
<Figure size 1100x884 with 1 Axes>
```

*[figure rendered]*

## Cell 35 — All-metrics heatmap  (exec 20)

```
<Figure size 800x1192 with 1 Axes>
```

*[figure rendered]*

## Cell 36 — ROC overlay (micro-averaged, one readable curve per model) f  (exec 21)

```
<Figure size 800x700 with 1 Axes>
```

*[figure rendered]*

## Cell 37 — Train/validation accuracy curves, one panel per group (kept  (exec 22)

```
<Figure size 1300x1200 with 6 Axes>
```

*[figure rendered]*

## Cell 38 — Confusion-matrix grid for the top-6 models  (exec 23)

```
<Figure size 1400x800 with 6 Axes>
```

*[figure rendered]*

## Cell 40 — §12 · Task 6 — results discussion (which classes, and why)  (exec 24)

```
CHAMPION MODEL: Ensemble: Stacking (LogReg) 

Per-class report (precision / recall / F1 for each of the 10 classes):

              precision    recall  f1-score   support

    airplane      0.936     0.960     0.948      1000
  automobile      0.961     0.965     0.963      1000
        bird      0.948     0.929     0.938      1000
         cat      0.869     0.881     0.875      1000
        deer      0.927     0.930     0.929      1000
         dog      0.910     0.895     0.902      1000
        frog      0.962     0.959     0.960      1000
       horse      0.960     0.956     0.958      1000
        ship      0.964     0.969     0.967      1000
       truck      0.963     0.954     0.958      1000

    accuracy                          0.940     10000
   macro avg      0.940     0.940     0.940     10000
weighted avg      0.940     0.940     0.940     10000
```

## Cell 41 — Per-class accuracy bar for the champion (diagonal of its con  (exec 25)

```
<Figure size 900x500 with 1 Axes>
```

*[figure rendered]*

```
Most common single mistake: a real "cat" predicted as "dog" (67 times).
```

## Cell 42 — Misclassification gallery: 10 test images the champion got W  (exec 26)

```
<Figure size 1200x500 with 10 Axes>
```

*[figure rendered]*

## Cell 43 — Grad-CAM: colour the pixels the CNN used to decide. Works on  (exec 27)

```
<Figure size 1200x500 with 10 Axes>
```

*[figure rendered]*

## Cell 45 — §13 · Task 7 — what each improvement was worth  (exec 28)

```
Baseline = CNN-5conv (F1 = 0.6947)
```

```
                      technique      F1  delta_vs_baseline
0   Ensemble: Stacking (LogReg)  0.9398             0.2451
1    Ensemble: Soft-Vote (top3)  0.9341             0.2394
2    Ensemble: Hard-Vote (top3)  0.9300             0.2353
3         ResNet50 (fine-tuned)  0.9220             0.2273
4      ConvNeXtTiny (frozen TL)  0.9149             0.2202
5          ResNet50 (frozen TL)  0.8894             0.1947
6   EfficientNetV2S (frozen TL)  0.8750             0.1803
7       MobileNetV2 (frozen TL)  0.8579             0.1632
8             VGG16 (frozen TL)  0.8403             0.1456
9         Deep+All(BN+Drop+Aug)  0.8010             0.1063
10    VGG-style (2xconv blocks)  0.7663             0.0716
11                 Deep+Augment  0.7656             0.0709
12   Residual-mini (skip conn.)  0.7545             0.0598
13                 Deep+Dropout  0.7420             0.0473
14               Deep+BatchNorm  0.7073             0.0126
```

## Cell 47 — §14 · Post-run analysis — expectation vs reality  (exec 29)

```
============================================================
AUTOMATIC POST-RUN CHECK
============================================================
Top 5 models by F1:
                             test_acc      f1  roc_auc
model                                                 
Ensemble: Stacking (LogReg)    0.9398  0.9398   0.9968
Ensemble: Soft-Vote (top3)     0.9341  0.9341   0.9973
Ensemble: Hard-Vote (top3)     0.9300  0.9300      NaN
ResNet50 (fine-tuned)          0.9220  0.9220   0.9961
ConvNeXtTiny (frozen TL)       0.9150  0.9149   0.9956 

Best FROM-SCRATCH : Deep+All(BN+Drop+Aug)  (F1=0.801)
Best TRANSFER     : ResNet50 (fine-tuned)  (F1=0.922)
Prediction "transfer beats from-scratch": CONFIRMED

Depth check: 6-conv minus 4-conv F1 = -0.027 -> deeper did NOT help (plateau CONFIRMED)

Overall champion: Ensemble: Stacking (LogReg)  (F1=0.940)
```

## Cell 50 — Save the leaderboard so it can go in the report as the evalu  (exec 30)

```
Saved cnn_leaderboard_results.csv

================ ASSIGNMENT COVERAGE ================
  [COVERED]  Task 1 — dataset described
  [COVERED]  Task 2 — preprocessing + augmentation
  [COVERED]  Task 3 — CNN designed (conv/pool/.../softmax)
  [COVERED]  Task 4 — training settings (Adam/lr/batch/epochs)
  [COVERED]  Task 5 — >=3 CNNs of increasing depth compared
  [COVERED]  Task 6 — per-class + confusion + Grad-CAM discussion
  [COVERED]  Task 7 — improvements (augment/dropout/BN/transfer/tuning)
====================================================

Total models trained: 26   |   Champion: Ensemble: Stacking (LogReg)

DELIVERABLES CHECKLIST:
  [ ] Written report (use the §14 discussion + the leaderboard table)
  [ ] This notebook (.ipynb), runs top-to-bottom on a GPU
  [ ] Dataset link: CIFAR-10 (Keras built-in / https://www.cs.toronto.edu/~kriz/cifar.html)
  [ ] Evaluation results: cnn_leaderboard_results.csv (saved above)
  [ ] Graphs: accuracy-vs-depth, F1 bar, heatmap, ROC, curves
  [ ] Confusion matrices: §11 top-6 grid + §12 champion
```

## Cell 52 — Reload the best SAVED model and predict on new images (needs  (exec 31)

```
SAVE_WEIGHTS is False -> no full weights were saved. Set it True in the control panel and re-run to enable model reuse.
```
