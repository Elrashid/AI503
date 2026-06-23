# Sentence-by-sentence review — REPORT.md (body prose)

**142 prose sentences** | FACT(cited): 18 | RESULT(own data): 23 | ASSUME(hedged): 4 | CK: 97

| # | Type | Cite | Words | Sentence |
|--:|------|:----:|:----:|----------|
| 1 | CK | - | 14 | This report studies image classification with Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset. |
| 2 | CK | - | 9 | The assignment asks for three CNNs of growing depth. |
| 3 | CK | - | 4 | This study goes further. |
| 4 | CK | - | 14 | It places twenty-six models on one leaderboard and changes only the architecture each time. |
| 5 | CK | - | 14 | The models range from a two-layer baseline to fine-tuned ImageNet backbones and stacked ensembles. |
| 6 | CK | - | 16 | A shared evaluation routine scores every model with the same metrics, so the comparison is fair. |
| 7 | RESULT | - | 12 | The best single network was a fine-tuned ResNet50 at 0.922 macro F1. |
| 8 | RESULT | - | 15 | A stacking ensemble of the three strongest backbones reached 0.940 macro F1 and 0.997 ROC-AUC. |
| 9 | CK | - | 13 | The from-scratch depth study showed accuracy rising to four convolutional layers, then falling. |
| 10 | CK | - | 12 | This confirms that raw depth alone does not help on small images. |
| 11 | CK | - | 14 | Transfer learning gave the largest gains, and class "cat" stayed the hardest category throughout. |
| 12 | CK | - | 12 | Image classification asks a model to assign one label to a picture. |
| 13 | CK | - | 8 | It is a core task in computer vision. |
| 14 | FACT | Y | 21 | CNNs are the standard tool for this task because they read the spatial structure of an image (LeCun et al., 1998). |
| 15 | CK | - | 15 | A dense network treats every pixel as independent and needs a huge number of weights. |
| 16 | FACT | Y | 20 | A CNN instead slides small filters across the image, so it shares weights and stays compact (Course Notes, W05–W06, p.18 |
| 17 | CK | - | 7 | The assignment has a clear core requirement. |
| 18 | CK | - | 13 | Students must build at least three CNNs of increasing depth and compare them. |
| 19 | CK | - | 15 | They must also apply preprocessing, train the models, evaluate them, and try one improvement technique. |
| 20 | CK | - | 11 | This study treats that requirement as a floor, not a ceiling. |
| 21 | CK | - | 16 | The main objective is to build a single, fair comparison of many architectures on one dataset. |
| 22 | CK | - | 17 | The design copies the style of the Week-7 model-comparison notebook, where many algorithms shared one results table. |
| 23 | CK | - | 15 | The secondary objective is to measure each improvement separately, so its true value is visible. |
| 24 | CK | - | 13 | A final objective is reproducibility, so every number can be traced and repeated. |
| 25 | FACT | Y | 11 | CIFAR-10 is a standard benchmark of 60,000 colour images (Krizhevsky, 2009). |
| 26 | RESULT | - | 9 | Each image is 32×32 pixels with three colour channels. |
| 27 | RESULT | - | 11 | The data splits into 50,000 training images and 10,000 test images. |
| 28 | FACT | Y | 17 | There are ten classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck (Krizhevsky, 2009). |
| 29 | CK | - | 5 | The classes are perfectly balanced. |
| 30 | RESULT | - | 12 | The run confirmed 5,000 training images for each of the ten classes. |
| 31 | CK | - | 19 | The set also has a fixed train and test split by design, so there is no leakage between them. |
| 32 | CK | - | 7 | Why does this dataset suit a CNN? |
| 33 | CK | - | 11 | The pictures are small and natural, and nearby pixels belong together. |
| 34 | FACT | Y | 13 | A CNN reads exactly this kind of local structure (Course Notes, W05–W06, p.18). |
| 35 | CK | - | 19 | The dataset is large enough to train deep models, yet small enough to run many experiments in one session. |
| 36 | CK | - | 16 | It is also a well-known benchmark, so the results are easy to place against published work. |
| 37 | CK | - | 10 | This study carved a validation set from the training data. |
| 38 | RESULT | - | 12 | The final split was 45,000 training, 5,000 validation, and 10,000 test images. |
| 39 | CK | - | 8 | The validation set guided early stopping and tuning. |
| 40 | CK | - | 9 | The test set stayed untouched until the final score. |
| 41 | CK | - | 5 | Four preprocessing steps were used. |
| 42 | CK | - | 19 | Pixel values were scaled from the 0–255 range to 0–1, which helps a network train in a stable way. |
| 43 | CK | - | 12 | Labels were one-hot encoded into ten values, to match the softmax output. |
| 44 | CK | - | 19 | The training set was split into training and validation parts, using a stratified split that keeps the class balance. |
| 45 | CK | - | 7 | Data augmentation was applied during training only. |
| 46 | CK | - | 8 | Augmentation makes small random changes to each image. |
| 47 | CK | - | 10 | The pipeline used horizontal flips, small rotations, zooms, and shifts. |
| 48 | FACT | Y | 25 | This shows the model slightly different pictures each epoch, so it cannot simply memorise the training set (Srivastava e |
| 49 | CK | - | 6 | One design choice is important here. |
| 50 | CK | - | 22 | Augmentation was treated as its own measured variable, not a fixed setting, so its effect could be read directly from th |
| 51 | CK | - | 8 | The pretrained backbones needed a second data version. |
| 52 | CK | - | 9 | These models expect larger inputs and their own scaling. |
| 53 | RESULT | - | 18 | So the raw 0–255 images were kept, upscaled to 128 pixels, and passed through each backbone's own preprocessing. |
| 54 | CK | - | 13 | Mixing the two scales would quietly harm accuracy, so both versions were stored. |
| 55 | FACT | Y | 22 | Every from-scratch model followed the standard CNN recipe: convolution, ReLU, pooling, then a dense head with softmax ou |
| 56 | FACT | Y | 21 | A convolution slides a small filter over the image to find a pattern, such as an edge (Course Notes, W05–W06, p.19). |
| 57 | FACT | Y | 14 | Pooling shrinks the feature maps and keeps the strongest signals (Course Notes, W05–W06, p.21). |
| 58 | FACT | Y | 18 | The flatten layer turns the final maps into one vector for the dense layer (Course Notes, W05–W06, p.21). |
| 59 | CK | - | 14 | The study built a ladder of plain CNNs with two to six convolutional layers. |
| 60 | CK | - | 27 | It then added regularised variants and four modern blocks: Transfer learning reused five ImageNet backbones: Each backbo |
| 61 | CK | - | 12 | ResNet50 was then fine-tuned, by unfreezing the body and training it gently. |
| 62 | CK | - | 11 | All models shared one training routine, so the comparison stayed fair. |
| 63 | FACT | Y | 14 | The optimiser was Adam with a learning rate of 0.001 (Kingma and Ba, 2014). |
| 64 | CK | - | 11 | The batch size was 128 and the loss was categorical cross-entropy. |
| 65 | CK | - | 11 | Early stopping watched the validation loss and kept the best weights. |
| 66 | CK | - | 9 | A learning-rate scheduler lowered the rate when progress stalled. |
| 67 | CK | - | 12 | Fine-tuning used a much smaller rate of 1e-5 in its second phase. |
| 68 | CK | - | 10 | The run used mixed-precision arithmetic on the GPU for speed. |
| 69 | CK | - | 13 | The final softmax layer stayed in full precision to keep the probabilities accurate. |
| 70 | CK | - | 14 | Each finished model was saved to disk, so a disconnect never lost completed work. |
| 71 | CK | - | 12 | A shared routine scored every model on the same untouched test set. |
| 72 | RESULT | - | 12 | It recorded accuracy, macro precision, macro recall, macro F1, and one-vs-rest ROC-AUC. |
| 73 | CK | - | 11 | Macro averaging treats every class equally, which suits a balanced dataset. |
| 74 | CK | - | 9 | Each model wrote one row into a single leaderboard. |
| 75 | CK | - | 18 | This design follows the single-table comparison method from the Week-7 module work, adapted from two classes to ten. |
| 76 | CK | - | 7 | The from-scratch ladder gave a clear pattern. |
| 77 | CK | - | 13 | Test accuracy rose from the two-layer model to the four-layer model, then fell. |
| 78 | RESULT | - | 18 | The two-layer CNN reached 0.704, the four-layer CNN reached 0.735, and the six-layer CNN dropped back to 0.704. |
| 79 | RESULT | - | 15 | The automatic check confirmed the plateau: the six-layer F1 minus the four-layer F1 was −0.027. |
| 80 | CK | - | 9 | This result matches the expectation set before the run. |
| 81 | CK | - | 7 | Plain depth stops helping for three reasons. |
| 82 | RESULT | - | 12 | After several poolings, a 32×32 image has almost no spatial size left. |
| 83 | FACT | Y | 13 | Deep plain networks also suffer from weak gradient signals (He et al., 2016). |
| 84 | CK | - | 13 | And more layers add more parameters, which makes overfitting worse on limited data. |
| 85 | CK | - | 17 | The training curves support this reading, since the gap between training and validation accuracy widened with depth. |
| 86 | CK | - | 6 | The complete leaderboard held twenty-six models. |
| 87 | RESULT | - | 9 | Table 1 lists the strongest entries by macro F1. |
| 88 | CK | - | 8 | Transfer learning filled the top of the board. |
| 89 | RESULT | - | 11 | The best single model was a fine-tuned ResNet50 at 0.922 F1. |
| 90 | RESULT | - | 12 | The best from-scratch model reached 0.801 F1, well below the pretrained networks. |
| 91 | FACT | Y | 21 | The gap shows the value of features learned on the large ImageNet dataset (Deng et al., 2009; He et al., 2016). |
| 92 | CK | - | 8 | The three ensembles took the very top places. |
| 93 | FACT | Y | 18 | A stacking model, which trains a small logistic-regression learner on the base outputs, reached 0.940 F1 (Wolpert, 1992) |
| 94 | CK | - | 22 | The meta-learner was trained on validation predictions and judged on the test set, so it never saw the test answers in a |
| 95 | RESULT | - | 9 | The champion ensemble was strong across all ten classes. |
| 96 | RESULT | - | 7 | Per-class F1 ranged from 0.875 to 0.967. |
| 97 | RESULT | - | 11 | The easiest classes were ship (0.967), automobile (0.963), and frog (0.960). |
| 98 | CK | - | 11 | These objects have clear, rigid shapes that are easy to separate. |
| 99 | RESULT | - | 13 | The hardest class was cat at 0.875 F1, followed by dog at 0.902. |
| 100 | CK | - | 6 | Why are cats and dogs harder? |
| 101 | CK | - | 9 | Both are four-legged animals with fur and similar poses. |
| 102 | RESULT | - | 14 | At only 32×32 pixels, fine details are lost, so the two classes look alike. |
| 103 | RESULT | - | 17 | The most common single mistake was a real cat predicted as a dog, which happened 67 times. |
| 104 | CK | - | 23 | This pattern held from the weak QUICK-mode run through to the final run, which suggests it reflects the data, not the mo |
| 105 | CK | - | 11 | The confusion matrices show that errors cluster among visually similar groups. |
| 106 | CK | - | 12 | Animal classes were confused with other animals, and vehicles with other vehicles. |
| 107 | CK | - | 11 | This is sensible behaviour, because the network groups images by appearance. |
| 108 | CK | - | 10 | The result also points to a limit of low-resolution data. |
| 109 | ASSUME | - | 14 | It is likely that higher-resolution inputs would lift the weakest animal classes the most. |
| 110 | CK | - | 7 | The assignment asks for one improvement technique. |
| 111 | CK | - | 15 | This study applied every listed technique and measured each one against a plain deep baseline. |
| 112 | RESULT | - | 14 | Table 2 shows the gain in macro F1 over the five-layer baseline (F1 0.695). |
| 113 | CK | - | 11 | Every technique helped, which is the expected result with enough training. |
| 114 | CK | - | 12 | Transfer learning and ensembles gave the largest gains by a wide margin. |
| 115 | CK | - | 11 | Among the from-scratch tricks, augmentation and combined regularisation were the strongest. |
| 116 | FACT | Y | 22 | Batch normalisation alone gave only a small gain here, though it usually pays off more in deeper networks (Ioffe and Sze |
| 117 | CK | - | 6 | The ensemble result deserves a note. |
| 118 | CK | - | 11 | The three base models were already strong and made different mistakes. |
| 119 | FACT | Y | 16 | Combining them let those mistakes cancel out, so the ensemble beat every single model (Wolpert, 1992). |
| 120 | FACT | Y | 16 | This is the same principle behind bagging and boosting from the Week 8–9 material (Breiman, 1996). |
| 121 | FACT | Y | 14 | The study also added Grad-CAM heatmaps to explain single predictions (Selvaraju et al., 2017). |
| 122 | CK | - | 10 | These maps colour the pixels that most affected the decision. |
| 123 | CK | - | 17 | They give a visual check on whether the network looks at the object or at the background. |
| 124 | CK | - | 6 | Several limits should be stated openly. |
| 125 | CK | - | 15 | First, deep-learning runs on a GPU are not bit-for-bit repeatable, even with a fixed seed. |
| 126 | CK | - | 18 | So the leaderboard order is stable, but the last decimal of each score may shift slightly between runs. |
| 127 | CK | - | 11 | Second, the from-scratch models trained for a modest number of epochs. |
| 128 | ASSUME | - | 8 | Longer training might raise the weakest custom networks. |
| 129 | ASSUME | - | 21 | Third, the pretrained backbones used 128-pixel inputs rather than the full 224 pixels, which likely held their scores be |
| 130 | CK | - | 6 | One result needs care in reading. |
| 131 | CK | - | 9 | Batch normalisation alone gave a smaller gain than expected. |
| 132 | ASSUME | - | 15 | This may suggest that the chosen depth and learning rate did not let it shine. |
| 133 | CK | - | 12 | A fairer test would tune the learning rate together with batch normalisation. |
| 134 | CK | - | 13 | This study answered the assignment in full and went well beyond its floor. |
| 135 | CK | - | 14 | It compared twenty-six CNN models on one fair leaderboard, rather than three in isolation. |
| 136 | CK | - | 19 | The depth study showed that plain depth helps only up to about four layers on small images, then plateaus. |
| 137 | RESULT | - | 13 | Transfer learning was the clear winner, with a fine-tuned ResNet50 reaching 0.922 F1. |
| 138 | RESULT | - | 17 | A stacking ensemble of the best backbones took the top score at 0.940 F1 and 0.997 ROC-AUC. |
| 139 | CK | - | 22 | The analysis also explained the errors, since the cat and dog classes stayed hardest due to their visual overlap at low  |
| 140 | CK | - | 5 | The wider lesson is practical. |
| 141 | CK | - | 19 | For a small natural-image task, reusing ImageNet features and combining diverse models beats building a deeper network f |
| 142 | CK | - | 18 | Future work could raise the input size, train longer, and add a Vision Transformer as a non-convolutional comparison. |
