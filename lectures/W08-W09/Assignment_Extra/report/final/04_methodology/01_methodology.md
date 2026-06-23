## 3. Methodology

### 3.1 Preprocessing (Task 2)

Four preprocessing steps were used. Pixel values were scaled from the 0–255 range to 0–1, which helps a network train in a stable way. Labels were one-hot encoded into ten values, to match the softmax output. The training set was split into training and validation parts, using a stratified split that keeps the class balance. Data augmentation was applied during training only.

Augmentation makes small random changes to each image. The pipeline used horizontal flips, small rotations, zooms, and shifts. This shows the model slightly different pictures each epoch, so it cannot simply memorise the training set (Srivastava et al., 2014 [R13]; Course Notes, W05–W06, p.16 [R16]). One design choice is important here. Augmentation was treated as its own measured variable, not a fixed setting, so its effect could be read directly from the leaderboard.

![One training image shown with five random augmentations.](../../figures/fig_augmentation.png)

> **Figure 3.1:** One training image shown with five random augmentations.








The pretrained backbones needed a second data version. These models expect larger inputs and their own scaling. So the raw 0–255 images were kept, upscaled to 128 pixels, and passed through each backbone's own preprocessing. Mixing the two scales would quietly harm accuracy, so both versions were stored.

### 3.2 Architecture Design (Task 3)

Every from-scratch model followed the standard CNN recipe: convolution, ReLU, pooling, then a dense head with softmax output (Course Notes, W05–W06, p.18 [R16]). A convolution slides a small filter over the image to find a pattern, such as an edge (Course Notes, W05–W06, p.19 [R16]). Pooling shrinks the feature maps and keeps the strongest signals (Course Notes, W05–W06, p.21 [R16]). The flatten layer turns the final maps into one vector for the dense layer (Course Notes, W05–W06, p.21 [R16]).

The study built a ladder of plain CNNs with two to six convolutional layers. It then added regularised variants and four modern blocks:

- a VGG-style stack of paired convolutions (Simonyan and Zisserman, 2014 [R12]);
- a residual network with skip connections (He et al., 2016 [R3]);
- a depthwise-separable network (Howard et al., 2017 [R4]);
- an all-convolutional network with global average pooling.

Transfer learning reused five ImageNet backbones:

- VGG16 (Simonyan and Zisserman, 2014 [R12]);
- ResNet50 (He et al., 2016 [R3]);
- MobileNetV2 (Sandler et al., 2018 [R10]);
- EfficientNetV2-S (Tan and Le, 2021 [R14]);
- ConvNeXt-Tiny (Liu et al., 2022 [R9]).

Each backbone first ran frozen, with only a new ten-class head trained. ResNet50 was then fine-tuned, by unfreezing the body and training it gently.

### 3.3 Training Protocol (Task 4)

All models shared one training routine, so the comparison stayed fair. The optimiser was Adam with a learning rate of 0.001 (Kingma and Ba, 2014 [R6]). The batch size was 128 and the loss was categorical cross-entropy. Early stopping watched the validation loss and kept the best weights. A learning-rate scheduler lowered the rate when progress stalled. Fine-tuning used a much smaller rate of 1e-5 in its second phase.

The run used mixed-precision arithmetic on the GPU for speed. The final softmax layer stayed in full precision to keep the probabilities accurate. Each finished model was saved to disk, so a disconnect never lost completed work.

### 3.4 Experimental Protocol

A shared routine scored every model on the same untouched test set. It recorded accuracy, macro precision, macro recall, macro F1, and one-vs-rest ROC-AUC. Macro averaging treats every class equally, which suits a balanced dataset. Each model wrote one row into a single leaderboard. This design follows the single-table comparison method from the Week-7 module work, adapted from two classes to ten.

---
