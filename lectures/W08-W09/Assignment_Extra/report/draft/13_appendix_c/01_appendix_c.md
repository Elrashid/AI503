## Appendix C - Assignment Brief

*The original assignment instructions, reproduced for reference. The notebook and report above implement all seven tasks (T1-T7).*

### Assignment: Image Classification Using CNN on a Large Dataset

**Title:** Deep Learning-Based Image Classification Using Convolutional Neural Networks.

**Objective:** The aim of this assignment is to design, train, and evaluate a Convolutional Neural Network (CNN) for classifying images from a large image dataset. Students will explore image preprocessing, CNN architecture design, model training, evaluation, and performance analysis.

### Dataset

Use one large image dataset such as:

- CIFAR-10
- CIFAR-100
- Fashion-MNIST
- PlantVillage Dataset
- Chest X-ray Dataset
- Kaggle Animals-10 Dataset
- Food-101 Dataset
- Brain Tumor MRI Dataset
- Traffic Signs Dataset

The dataset should include:

- At least 10,000 images
- Multiple image classes
- Training and testing sets

### Task 1: Dataset Description

Describe the selected image dataset, including the number of images, number of classes, image size, and examples of categories. Explain why this dataset is suitable for CNN classification.

### Task 2: Image Preprocessing

Apply suitable preprocessing techniques such as image resizing, normalization, train-test split, and data augmentation. Explain why preprocessing is important for CNN performance.

### Task 3: CNN Model Design

Build a CNN model that includes:

- Convolutional layers
- Pooling layers
- Activation function such as ReLU
- Flatten layer
- Dense fully connected layers
- Output layer with Softmax activation

### Task 4: Model Training

Train the CNN model using the training dataset. Select suitable hyperparameters such as batch size, number of epochs, optimizer, and learning rate.

### Task 5: Model Evaluation and Architecture Comparison

Evaluate the CNN model using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Training and validation accuracy curves
- Training and validation loss curves

In addition, students must perform an architecture comparison by gradually increasing the number of hidden layers in the CNN model and evaluating the performance each time. Students should train and compare at least three CNN architectures:

| Model | CNN Architecture |
|-------|------------------|
| Model 1 | Basic CNN with 2 convolutional layers |
| Model 2 | CNN with 3-4 convolutional layers |
| Model 3 | Deeper CNN with 5 or more convolutional layers |

For each model, students should report: training accuracy, validation accuracy, test accuracy, training loss, validation loss, precision, recall, and F1-score.

### Task 6: Results Discussion

Discuss the model performance. Identify which classes were classified correctly and which classes caused confusion. Explain possible reasons for misclassification.

### Task 7: Improvement

Improve the CNN model using one or more of the following:

- Data augmentation
- Dropout
- Batch normalization
- Transfer learning using VGG16, ResNet50, or MobileNet
- Hyperparameter tuning

### Final Deliverables

Students must submit:

- A written report.
- Python code or Jupyter Notebook.
- Dataset source link.
- Model evaluation results.
- Graphs and confusion matrix.

<!-- APPENDIX END -->
