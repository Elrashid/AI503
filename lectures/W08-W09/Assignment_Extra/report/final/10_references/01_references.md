## References

**[R1]** Breiman, L. (1996) 'Bagging predictors', *Machine Learning*, 24(2), pp. 123–140. https://doi.org/10.1007/BF00058655

**[R2]** Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K. and Fei-Fei, L. (2009) 'ImageNet: A large-scale hierarchical image database', *IEEE Conference on Computer Vision and Pattern Recognition*, pp. 248–255. https://doi.org/10.1109/CVPR.2009.5206848

**[R3]** He, K., Zhang, X., Ren, S. and Sun, J. (2016) 'Deep residual learning for image recognition', *IEEE Conference on Computer Vision and Pattern Recognition*, pp. 770–778. https://doi.org/10.1109/CVPR.2016.90

**[R4]** Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M. and Adam, H. (2017) 'MobileNets: Efficient convolutional neural networks for mobile vision applications', *arXiv:1704.04861*. https://doi.org/10.48550/arXiv.1704.04861

**[R5]** Ioffe, S. and Szegedy, C. (2015) 'Batch normalization: Accelerating deep network training by reducing internal covariate shift', *International Conference on Machine Learning*, pp. 448–456. https://doi.org/10.48550/arXiv.1502.03167

**[R6]** Kingma, D.P. and Ba, J. (2014) 'Adam: A method for stochastic optimization', *arXiv:1412.6980*. https://doi.org/10.48550/arXiv.1412.6980

**[R7]** Krizhevsky, A. (2009) *Learning multiple layers of features from tiny images*. Technical report, University of Toronto. Available at: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

**[R8]** LeCun, Y., Bottou, L., Bengio, Y. and Haffner, P. (1998) 'Gradient-based learning applied to document recognition', *Proceedings of the IEEE*, 86(11), pp. 2278–2324. https://doi.org/10.1109/5.726791

**[R9]** Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T. and Xie, S. (2022) 'A ConvNet for the 2020s', *IEEE Conference on Computer Vision and Pattern Recognition*, pp. 11976–11986. https://doi.org/10.1109/CVPR52688.2022.01167

**[R10]** Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. and Chen, L.-C. (2018) 'MobileNetV2: Inverted residuals and linear bottlenecks', *IEEE Conference on Computer Vision and Pattern Recognition*, pp. 4510–4520. https://doi.org/10.1109/CVPR.2018.00474

**[R11]** Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. and Batra, D. (2017) 'Grad-CAM: Visual explanations from deep networks via gradient-based localization', *IEEE International Conference on Computer Vision*, pp. 618–626. https://doi.org/10.1109/ICCV.2017.74

**[R12]** Simonyan, K. and Zisserman, A. (2014) 'Very deep convolutional networks for large-scale image recognition', *arXiv:1409.1556*. https://doi.org/10.48550/arXiv.1409.1556

**[R13]** Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. and Salakhutdinov, R. (2014) 'Dropout: A simple way to prevent neural networks from overfitting', *Journal of Machine Learning Research*, 15(1), pp. 1929–1958. Available at: https://jmlr.org/papers/v15/srivastava14a.html

**[R14]** Tan, M. and Le, Q. (2021) 'EfficientNetV2: Smaller models and faster training', *International Conference on Machine Learning*, pp. 10096–10106. https://doi.org/10.48550/arXiv.2104.00298

**[R15]** Wolpert, D.H. (1992) 'Stacked generalization', *Neural Networks*, 5(2), pp. 241–259. https://doi.org/10.1016/S0893-6080(05)80023-1

**[R16]** *Course Notes (W05–W06): Deep Learning: A Comprehensive Guide. AI503 Machine Learning, The British University in Dubai.* (internal teaching material; no DOI)


<!-- NON-CITED CLAIMS
[CK] "Image classification asks a model to assign one label to a picture" — Common knowledge; standard definition of the task.
[CK] "CNNs are the standard tool for this task" — Common knowledge in computer vision; also supported by LeCun et al. (1998).
[CK] "Pixel values were scaled from the 0-255 range to 0-1" — Common knowledge; standard normalisation step.
[CK] "Macro averaging treats every class equally" — Common knowledge; definition of macro averaging.
[ASSUME] "higher-resolution inputs would lift the weakest animal classes the most" — Marked with "it is likely that"; reasonable inference from the low-resolution confusion pattern, not directly tested.
[ASSUME] "the chosen depth and learning rate did not let it shine" (batch normalisation) — Marked with "this may suggest"; a plausible explanation for the small observed gain, not proven.
[ASSUME] "Longer training might raise the weakest custom networks" — Marked as a limitation; standard expectation, not tested in this run.
[ASSUME] "128-pixel inputs ... likely held their scores below their ceiling" — Marked with "likely"; inference, since full 224-pixel inputs were not run.
RESULTS: All numeric results (accuracies, F1, ROC-AUC, per-class scores, deltas, counts) are this study's own experimental findings, recorded in run_outputs.md and cnn_leaderboard_results.csv. They are reported as results and need no external citation.
-->

<!-- CITATION SOURCES
(Course Notes, W05-W06, p.18) = "Deep Learning: A Comprehensive Guide" (W05-W06 PDF) Chapter 4, page 18 — CNN pipeline diagram + why CNN over a dense network for images.
(Course Notes, W05-W06, p.19) = same PDF, Ch 4, p.19 — convolution / filters / feature maps.
(Course Notes, W05-W06, p.21) = same PDF, Ch 4, p.21 — pooling, flatten, dense, softmax head.
(Course Notes, W05-W06, p.16) = same PDF, Ch 3, p.16 — dropout and data augmentation.
(Krizhevsky, 2009) = CIFAR-10 dataset description (60,000 32x32 images, 10 classes).
(LeCun et al., 1998) = origin of the modern CNN (LeNet).
(Simonyan and Zisserman, 2014) = VGG architecture. (He et al., 2016) = ResNet / residual connections.
(Howard et al., 2017) = MobileNet depthwise-separable convolutions. (Sandler et al., 2018) = MobileNetV2.
(Tan and Le, 2021) = EfficientNetV2. (Liu et al., 2022) = ConvNeXt. (Deng et al., 2009) = ImageNet.
(Kingma and Ba, 2014) = Adam optimiser. (Srivastava et al., 2014) = Dropout. (Ioffe and Szegedy, 2015) = Batch normalisation.
(Selvaraju et al., 2017) = Grad-CAM. (Wolpert, 1992) = Stacking. (Breiman, 1996) = Bagging.
-->

<!-- APPENDIX START -->
