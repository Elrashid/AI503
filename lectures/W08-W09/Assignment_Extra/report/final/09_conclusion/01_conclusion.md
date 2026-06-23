## 8. Conclusion

This study answered the assignment in full and went well beyond its floor. It compared twenty-six CNN models on one fair leaderboard, rather than three in isolation. The depth study showed that plain depth helps only up to about four layers on small images, then plateaus. Transfer learning was the clear winner, with a fine-tuned ResNet50 reaching 0.922 F1. A stacking ensemble of the best backbones took the top score at 0.940 F1 and 0.997 ROC-AUC. The analysis also explained the errors, since the cat and dog classes stayed hardest due to their visual overlap at low resolution.

The wider lesson is practical. For a small natural-image task, reusing ImageNet features and combining diverse models beats building a deeper network from scratch. Future work could raise the input size, train longer, and add a Vision Transformer as a non-convolutional comparison.

---
