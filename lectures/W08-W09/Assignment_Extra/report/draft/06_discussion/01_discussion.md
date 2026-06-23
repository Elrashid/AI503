## 5. Discussion (Task 6)

### 5.1 Per-Class Performance

The champion ensemble was strong across all ten classes. Per-class F1 ranged from 0.875 to 0.967. The easiest classes were ship (0.967), automobile (0.963), and frog (0.960). These objects have clear, rigid shapes that are easy to separate. The hardest class was cat at 0.875 F1, followed by dog at 0.902.

![Per-class accuracy of the champion ensemble.](../../figures/fig_per_class_acc.png)

> **Figure 5.1:** Per-class accuracy of the champion ensemble.







Why are cats and dogs harder? Both are four-legged animals with fur and similar poses. At only 32×32 pixels, fine details are lost, so the two classes look alike. The most common single mistake was a real cat predicted as a dog, which happened 67 times. This pattern held from the weak QUICK-mode run through to the final run, which suggests it reflects the data, not the model size.

![Confusion matrices for the top six models.](../../figures/fig_confusion_grid.png)

> **Figure 5.2:** Confusion matrices for the top six models.







### 5.2 Reading the Confusion Structure

The confusion matrices show that errors cluster among visually similar groups. Animal classes were confused with other animals, and vehicles with other vehicles. This is sensible behaviour, because the network groups images by appearance. The result also points to a limit of low-resolution data. It is likely that higher-resolution inputs would lift the weakest animal classes the most.

![Ten test images the champion misclassified (true vs predicted).](../../figures/fig_misclassified.png)

> **Figure 5.3:** Ten test images the champion misclassified (true vs predicted).







---
