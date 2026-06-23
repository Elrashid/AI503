## Appendix B - Numeric Confusion Matrices and Worked Metric Calculation

### B.1 Numeric Confusion Matrices (Counts)

Each row is the true class; each column is the predicted class. The diagonal holds the correct predictions. Class names are abbreviated (plane = airplane, auto = automobile).

**Ensemble: Stacking (LogReg):**

| true/pred | plane | auto | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| **plane** | 960 | 4 | 4 | 1 | 1 | 1 | 1 | 2 | 21 | 5 |
| **auto** | 6 | 965 | 0 | 1 | 0 | 0 | 0 | 1 | 2 | 25 |
| **bird** | 14 | 0 | 929 | 14 | 23 | 4 | 14 | 2 | 0 | 0 |
| **cat** | 3 | 2 | 11 | 881 | 18 | 67 | 10 | 3 | 2 | 3 |
| **deer** | 6 | 0 | 12 | 19 | 930 | 5 | 10 | 17 | 1 | 0 |
| **dog** | 0 | 0 | 6 | 66 | 14 | 895 | 2 | 15 | 2 | 0 |
| **frog** | 2 | 0 | 12 | 19 | 3 | 3 | 959 | 0 | 2 | 0 |
| **horse** | 7 | 0 | 4 | 11 | 14 | 8 | 0 | 956 | 0 | 0 |
| **ship** | 23 | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 969 | 4 |
| **truck** | 5 | 30 | 1 | 2 | 0 | 1 | 1 | 0 | 6 | 954 |

**Ensemble: Soft-Vote (top3):**

| true/pred | plane | auto | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| **plane** | 945 | 4 | 5 | 2 | 3 | 0 | 1 | 3 | 31 | 6 |
| **auto** | 6 | 960 | 0 | 1 | 0 | 0 | 1 | 1 | 1 | 30 |
| **bird** | 16 | 0 | 907 | 18 | 32 | 4 | 17 | 4 | 2 | 0 |
| **cat** | 0 | 1 | 12 | 885 | 14 | 59 | 15 | 7 | 4 | 3 |
| **deer** | 1 | 0 | 10 | 23 | 922 | 5 | 16 | 22 | 0 | 1 |
| **dog** | 0 | 0 | 6 | 82 | 15 | 872 | 5 | 18 | 2 | 0 |
| **frog** | 2 | 0 | 6 | 18 | 2 | 1 | 969 | 0 | 2 | 0 |
| **horse** | 6 | 0 | 3 | 10 | 20 | 6 | 1 | 953 | 1 | 0 |
| **ship** | 19 | 3 | 2 | 0 | 1 | 0 | 1 | 0 | 970 | 4 |
| **truck** | 5 | 25 | 0 | 2 | 0 | 1 | 1 | 0 | 8 | 958 |

**Ensemble: Hard-Vote (top3):**

| true/pred | plane | auto | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| **plane** | 941 | 5 | 6 | 2 | 5 | 0 | 1 | 2 | 33 | 5 |
| **auto** | 10 | 961 | 0 | 1 | 0 | 0 | 1 | 1 | 2 | 24 |
| **bird** | 18 | 0 | 907 | 20 | 29 | 4 | 15 | 5 | 2 | 0 |
| **cat** | 9 | 2 | 20 | 885 | 9 | 52 | 13 | 5 | 2 | 3 |
| **deer** | 6 | 0 | 21 | 22 | 909 | 5 | 15 | 22 | 0 | 0 |
| **dog** | 1 | 1 | 9 | 95 | 11 | 861 | 4 | 16 | 2 | 0 |
| **frog** | 4 | 0 | 8 | 18 | 2 | 2 | 965 | 0 | 1 | 0 |
| **horse** | 9 | 0 | 4 | 15 | 21 | 5 | 0 | 946 | 0 | 0 |
| **ship** | 24 | 2 | 3 | 0 | 0 | 0 | 1 | 0 | 967 | 3 |
| **truck** | 8 | 23 | 0 | 3 | 0 | 0 | 1 | 0 | 7 | 958 |

### B.2 Worked Example - Manual Calculation of Per-Class Metrics

Champion model: **Ensemble: Stacking (LogReg)**. From the confusion matrix C, for each class i:

- TP = C[i, i]  (class i predicted correctly)
- FN = (sum of row i) - TP  (class i predicted as something else)
- FP = (sum of column i) - TP  (other classes predicted as i)
- TN = total - TP - FN - FP
- **Recall** = TP / (TP + FN)   **Precision** = TP / (TP + FP)   **F1** = 2 P R / (P + R)   **Accuracy(class)** = (TP + TN) / total

**Worked example for class "cat":**

- TP = 881, FN = 119, FP = 133, TN = 8867
- Recall = 881 / (881 + 119) = **0.881**
- Precision = 881 / (881 + 133) = **0.869**
- F1 = 2 x 0.869 x 0.881 / (0.869 + 0.881) = **0.875**
- Accuracy(cat) = (881 + 8867) / 10000 = **0.975**

**All ten classes (champion):**

| class | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| airplane | 960 | 66 | 40 | 0.936 | 0.960 | 0.948 |
| automobile | 965 | 39 | 35 | 0.961 | 0.965 | 0.963 |
| bird | 929 | 51 | 71 | 0.948 | 0.929 | 0.938 |
| cat | 881 | 133 | 119 | 0.869 | 0.881 | 0.875 |
| deer | 930 | 73 | 70 | 0.927 | 0.930 | 0.929 |
| dog | 895 | 89 | 105 | 0.910 | 0.895 | 0.902 |
| frog | 959 | 38 | 41 | 0.962 | 0.959 | 0.960 |
| horse | 956 | 40 | 44 | 0.960 | 0.956 | 0.958 |
| ship | 969 | 36 | 31 | 0.964 | 0.969 | 0.967 |
| truck | 954 | 37 | 46 | 0.963 | 0.954 | 0.958 |

Overall accuracy = sum(diagonal) / total = 9398 / 10000 = **0.940**

<!-- APPENDIX END -->
