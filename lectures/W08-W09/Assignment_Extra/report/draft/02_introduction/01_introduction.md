## 1. Introduction and Objectives

Image classification asks a model to assign one label to a picture. It is a core task in computer vision. CNNs are the standard tool for this task because they read the spatial structure of an image (LeCun et al., 1998 [R8]). A dense network treats every pixel as independent and needs a huge number of weights. A CNN instead slides small filters across the image, so it shares weights and stays compact (Course Notes, W05–W06, p.18 [R16]).

The assignment has a clear core requirement. Students must build at least three CNNs of increasing depth and compare them. They must also apply preprocessing, train the models, evaluate them, and try one improvement technique.

This study treats that requirement as a floor, not a ceiling. The main objective is to build a single, fair comparison of many architectures on one dataset. The design copies the style of the Week-7 model-comparison notebook, where many algorithms shared one results table. The secondary objective is to measure each improvement separately, so its true value is visible. A final objective is reproducibility, so every number can be traced and repeated.

---
