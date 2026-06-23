## 7. Limitations and Threats to Validity

Several limits should be stated openly. First, deep-learning runs on a GPU are not bit-for-bit repeatable, even with a fixed seed. So the leaderboard order is stable, but the last decimal of each score may shift slightly between runs. Second, the from-scratch models trained for a modest number of epochs. Longer training might raise the weakest custom networks. Third, the pretrained backbones used 128-pixel inputs rather than the full 224 pixels, which likely held their scores below their ceiling.

One result needs care in reading. Batch normalisation alone gave a smaller gain than expected. This may suggest that the chosen depth and learning rate did not let it shine. A fairer test would tune the learning rate together with batch normalisation.

---
