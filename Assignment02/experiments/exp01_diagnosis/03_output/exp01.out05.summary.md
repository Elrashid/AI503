| Model                      | Quantization   |   Refusal AdvBench |   Refusal HarmBench |   False-refusal XSTest |   MMLU acc |   WikiText-2 PPL |
|:---------------------------|:---------------|-------------------:|--------------------:|-----------------------:|-----------:|-----------------:|
| Qwen/Qwen2.5-1.5B-Instruct | FP16           |           0.996667 |               0.915 |                   0.62 |      0.611 |          8.47669 |
| Qwen/Qwen2.5-1.5B-Instruct | INT8           |           0.996667 |               0.93  |                   0.56 |      0.609 |          8.52528 |
| Qwen/Qwen2.5-1.5B-Instruct | NF4            |           0.996667 |               0.765 |                   0.48 |      0.585 |          9.15365 |