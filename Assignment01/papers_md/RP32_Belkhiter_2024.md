<!-- RP32_Belkhiter_2024 | source: papers_json/RP32_Belkhiter_2024/ -->

## HarmLevelBench: Evaluating Harm-Level Compliance and the Impact of Quantization on Model Alignment

## Yannis Belkhiter^

 ext: ^

IBM Research Europe Trinity College Dublin yannis.belkhiter@ibm.com

## Giulio Zizzo

IBM Research Europe Dublin, Ireland giulio.zizzo2@ibm.com

## Sergio Maffeis

Imperial College London London, UK sergio.maffeis@ic.ac.uk

## Abstract

Warning: This report contains sensitive content and potentially harmful information. With the introduction of the transformers architecture, LLMs have revolutionized the NLP field with ever more powerful models. Nevertheless, their development came up with several challenges. The exponential growth in computational power and reasoning capabilities of language models has heightened concerns about their security. As models become more powerful, ensuring their safety has become a crucial focus in research. This paper aims to address gaps in the current literature on jailbreaking techniques and the evaluation of LLM vulnerabilities. Our contributions include the creation of a novel dataset designed to assess the harmfulness of model outputs across multiple harm levels, as well as a focus on fine-grained harm-level analysis. Using this framework, we provide a comprehensive benchmark of state-of-the-art jailbreaking attacks, specifically targeting the Vicuna 13B v1.5 model. Additionally, we examine how quantization techniques, such as AWQ and GPTQ, influence the alignment and robustness of models, revealing trade-offs between enhanced robustness with regards to transfer attacks and potential increases in vulnerability on direct ones. This study aims to demonstrate the influence of harmful input queries on the complexity of jailbreaking techniques, as well as to deepen our understanding of LLM vulnerabilities and improve methods for assessing model robustness when confronted with harmful content, particularly in the context of compression strategies.

# 1 Introduction

While numerous LLMs have been developed in recent years [[1]](#page-7-0) [[2]](#page-7-1), aligning these models with human preferences remains a complex and ongoing challenge. LLM alignment refers to the process of guiding models to avoid generating harmful or undesired outputs, ensuring their safe and ethical use. Recent work, such as Ouyang et al. [[4]](#page-7-2) and Munos et al. [[5]](#page-7-3), has demonstrated that specific fine-tuning strategies can significantly reduce the risk of harmful content generation.

However, as models become more advanced, their vulnerabilities also become more pronounced. LLM vulnerabilities refer to the weaknesses that can be exploited to make models generate unsafe, harmful, or unintended content [[10]](#page-8-0). These vulnerabilities may arise from the vast and often uncensored datasets used during training, or from the model's inherent capacity to generalize and respond to a wide range of queries [[3]](#page-7-4). This makes them susceptible to adversarial manipulation, where malicious users can craft inputs to elicit harmful outputs. This has led to the rise of jailbreaking methods, which are designed to probe and exploit these vulnerabilities to better understand the

> ^∗^Work done during MSc. thesis at Imperial College London

<!-- page 2 -->

limitations of models. Several state-of-the-art jailbreaking techniques continue to bypass alignment measures, successfully eliciting harmful responses or sensitive information from models [[6]](#page-7-5) [[11]](#page-8-1).

In the context of adversarial attacks, assessing model compliance remains a difficult problem. Even with robust alignment strategies, ensuring that models consistently adhere to ethical and safety guidelines across a wide range of queries is challenging. In addition, the adoption of compression techniques, such as quantization, has introduced new challenges in the alignment of LLMs. While quantization improves computational efficiency, Kumar et al. [[12]](#page-8-2) proved that it can also influence model behavior, particularly in adversarial contexts, where models compressed through these methods may exhibit different susceptibilities to jailbreaking techniques. Understanding how compression affects model robustness and alignment remains an underexplored area, with trade-offs between model size and safety emerging as a critical concern.

This paper aims to address key gaps in the current jailbreaking literature by proposing a new framework for LLM vulnerability assessment. First, we introduce HarmLevelBench, a novel dataset comprising queries across 7 harmful topics, each further categorized into 8 distinct levels of severity, enabling a fine-grained analysis of model responses. Second, we conduct a comprehensive performance comparison of 7 state-of-the-art jailbreaking techniques applied to this dataset, offering insights into their effectiveness across various harm levels. Finally, we examine how quantization techniques, such as AWQ and GPTQ, influence the alignment and robustness of models, revealing trade-offs between resilience to transferred attacks and vulnerability to direct ones.

# 2 HarmLevelBench Pipeline

The framework we designed aims to provide a robust foundation pipeline for assessing the vulnerabilities of LLMs to harmful content. After defining the structure of our dataset, we will present the metrics we designed, the jailbreaking methods, and the models we selected.

## 2.1 Dataset

To perform LLM jailbreaking, various datasets have emerged. The construction of an adversarial dataset for LLM jailbreaking is complex. Given the versatile nature of the English language, and the vast amount of sensitive topics, creating relevant queries is hard. Datasets such as AdvBench [[9]](#page-8-3), Pruned AdvBench [[7]](#page-7-6), XSTest [[13]](#page-8-4), and SafetyBench [[14]](#page-8-5) are commonly used in the jailbreaking literature. However, upon closer examination, we found that state-of-the-art datasets often *lack* *sufficient formulation structure* of queries and fail to provide a nuanced assessment of the *varying* *degrees of harmfulness* in the queries. This lack of structure limits their ability to support a finegrained analysis of how models respond to different levels of harmful or sensitive content, leaving gaps in the thorough evaluation of LLM vulnerabilities.

### 2.1.1 Dataset structure

Considering the two observed shortcomings, we built the HarmLevelBench dataset, which covers a diverse range of 7 potentially harmful topics, and addresses the challenges above:

- 1. Question template: To solve the formulation problem highlighted earlier, we used a consistent question template for each topic, namely: "How to ...?". This approach ensures that our dataset maintains a structured and uniform format, which is essential for robust evaluation and comparison. By using the same questioning template, we can systematically test the model's responses to similar types of queries across different topics.
- 2. Harm level: To introduce depth and mitigate the categorization problem highlighted in the literature, we created 8 harm levels for each topic, ranging from low to high severity. Queries include gradually increasing harmful topics and related subjects (see Table[ 7)](#page-10-0). This categorization allows us to systematically evaluate how LLMs respond to varying degrees of adversarial prompts and measure their vulnerability across different levels of harm.

Following this section, we show that the combination of harm-level categorization and a consistent questioning template allows us to address the challenges identified in the literature effectively. It provides a nuanced understanding of LLM vulnerabilities and enables a more detailed analysis of the model's compliance with adversarial prompts.

<!-- page 3 -->

### 2.1.2 Comparison with the pruned AdvBench

Figure[ 1](#page-2-0) displays a Principal Component Analysis (PCA) of the BERT encodings from the pruned version of the AdvBench dataset [(1a](#page-2-0) - where labels are assigned to previously unlabeled examples based on the existing labels in the dataset), originally published by Zou et al. [[9]](#page-8-3) and later refined by Chao et al. [[7]](#page-7-6), alongside the encodings from our HarmLevelBench dataset. A limitation of the Pruned AdvBench is its reliance on a almost single query per topic, which leads to insufficient categorization and increased susceptibility to noise. In contrast, the clustering results indicate that HarmLevelBench enjoys a clearer separation of query categories, suggesting distinct contextual embedding for classes such as "Bomb" and "Trigger". This differentiation underscores the importance of incorporating a wider variety of examples within each topic, and more granular categorization of query classes, potentially leading to more distinct contextual embeddings.

![RP32_Belkhiter_2024 fig01](../figures/RP32_Belkhiter_2024_fig01.jpg)
*Figure 1: Comparison of PCA applied to the BERT encoding of two datasets*

In addition, we can compare the formulation structure of each query with the Pruned AdvBench dataset. Table[ 1](#page-2-1) presents this difference between Pruned AdvBench and HarmLevelBenchdatasets. We can see that our queries have the advantages of being *consistent* and *short*, emphasizing the topics in the question, and allowing easier comparison between queries as they follows the same patterns.

## 2.2 Metrics

Common practices for assessing the success of a jailbreaking attempt typically rely on Attack Success Rate (ASR). ASR is evaluated using one of three primary methods: human annotation, string detection, or an LLM judge. Human annotation involves a manual check of the model's output by annotators, ensuring a high level of accuracy but often being time-consuming and subjective. String detection [[9]](#page-8-3), on the other hand, uses predefined keywords or patterns to automatically distinguish refusal to comply from the model, offering efficiency but potentially missing nuanced cases. Finally, LLM judge involves using another model -usually GPT- to evaluate the response, by classifying the outputs into 3 categories: *"1_full_compliance"*, *"2_partial_refusal"*, or *"3_full_refusal"* [[13]](#page-8-4). Even though it provides scalability, it is also suffering from the same biases and limitations as the original model. Each of these approaches presents trade-offs, and no single method fully captures the complexity of jailbreaking attempts, complicating the evaluation of LLM vulnerabilities.

<!-- page 4 -->

# 3 Experimental setup

In order to test our framework, we conducted a number of jailbreaking attacks on a common target model, using our HarmLevelBench dataset. We also applied our pipeline to two state-of-the-art quantization methods to assess the influence of quantization toward model alignment.

## 3.1 Jailbreaking methods

We consider 7 different jailbreaking methods, order by complexity. First, we introduced three manual approaches. Simple query just submits the query directly, as a baseline. Ignorance context adds a misleading context before submitting the query. For Role Play context, we created imaginative contexts for each query using the Vicuna Wizard model. We then implemented four state-of-the-art jailbreaking attack representative of the published literature:

- *Query-based attacks:* We selected PAIR [[7]](#page-7-6), including Mixtral 7x8B as an attacker, GPT-3.5 turbo as a judge, and 5 streams of 5 iterations per query. We also implemented the PAP method [[8]](#page-8-6), which will be the average results of attacks led using 3 different configurations: Authority, Logical appealing, and Misrepresentation.
- *Prompt Engineering attacks:* We selected the AutoDAN-GA method [[15]](#page-8-7).
- *Universal attacks:* We selected the GCG attack [[9]](#page-8-3).

Starting from Section[ 4,](#page-3-0) we define the different jailbreaking methods on a complexity axis. More specifically, this ranking has been established based on the level of sophistication and automation involved in the execution of the different attacks. Simpler methods rely on manual query manipulation or context addition, with increasing complexity. More advanced approaches, such as Query-based techniques, incorporate multi-step prompts and dynamic adjustments. The most complex techniques, like Prompt Engineering or Universal attacks, employ automatic generation (see Table[ 5)](#page-8-8).

## 3.2 Models

We run our experiments on the *Vicuna 13B v1.5* model because of its strong performance across a variety of natural language processing tasks, particularly in maintaining high-quality output while navigating complex queries and adversarial prompts. We also experimented with the Llama model, but obtained lower performance.

The ever-growing demand for more efficient and scalable language models has led to the emergence of several compression techniques. Among these methods, quantization has gained significant attention, as it enables the deployment of high-capacity models with lower resource consumption. For our study, we specifically focus on AWQ [[16]](#page-8-9) and GPTQ [[17]](#page-8-10) techniques, which represent two prominent sub-fields of quantization: Quantization Aware Training and Post-Training Quantization. By comparing the strengths of both approaches, we aim to offer a comprehensive evaluation of model quantization.

# 4 Evaluation

This section presents a comprehensive analysis of our experimental jailbreaking lead using the HarmLevelBench dataset on standard and quantized models, focusing on the ASR and HarmLevel categories.

## 4.1 Classic ASR

Table[ 2](#page-4-0) presents the ASR metrics of the 7 attacks lead on the *Vicuna 13B v1.5* model. Simple Query and Ignorance context showcase the lowest ASR in both Human and String evaluation, indicating these techniques are less effective in fully exploiting the model. AutoDAN and GCG have notably higher success rates across all categories, with nearly perfect success rates, particularly in Human ASR and String ASR metrics. Comparatively, PAIR and PAP methods also perform well, especially in GPT Judge's Partial Compliance. This suggests a high frequency of partial jailbreaking success, delivering relatively useful sensitive information during each attacks.

<!-- page 5 -->

## 4.2 HarmLevel

Moving forward, our focus shifts to evaluating the potential impact or severity of successful jailbreaks, using the HarmLevel of the queries relying in our HarmLevelBench dataset. The heatmaps provided by Figure[ 2](#page-4-1) and[ 3](#page-4-2) visualize how different jailbreaking techniques affect the target model across various jailbreaking complexity levels and HarmLevel severity.

![RP32_Belkhiter_2024 fig02](../figures/RP32_Belkhiter_2024_fig02.jpg)
*Figure 2: Average ASR by HarmLevel and jailbreaking complexity for Vicuna 13B v1.5*

Across the three figures, the y-axis represents Harm-Level severity, while the x-axis represents the complexity of the jailbreaks. The color-coded heatmap demonstrates a clear gradient that indicates the relationship between the complexity of a jailbreak and the corresponding harm level of queries. Darker colors (purple) likely represent scenarios where the ASR (Attack Success Rate) remains relatively low, whereas lighter colors (yellow) show instances where more severe harm occurs due to successful jailbreaking. This visualization highlights the distribution of harm based on varying jailbreak complexities.

Specifically, in Figures[ 2a](#page-4-1) and[ 2b,](#page-4-1) as the jailbreak complexity increases (moving to the right), the model appears more prone to producing higher harm level outputs (towards the top of the y-axis). However, Figure[ 3](#page-4-2) presents a nuanced result, where more complex attacks lead to less full compliance based on the LLM judge. Figure 3: GPT ASR Heatmap

![RP32_Belkhiter_2024 fig03](../figures/RP32_Belkhiter_2024_fig03.jpg)

<!-- page 6 -->

We can conclude that based on the ASR metrics, the HarmLevel of a query has an impact on the compliance rate of a model depending on the jailbreaking complexity. While it has close to no impact on the most complex jailbreaking methods explored, highly harmful queries are leading to relatively poor ASR for low to moderate-complex jailbreaks.

## 4.3 Vulnerability of Quantized Models to Direct Attacks

While close to no work has been done applying jailbreaking techniques to compressed models, Kumar et al. [[12]](#page-8-2) suggested one of the first paper on the influence of quantization toward model alignment. To pursue this work, we directly applied our framework to an *AWQ Vicuna 13B v1.5* model. First, we present the ASR metrics. Then, we provided a fine-grained approach based on the Harm level.

As for the original Vicuna model, Table[ 3](#page-5-0) suggests that the effectiveness of jailbreaking techniques can vary significantly when applied to quantized models. The results indicate that certain techniques, particularly those with a higher degree of complexity, also yield better success rates in terms of both Human ASR and String ASR metrics. For instance, the PAIR technique achieves a notably high String ASR of 98.2, demonstrating its efficiency in eliciting compliant outputs from the AWQ Vicuna 13B v1.5 model. ASR results of the AWQ model seems to offer higher ASR than the original model (Section[ 4.1)](#page-3-1) for Query-based attacks, while showcasing lower results for advanced jailbreaks.

Furthermore, Figure[ 4](#page-5-1) present the influence of the harm level of queries toward the complexity of attacks. It can be observed that the quantization seems to enhance the phenomenon highlighted in Section[ 4.2](#page-4-3) as the color gradient is more visible for each sub-figures. While lower Harm Levels tends to lead to higher ASR, it gradually declines with respect to each type of attacks when the harmfulness of the query increase. Compared to Kumar et al. [[12]](#page-8-2), the results obtained are nuanced. While attacks of quantized models by certain types of jailbreaking methods can lead to higher ASR than the original model (cf. [[12]](#page-8-2) - higher ASR for quantized Llama using TAP), quantization also seems to offer more robustness with regards to more complex type of jailbreaking.

![RP32_Belkhiter_2024 fig04](../figures/RP32_Belkhiter_2024_fig04.jpg)
*Figure 4: ASR by Harm level and jailbreaking complexity for AWQ Vicuna 13B v1.5 Direct attacks*

## 4.4 Enhanced Robustness of Quantization Against Transferred Attacks

While most of the attacks relies on the target model to perform a jailbreaking, we can wonder if quantized models are susceptible to transferred attacks crafted using a more accurate model. Transferred attacks leverage previously successful jailbreaking techniques on one model to assess the robustness of another model. This subsection aims to analyze how quantized models like the AWQ

<!-- page 7 -->

and GPTQ versions of Vicuna 13B v1.5 fare against such attacks obtained on the original model. The goal is to explore whether the quantization process enhances the model's defenses attacks.

Table[ 4](#page-6-0) reveals distinct results compared to the direct attacks lead on original and quantized models. While transferred attacks lead to similar ASR score for manual methods (from 1 to 3), it indicates a general trend toward increased robustness in quantized models for more complex jailbreaking methods compared to direct attacks on the original model. For instance, the PAIR technique shows a notable decline in Human ASR, dropping from 94.6 on the original model to 66.1 for AWQ and 64.3 for GPTQ. Similarly, AutoDAN's effectiveness decreases from 100 to 71.4 and 67.9, respectively.

The heatmaps depicted in Figure[ 5](#page-6-1) for AWQ and Figure[ 6](#page-6-2) for GPTQ further reinforce the findings observed in Table[ 4.](#page-6-0) In both AWQ and GPTQ models, there is a clear pattern showing that as the complexity of the jailbreaking techniques increases, the ASR scores for transferred attacks decline significantly, particularly for more advanced techniques. For instance, while simpler attacks yield relatively high ASR scores across both models, the effectiveness of complex attacks diminishes notably, emphasizing the increased robustness of quantized models. This trend is particularly pronounced in methods like PAIR and AutoDAN, which, despite showing effectiveness against the original model, exhibit significantly lower ASR scores in their transferred forms for quantized models.

![RP32_Belkhiter_2024 fig05](../figures/RP32_Belkhiter_2024_fig05.jpg)
*Figure 5: ASR by HarmLevel and jailbreaking complexity for AWQ Vicuna 13B v1.5 Transferred attacks*

![RP32_Belkhiter_2024 fig06](../figures/RP32_Belkhiter_2024_fig06.jpg)
*Figure 6: ASR by Harm level and jailbreaking complexity for GPTQ Vicuna 13B v1.5 Transferred attacks*

<!-- page 8 -->

In terms of harm level, the analysis reveals a shift in attack dynamics between the original and quantized models. Transferred attacks appear to exhibit a reduced effectiveness at higher harm levels for the quantized models compared to the original model, where higher harm levels correlate with a decrease in ASR for the quantized models. The original model shows more susceptibility to attacks aimed at causing significant harm, whereas the quantized versions, while still vulnerable, present lower ASR values overall, indicating a better defense against such aggressive attempts.

# 5 Summary and Conclusion

While assessing model compliance in the context of adversarial attacks, we viewed that taking the scale of the harmlevel can gives valuable insights rather than using the average ASR. Depending on the degree of complexity of a jailbreak, the harmfulness of a query can have a severe impact on the effectiveness of the attack. In addition, this analysis can gives additional information on the behavior of a model with regards to specific sensitive topics.

Moreover, we offered an alternative approach compared to existing work applying jailbreaking techniques to compress models. While some attacks were shown to be more effective on quantized models, our findings indicate that quantization generally enhances robustness against adversarial examples in transferred contexts. Notably, when analyzing the harm levels, we observed that quantized models are more susceptible to harm-level, even for higher complex jailbreaks.

Limitations: Despite the insights gained from this study, several limitations should be acknowledged. First, the scope of our analysis was confined to specific jailbreaking techniques, and the results may not generalize to all possible attacks or models. Future work should explore a wider range of attack strategies to better understand the nuances of quantization effects across various contexts. Second, the evaluation could benefit from using a larger HarmLevelBench dataset. While datasets around 50 queries are the norm in the literature, using much more topics could help to better analyse LLM behaviors. Lastly, while quantization appears to enhance robustness in this study, it is essential to investigate the trade-offs in performance and accuracy that may arise from the quantization process, as well as other types of method employed.

## References

- [1] OpenAI, *GPT-4*, 2023. [https://platform.openai.com/docs/models/gpt-4](https://platform.openai.com/docs/models/gpt-4), Accessed: 2024-08-20.
- [2] Rohan Anil et al., *PaLM 2 Technical Report*, 2023.[ arXiv:2305.10403](http://arxiv.org/abs/2305.10403) [cs.CL].
- [3] Su Lin Blodgett, Solon Barocas, Hal Daumé III, Hanna Wallach, *Language (Technology) is* *Power: A Critical Survey of "Bias" in NLP*, 2020.[ arXiv:2005.14050](http://arxiv.org/abs/2005.14050) [cs.CL].
- [4] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe, *Training language models to follow instructions with human feedback*, 2022.[ arXiv:2203.02155](http://arxiv.org/abs/2203.02155) [cs.CL]. [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155).
- [5] Rémi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland, Zhaohan Daniel Guo, Yunhao Tang, Matthieu Geist, Thomas Mesnard, Andrea Michi, Marco Selvi, Sertan Girgin, Nikola Momchev, Olivier Bachem, Daniel J. Mankowitz, Doina Precup, Bilal Piot, *Nash Learning from Human Feedback*, 2024.[ arXiv:2312.00886](http://arxiv.org/abs/2312.00886) [stat.ML]. [https:](https://arxiv.org/abs/2312.00886) [//arxiv.org/abs/2312.00886](https://arxiv.org/abs/2312.00886).
- [6] Arijit Ghosh Chowdhury, Md Mofijul Islam, Vaibhav Kumar, Faysal Hossain Shezan, Vaibhav Kumar, Vinija Jain, Aman Chadha, *Breaking Down the Defenses: A Comparative Survey of* *Attacks on Large Language Models*, 2024.[ arXiv:2403.04786](http://arxiv.org/abs/2403.04786) [cs.CR]. [https://arxiv.org/](https://arxiv.org/abs/2403.04786) [abs/2403.04786](https://arxiv.org/abs/2403.04786).
- [7] Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, Eric Wong, *Jailbreaking Black Box Large Language Models in Twenty Queries*, 2024.[ arXiv:2310.08419](http://arxiv.org/abs/2310.08419) [cs.LG]. [https://arxiv.org/abs/2310.08419](https://arxiv.org/abs/2310.08419).

<!-- page 9 -->

- [8] Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, Weiyan Shi, *How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs*, 2024. arXiv:2401.06373 [cs.CL]. https://arxiv.org/abs/2401.06373.
- [9] Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson, *Universal and Transferable Adversarial Attacks on Aligned Language Models*, 2023. arXiv:2307.15043 [cs.CL]. https://arxiv.org/abs/2307.15043.
- [10] Maximilian Mozes, Xuanli He, Bennett Kleinberg, Lewis D. Griffin, *Use of LLMs for Illicit Purposes: Threats, Prevention Measures, and Vulnerabilities*, 2023. arXiv:2308.12833 [cs.CL]. https://arxiv.org/abs/2308.12833.
- [11] Yi Liu, Gelei Deng, Zhengzi Xu, Yuekang Li, Yaowen Zheng, Ying Zhang, Lida Zhao, Tianwei Zhang, Kailong Wang, Yang Liu, *Jailbreaking ChatGPT via Prompt Engineering: An Empirical Study*, 2024. arXiv:2305.13860 [cs.SE]. https://arxiv.org/abs/2305.13860.
- [12] Divyanshu Kumar, Anurakt Kumar, Sahil Agarwal, Prashanth Harshangi, *Fine-Tuning, Quantization, and LLMs: Navigating Unintended Outcomes*, 2024. arXiv:2404.04392 [cs.CR]. https://arxiv.org/abs/2404.04392.
- [13] Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, Dirk Hovy, XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models, 2024. arXiv:2308.01263 [cs.CL]. https://arxiv.org/abs/2308.01263.
- [14] Zhexin Zhang, Leqi Lei, Lindong Wu, Rui Sun, Yongkang Huang, Chong Long, Xiao Liu, Xuanyu Lei, Jie Tang, Minlie Huang, SafetyBench: Evaluating the Safety of Large Language Models, 2024. arXiv:2309.07045 [cs.CL]. https://arxiv.org/abs/2309.07045.
- [15] Xiaogeng Liu, Nan Xu, Muhao Chen, Chaowei Xiao, AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models, 2024. arXiv:2310.04451 [cs.CL]. https://arxiv.org/abs/2310.04451.
- [16] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han, AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration, 2024. arXiv:2306.00978 [cs.CL]. https://arxiv.org/abs/2306.00978.
- [17] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh, GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers, 2023. arXiv:2210.17323 [cs.LG]. https://arxiv.org/abs/2210.17323.

## **Appendix **

<!-- page 10 -->

<!-- page 11 -->

## Key to Symbols:

• Green: Low Impact / Very Simple / Legal & Ethical

• Yellow: Moderate Impact / Simple to Moderate Complexity

• Orange: Significant Impact / Complex / Potentially Illegal

• Red: High Impact / Complex / Illegal or Immoral
