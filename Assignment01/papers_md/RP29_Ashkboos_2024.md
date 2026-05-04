<!-- RP29_Ashkboos_2024 | source: papers_json/RP29_Ashkboos_2024/ -->

![RP29_Ashkboos_2024 fig01](../figures/RP29_Ashkboos_2024_fig01.jpg)

## QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs

## Saleh Ashkboos

ETH Zurich

saleh.ashkboos@inf.ethz.ch

## Amirkeivan Mohtashami

EPFL

amirkeivan.mohtashami@epfl.ch

Maximilian L. Croci

Microsoft Research mcroci@microsoft.com

Bo Li ETH Zurich

bolibo@ethz.ch

Pashmina Cameron

Microsoft

pcameron@microsoft.com

Martin Jaggi

EPFL martin.jaggi@epfl.ch

Dan Alistarh

IST Austria & NeuralMagic dan.alistarh@ist.ac.at

Torsten Hoefler

ETH Zurich torsten.hoefler@inf.ethz.ch

James Hensman

Microsoft Research jameshensman@microsoft.com

## Abstract

We introduce QuaRot, a new *Qua*ntization scheme based on *Rot*ations, which is able to quantize LLMs end-to-end, including all weights, activations, and KV cache in 4 bits. QuaRot rotates LLMs in a way that removes outliers from the hidden state without changing the output, making quantization easier. This *computational* *invariance* is applied to the hidden state (residual) of the LLM, as well as to the activations of the feed-forward components, aspects of the attention mechanism, and to the KV cache. The result is a quantized model where all matrix multiplications are performed in 4 bits, without any channels identified for retention in higher precision. Our 4-bit quantized LLAMA2-70B model has losses of at most 0.47 WikiText-2 perplexity and retains 99% of the zero-shot performance. We also show that QuaRot can provide lossless 6 and 8 bit LLAMA-2 models without any calibration data using round-to-nearest quantization. Code is available at [github.com/spcl/QuaRot](github.com/spcl/QuaRot).

# 1 Introduction

Large language models (LLMs) have become increasingly important due to their countless applications. However, using these models in practice, known as inference, requires a significant amount of computation, memory, and energy, specifically during the *prefill* phase, in which the model is supposed to process large prompts and cache them in each layer. Quantization is among the most important techniques to improve both memory and compute issues by keeping the data types at lower precision during the forward pass.

As the prefill stage is known to be compute-bound [[Ashkboos et al.,](#page-9-0) [2023]](#page-9-0), joint quantization aims to reduce the precision of parameters and KV cache (which results in lower memory usage) as well as inputs (known as activations) and compute the forward pass in low precision. However, quantizing the activations is hard as they have large outlier elements (see Figure [1](#page-1-0) for an illustrative example) with much larger values, making activation quantization more difficult than weight quantization, especially for the 4-bit case. Previous work relies on using a calibration set to characterize the outlier features and keeping them in higher precision for inference [[Zhao et al.,](#page-11-0) [2023,](#page-11-0) [Ashkboos et al.,](#page-9-0) [2023]](#page-9-0).

<!-- page 2 -->

![RP29_Ashkboos_2024 fig02](../figures/RP29_Ashkboos_2024_fig02.jpg)
*Figure 1: The distributions of activations at the input to the FFN block in LLAMA2-7B model, in the tenth layer. Left: using the default configuration as downloaded from Hugging Face. Right: after processing using QuaRot. The processed distribution has no outliers, leading to superior quantization.*

In this work, we address the issue of outlier features by rotating the inputs of the model using randomized Hadamard transformations. We do this using the *computational invariance* idea [[Ashkboos et al.,](#page-9-1) [2024]](#page-9-1) and fuse Hadamard transformations into the weight matrices, resulting in an equivalent network without outlier features. This enables the weights, activations, and KV caches to be quantized to 4 bits with minimal accuracy drop. Our main contributions are:

- We show that randomized Hadamard transformations can be applied to the weight matrices without additional model modifications. In turn, this completely eliminates outlier features and makes the activations easy to quantize, without changing the output of the model. This can be seen as an extension of the *computational invariance* idea, proposed in SliceGPT [[Ashkboos et al.,](#page-9-1) [2024]](#page-9-1) in the context of structured pruning.
- We extend this approach to apply *online* Hadamard transformations to the attention module to remove outlier features in keys and values, enabling the KV cache to be quantized.
- Using the above modifications, QuaRot enables 4-bit LLM inference by quantizing all weights, activations, and KV caches using integer quantization. We provide efficient kernel support for QuaRot: on a LLAMA2-70B model, QuaRot achieves up to 3.33× prefill speedups (on a batch size 64 with 2048 sequence length), and 3.89× memory saving during the decoding stage, with at most 0.47 WikiText-2 perplexity loss. QuaRot preserves 99% of the accuracy of zero-shot tasks and we show that our 6 and 8-bit quantization is lossless with simple round-to-nearest quantization.

# 2 Related Work

The majority of quantization schemes focus on compressing LLMs by using *weight-only quantization*, [[Frantar et al.,](#page-10-0) [2022,](#page-10-0) [Dettmers et al.,](#page-9-2) [2023,](#page-9-2) [Lin et al.,](#page-10-1) [2023,](#page-10-1) [Egiazarian et al.,](#page-10-2) [2024,](#page-10-2) [Tseng et al.,](#page-11-1) [2024]](#page-11-1). These methods downcast each weight into a low-precision representation and upcast it before the actual computation. The main computation is still performed in high precision. Several works show that, unlike weights, quantizing the activations is hard due to the outlier features [[Wei et al.,](#page-11-2) [2022,](#page-11-2) [Dettmers et al.,](#page-9-3) [2022,](#page-9-3) [Xiao et al.,](#page-11-3) [2023]](#page-11-3). For 8-bit case, LLM.int8() [[Dettmers et al.,](#page-9-3) [2022]](#page-9-3) identifies the outlier features during inference and keeps them in 16 bits which results in poor performance. SmoothQuant [[Xiao et al.,](#page-11-3) [2023]](#page-11-3) normalizes the features using some scaling factors from a calibration set, solving the issue for the 8-bit case at the cost of introducing extra hyper-parameters. For 4-bit quantization, recent studies identify the outlier features offline and keep them in high precision. Atom [[Zhao et al.,](#page-11-0) [2023]](#page-11-0) developed a complex kernel for mixed-precision MatMul in the presence of outliers while QUIK [[Ashkboos et al.,](#page-9-0) [2023]](#page-9-0) keeps the down-projection layer in 8 bits.

Two weight-only quantization methods, QuIP [[Chee et al.,](#page-9-4) [2024]](#page-9-4) and QuIP# [[Tseng et al.,](#page-11-1) [2024]](#page-11-1) have previously considered improving quantization by applying rotations. [Chee et al.](#page-9-4) [[2024]](#page-9-4) introduced the idea of *incoherence processing* which applies rotation matrices to the left and right of each weight matrix, as well as the Hessian, which is used in minimizing the weight-quantization objective. [Xi](#page-11-4)

<!-- page 3 -->

et al. [2023] uses a similar idea during training, using exact Hadamard transformations for each linear layer in the forward pass.

Finally, KV cache quantization is another line of research that aims to compress the cached keys and values during the generation phase. This is crucial for large batch size and long-context length generation as the KV cache will be the main memory bottleneck in such problems. Sheng et al. [2023] quantizes the KV cache using 4-bit group-wise quantization. KVQuant [Hooper et al., 2024] pushes this limit to 3-bit quantization and KIVI [Liu et al., 2024] shows promising results on 2-bit KV cache quantization. Such methods show that outliers also exist in the keys, and apply a set of complex ideas (like feature-wise quantization, non-uniform representation, and keeping high precision outliers) to recover the accuracy of a quantized KV cache.

In this work we also adopt the Hadamard transform to improve quantization of weights through incoherence processing. Instead of undoing the Hadamard transform during the forward pass, we adopt the computational invariance theorem from SliceGPT [Ashkboos et al., 2024] to fuse the transformations into the weights where possible. Instead of requiring two Hadamard transforms per weight-matrix in the forward pass, QuaRot requires just 1\frac{1}{2} Hadamard transforms per transformer layer. Computational invariance also means that the *activations* are incoherence-processed, enabling them to be effectively quantized. We also apply a similar technique to the attention block and quantize the KV cache in 4 bits with minimal accuracy loss.

# 3 Background

Here we introduce some mathematical concepts and notation that are necessary for QuaRot.

## 3.1 Orthogonal, Rotation and Hadamard Matrices

An orthogonal matrix \mathbf{Q} is a square matrix such that \mathbf{Q}\mathbf{Q}^{\top} = \mathbf{I}. In this work, we consider only real orthogonal matrices. A rotation matrix is an orthogonal matrix. A Hadamard matrix is an orthogonal matrix with entries drawing from \{+1,-1\}. A Walsh-Hadamard matrix is a square matrix of size d=2^n, with

$$
\mathbf{H}_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \quad \text{and} \quad \mathbf{H}_{2^n} = \mathbf{H}_2 \otimes \mathbf{H}_{2^{n-1}}. \tag{1}
$$

These identities give rise to the Walsh-Hadamard transform, which computes the matrix-vector product \mathbf{H}\boldsymbol{x} in \mathcal{O}(d\log_2(d)) operations.

For matrix sizes that are not 2^n, the existence of a Hadamard matrix is not guaranteed. A useful list of known Hadamard matrices is made available by Sloane [2024]. Where we require a Hadamard matrix of size d \neq 2^n, we factorize d = 2^n m, where m is the size of a known Hadamard matrix. Then we use a Kronecker construction \mathbf{H}_d = \mathbf{H}_{2^n} \otimes \mathbf{H}_m. This allows computation of \mathbf{H}_d \mathbf{x} in \mathcal{O}(d(m+n)) operations.

Following Tseng et al. [2024] we make use of *randomized* Hadamard matrices where convenient. Let s be a vector containing random draws from \{+1,-1\}, and \tilde{\mathbf{H}} = \mathbf{H} \operatorname{diag}(s). It is straightforward to see that \tilde{\mathbf{H}} is also an orthogonal matrix.

## 3.2 Incoherence Processing

The idea of *incoherence processing* was introduced by [Chee et al., 2024] in the context of weight normalization for weight-only LLM quantization. We define a weight matrix \mathbf{W} to be \mu-incoherent if

$$ \max(\mathbf{W}) \le \mu \|\mathbf{W}\|_F / \sqrt{mn} \tag{2} $$

where max is the element-wise max of the matrix, and mn is the number of elements. A weight matrix that has high incoherence is hard to quantize: the largest element is an outlier relative to the magnitude of the average element. Chee et al. [2024] showed that multiplying a weight matrix on the left and right by an orthogonal matrix can reduce the incoherence, making matrices easier to quantize. In this work we adopt a similar technique, multiplying weight matrices by orthogonal matrices to improve incoherence, though we add fewer operations to the forward pass. Importantly, we additionally apply incoherence processing to the activations, enabling improved weight and activation quantization. Figure 1 shows the effect of applying incoherence processing to the activations of LLAMA-2.

<!-- page 4 -->

## 3.3 Transformer structures

Large Language Models are neural networks with repeating attention and feed-forward layers. We introduce our notation through Figures 2 and 5, which show the construction of these blocks. We assume that the construction of the network is "pre-norm", in that each block is preceded by a LayerNorm or RMSNorm operation. We also assume that the feed-forward network uses a gated architecture, as in LLAMA-2, though our methodology is straightforwardly applied to MLP architectures also.

## 3.4 Computational Invariance

The computational invariance theorem [Ashkboos et al., 2024, Theorem 1] states that the weights and between-block activations in a transformer can be transformed using an orthogonal matrix with no change to the model output. Here we sketch the main idea. If \mathbf{W}_{in} is a weight matrix that appears on the left of a transformer block (i.e., \mathbf{W}_{gate}, \mathbf{W}_{up} in Figure 2, or \mathbf{W}_k, \mathbf{W}_q, \mathbf{W}_v in Figure 5) then we can multiply on the left by an orthogonal matrix \mathbf{Q}, and cancel out this effect by multiplying the output matrix (\mathbf{W}_{down}, \mathbf{W}_{out}) by \mathbf{Q}^{\top}. This applies despite the fact that RMSNorm is applied between the two blocks, so long as no re-scaling happens in the RMSNorm block (and in practice, we absorb any re-scaling into adjacent weight matrices first). Conceptually, this is because RMSNorm divides the activations by their norm, and applying a rotation \mathbf{Q} to the activations does not affect the norm. We have the commutation property

$$
RMSNorm(\mathbf{X}) = RMSNorm(\mathbf{X}\mathbf{Q}^{\top})\mathbf{Q},
(3)
$$

where we assume here that RMSNorm applied to each row of the activations \mathbf{X} as \mathbf{x}_i \leftarrow \mathbf{x}_i/\|\mathbf{x}_i\|. This means that multiplying an output matrix by \mathbf{Q}^{\top} makes the linear layer output \mathbf{X}\mathbf{Q}^{\top}, which is normalized and then passed into the next block whose input weight matrix is now \mathbf{Q}\mathbf{W}, and so *this* linear layer outputs the original activations without modification.

# 4 Method

QuaRot consists of two stages. In the first stage, the model weights are manipulated (in full precision), and two additional Hadamard operations are inserted into the model's forward pass. In the second stage, the weights are quantized using some existing method, and quantization operations are added to the forward pass to enable on-line quantization of the activations (and caches). By default, we use GPTQ [Frantar et al., 2022] for quantizing weights, whilst activations are quantized on-the-fly using a simple round-to-nearest scheme. Figures 3 and 6 show updated block diagrams for the forward pass with QuaRot modifications, including updated weight matrices, inserted blocks and the bit-width of weights and activations.

**Stage 1a: Weight Modification.** We first make use of computational invariance to multiply each weight matrix by an orthogonal matrix. To enable this, the linear parts of LayerNorm or RMSNorm are fused into adjacent weight matrices. Figure 3 shows how the feed-forward block of a transformer is modified by removing the scaling operation from RMSNorm (diag(\alpha)) and absorbing into the

<!-- page 5 -->

![RP29_Ashkboos_2024 fig03](../figures/RP29_Ashkboos_2024_fig03.jpg)
*Figure 3: QuaRot applied to a LLaMa-style FFN. The RMSNorm scaling (\alpha) has been absorbed into the weight matrices ((\alpha) is a diagonal matrix with RMSNorm parameters). The hidden state **X** has been rotated by **Q**, which is canceled out by the absorption of \mathbf{Q}^{\top} into the first two weight matrices. All weights are stored in INT4, and all activations immediately before the weights are also quantized to INT4. The result of the matmul between the INT4 weights and activations on a TensorCore is INT32, which we immediately cast (and scale) to FP16 which is the default precision of the model. Whilst the signal is still in FP16, we perform a single on-the-fly Hadamard transform before quantizing and computing a (modified) down-proj, which results in a rotated output **YQ**.*

subsequent weight matrices. We select a randomized Hadamard matrix with size that matches the hidden dimension of the model and pre- or post-multiply each weight matrix. In Figures 3 and 6 this matrix is denoted \mathbf{Q}. For example the key-projection weight matrix \mathbf{W}_k is modified as

$$
\mathbf{W}_k \leftarrow \mathbf{Q}^{\top} \operatorname{diag}(\boldsymbol{\alpha}) \mathbf{W}_k \,, \tag{4}
$$

and similarly for other weight matrices. Matrices that appear on the *output* side of a block are post-multipled by \mathbf{Q}.

This weight modification does not affect the output of the model (assuming sufficient precision) as per the computational invariance theorem [Ashkboos et al., 2024]. We note that the modified weights resemble the modifications used in QuIP# [Tseng et al., 2024], reducing the incoherence of the weights, though our modification does not require any additional processing at run-time. Additionally, the activation matrix passed between blocks of the transformer is also incoherence processed, becoming \mathbf{X} \leftarrow \mathbf{XQ}. Figure 1 shows the result of this processing: we see that the processed activations no longer contain any outliers.

**Stage 1b: Rotate FFN activations.** With the above weight-modifications in place, we have multiplied many weight matrices on one side by a Hadamard matrix and the activations have been changed. It remains to improve the quantization of the activations *within* each block, which we achieve by inserting on-line Hadamard operations.

We first insert a Hadamard operation into the feed-forward network, before the down-projection matrix. This operation is performed in full precision, and implemented using a fast kernel following Tseng et al. [2024]. This operation is implicitly reversed by fusing a Hadamard matrix into the down-projection matrix of the network: \mathbf{W}_{down} \leftarrow \mathbf{H}\mathbf{W}_{down}. Combined with the global matrix \mathbf{Q}, this means that the down-projection matrix now becomes \mathbf{H}\mathbf{W}_{down}\mathbf{Q} (see Figure 3).

**Stage 1c: Attention Value Projection.** Next, we apply an additional Hadamard operation to each attention block. This modification is partially on-line, and partially fused into the weight matrices as we will now detail.

First, note that in the computation of attention, the W_v and W_{out} matrices are implicitly multiplied together within each head. To see this, note that the attention computation consists of

$$
\mathbf{Y} = \operatorname{concat}[(\mathbf{P}_1 \mathbf{V}_1) \dots (\mathbf{P}_{n_b} \mathbf{V}_{n_b})] \mathbf{W}_{\text{out}} 
(5)
$$

$$
= \sum_{h=1}^{H} \mathbf{P}_h \mathbf{X} \mathbf{W}_v^{(h)} \mathbf{W}_{\text{out}}^{(h)} (6)
$$

where \mathbf{P}_h is a sequence-length sized square matrix computed by softmaxing keys and values, and \mathbf{V}_h = \mathbf{X}\mathbf{W}_v^{(h)} is the value matrix for one head. This presents an opportunity to perform additional

<!-- page 6 -->

processing on \mathbf{W}_v and \mathbf{W}_{\text{out}} using a Hadamard matrix \mathbf{H}_{d_h} which matches the dimension of each head:

$$
\mathbf{W}_{v}^{(h)} \leftarrow \mathbf{W}_{v}^{(h)} \mathbf{H}_{d_{h}}, \qquad \mathbf{W}_{\text{out}}^{(h)} \leftarrow \mathbf{H}_{d_{h}} \mathbf{W}_{\text{out}}^{(h)}. \tag{7}
$$

Substituting these modifications into equation (6), we see that the computed result of attention remains unchanged. Since the weights for each head are concatenated in the weight representation, we can equivalently perform a single Kronecker structured multiplication:

$$
\mathbf{W}_v \leftarrow \mathbf{W}_v(\mathbf{I} \otimes \mathbf{H}_{d_h}), \qquad \mathbf{W}_{\text{out}} \leftarrow (\mathbf{I} \otimes \mathbf{H}_{d_h}) \mathbf{W}_{\text{out}}. (8)
$$

This transformation has now been applied head-wise to the weight matrices, and results in computed activations (emitted by the block *multi-head attention*) rotated head-wise also. To complete a "full" Hadamard operation on the attention-activations, sharing the transform across heads, we make use of the identity

$$
\mathbf{H}_{n_h \times d_h} = (\mathbf{I} \otimes \mathbf{H}_{d_h})(\mathbf{H}_{n_h} \otimes \mathbf{I}) \tag{9}
$$

which holds when the number of heads n_h and the dimension of each head d_h are both powers of 2. Since we have already applied (\mathbf{I} \otimes \mathbf{H}_{d_h}) to both \mathbf{W}_v and \mathbf{W}_{\text{out}}, it remains to apply (\mathbf{H}_{d_h} \otimes \mathbf{I}) to \mathbf{W}_{\text{out}}, which results in a complete transformation of \mathbf{W}_{\text{out}} \leftarrow \mathbf{H}\mathbf{W}_{\text{out}}, and to insert a block into the forward pass that computes \mathbf{Z} \leftarrow \mathbf{Z}(\mathbf{H}_{n_h} \otimes \mathbf{I}) where \mathbf{Z} is the attention activation. This block is denoted *Hadamard heads* in Figure 6 and can be computed efficiently using a reshape to deal with the Kronecker structure, and a Walsh-Hadamard transform on the reshaped data.

**Stage 1d: Key Rotation.** Using the method above, we can successfully quantize the value vectors. However, key vectors in the attention module are also known to suffer from outliers [Hooper et al., 2024, Liu et al., 2024]. Similar to above, we can use a Hadamard rotation to alleviate this issue, allowing us to have a fully quantized KV cache. First note that the attention scores \mathbf{P}_1, \ldots, \mathbf{P}_h are computed as:

$$
\mathbf{Q} \leftarrow \operatorname{Pos}(\mathbf{XW}_q) = \operatorname{concat}[\operatorname{Pos}(\mathbf{Q}_1), \dots, \operatorname{Pos}(\mathbf{Q}_{n_h})] \tag{10}
$$

$$
\mathbf{K} \leftarrow \operatorname{Pos}(\mathbf{X}\mathbf{W}_k) = \operatorname{concat}[\operatorname{Pos}(\mathbf{K}_1), \dots, \operatorname{Pos}(\mathbf{K}_{n_h})] (11)
$$

$$
\mathbf{P}_h \leftarrow \operatorname{Softmax}(\alpha \operatorname{Pos}(\mathbf{Q}_h) \operatorname{Pos}(\mathbf{K}_h^{\top}) \odot \mathbf{M}), \tag{12}
$$

where \alpha is the Softmax scale usually set to \frac{1}{\sqrt{d_h}}, M is the attention mask (e.g., causal), and Pos denotes the positional embedding. Previously, positional embedding was only added before the first layer to the input, in which case Pos is an identity function. However, recent methods such as RoPE [Su et al., 2021] add position information directly to the key and query vectors.

We can now observe the same interaction between \mathbf{Q} and \mathbf{K} as we observed between \mathbf{W}_v and \mathbf{W}_{\text{out}}. However, the existence of Pos prevents us from directly fusing the Hadamard matrix into \mathbf{W}_q and \mathbf{W}_k. Therefore, we use online head-wise Hadamard rotation to rotate both the queries and keys. As a result, the computation of query and key matrices is altered as follows:

$$
\mathbf{Q} \leftarrow \operatorname{Pos}(\mathbf{XW}_q)(\mathbf{I} \otimes \mathbf{H}_{d_h}) = \operatorname{concat}[\operatorname{Pos}(\mathbf{Q}_1)\mathbf{H}_{d_h}, \dots, \operatorname{Pos}(\mathbf{Q}_{n_h})\mathbf{H}_{d_h}] (13)
$$

$$
\mathbf{K} \leftarrow \operatorname{Pos}(\mathbf{X}\mathbf{W}_k)(\mathbf{I} \otimes \mathbf{H}_{d_h}) = \operatorname{concat}[\operatorname{Pos}(\mathbf{K}_1)\mathbf{H}_{d_h}, \dots, \operatorname{Pos}(\mathbf{K}_{n_h})\mathbf{H}_{d_h}]. \tag{14}
$$

Since both queries and keys are rotated, the final attention scores \mathbf{P}_1, \dots, \mathbf{P}_h remain unchanged. We note that an alternative to the above process is caching the keys before applying the positional encoding. This approach (called Pre-RoPE Caching [Hooper et al., 2024]) needs the inverse rotation to be applied online before applying the positional encoding but removes the need to rotate the query vector. It also adds the overhead of rotating the keys and values for every query. Given that at the time of decoding there is a single query vector and many cached key vectors, we use Post-RoPE caching. This helps us to apply a Hadamard transformation on a single token at each decoding step.

Overall, our modifications to the forward pass, including the insertion of special Hadamard blocks and adjustments to the weights do not change the forward pass of the model. The effect is that the activations between blocks have been multiplied by a Hadamard matrix, and the activations within blocks are processed on-line using Hadamard transforms in a way that is undone by corresponding weight matrix modifications. We are now ready to quantize the weights and activations.

**Stage 2a: Weight Quantization.** We apply GPTQ [Frantar et al., 2022] to quantize the weights of the network. We note that after the above forward-pass modifications, any quantization method could be applied. In subsequent sections, we show that a simple round-to-nearest (RTN) scheme can be applied instead of GPTQ, at the cost of some accuracy.

<!-- page 7 -->

Stage 2b: Online Quantization Operations. With the weights quantized, we are ready to apply operations to the forward pass that quantize the activations. Following PyTorch implementation, we leave the computation of RMSNorm (without scaling) in FP32. We quantize the input of the linear layers using symmetric per-token (rows of the input matrix). During symmetric quantization, the row scales are computed by dividing the maximum absolute value of each token by 7 (largest representable number in INT4). We then divide each row to its corresponding scale and round the result to its nearest integer. The dequantization is also done by casting the INT32 output of GEMM into FP16, multiply the corresponding scale for the row (from input scales) and column (from weight scales).

Stage 2c: Quantized Attention. Attention is significantly memory bound for longer sequences and larger batch sizes. Having rotated both keys and values, we can successfully quantize the cache into low bit-width. This reduces the number of IO operations needed. We keep the queries in FP16 and use online softmax calculation similar to Flash Attention [[Dao et al.,](#page-9-5) [2022]](#page-9-5). After a segment of the KV vectors are loaded from the memory, we dequantize and compute the dot product in FP16.

# 5 Experimental Validation

Setup. We implement QuaRot using Hugging Face [[Wolf et al.,](#page-11-5) [2019]](#page-11-5) on top of the PyTorch framework [[Paszke et al.,](#page-10-8) [2019]](#page-10-8). To quantize the inputs, we use per-token symmetric quantization (a single scale for every row) with a constant clipping ratio of 0.9 in all our experiments. We quantize the KV caches using asymmetric quantization with a group size 128 with a constant clipping ratio of 0.95. For weight quantization, we use round-to-nearest (RTN) and GPTQ [[Frantar et al.,](#page-10-0) [2022]](#page-10-0) with per-column (also known as per-channel) symmetric quantization, where we extract the clipping ratio using a linear search over the squared error. We use 128 samples from WikiText-2 [[Merity et al.,](#page-10-9) [2016]](#page-10-9) training set with 2048 sequence length as the calibration set during GPTQ quantization. On a single NVIDIA A100 GPU, modifying LLAMA2-70B with QuaRot takes 5 minutes and quantizing the model with GPTQ takes a further 2 hours. We present LLAMA-3 results in Appendix [A.8.](#page-15-0)

Models, Tasks, and GPUs. We evaluate QuaRot on the LLAMA-2 family [[Touvron et al.,](#page-10-10) [2023]](#page-10-10) on both language generation and zero-shot tasks. We implement our low-level CUDA kernel to perform 4-bit matrix-multiplication using the CUTLASS [[NVIDIA,](#page-10-11) [2023]](#page-10-11) library. We use the FlashInfer [[Ye,](#page-11-6) [2023]](#page-11-6) library for implementing our KV cache quantization. As we target consumer-type GPUs, we evaluate all the performance experiments on NVIDIA RTX 3090 GPUs.

## 5.1 Accuracy Results

Language Generation Tasks. First, we evaluate the accuracy of QuaRot on the language generation task. Table [1](#page-7-0) shows the perplexity of LLAMA-2 models on WikiText-2 when we quantize the weights using GPTQ. We compare against 4-bit SmoothQuant [[Xiao et al.,](#page-11-3) [2023]](#page-11-3) and OmniQuant [[Shao](#page-10-12) [et al.,](#page-10-12) [2023]](#page-10-12). We also include the QUIK [[Ashkboos et al.,](#page-9-0) [2023]](#page-9-0) results when they keep all the layers (including down-projection) in 4 bits. QuaRot outperforms all previous work with at most 0.63 perplexity loss (0.47 on LLAMA2-70B model) without any re-training (as in OmniQuant) nor higher precision outlier features and asymmetric quantization (as in QUIK). We also apply group-wise quantization to compare against Atom [[Zhao et al.,](#page-11-0) [2023]](#page-11-0) on the same number of groups for weight and activations. In this setting, QuaRot doesn't need to keep any higher precision features and related operations (like re-ordering). QuaRot outperforms Atom with 0.1 perplexity points in the 7B model. On the 13B model, we get the same perplexity number as Atom.

Zero-Shot Tasks. Next, we focus on evaluating QuaRot on six important zero-shot tasks: PIQA [[Bisk et al.,](#page-9-6) [2020]](#page-9-6), WinoGrande [[Sakaguchi et al.,](#page-10-13) [2021]](#page-10-13), HellaSwag [[Zellers et al.,](#page-11-7) [2019]](#page-11-7), LAMBADA (OpenAI) [[Radford et al.,](#page-10-14) [2019]](#page-10-14), and Arc (Easy and Challenge) [[Clark et al.,](#page-9-7) [2018]](#page-9-7). We use the LM Evaluation Harness [[Gao et al.,](#page-10-15) [2021]](#page-10-15) with default parameters for our experiments. Table [2](#page-7-1) shows the accuracy of our scheme on the above tasks as well as the average score. On LLAMA-2 family, QuaRot preserves the accuracy with at most 4.18% average score loss (1.09% for 70B model).

## 5.2 Performance Analysis

We implement QuaRot using CUDA/12.1 on top of PyTorch and use CUTLASS for performing INT-4 matrix multiplication on TensorCore (where the results will be saved in an INT32 accumulator). In this section, we evaluate the performance of our kernels for both prefill and decoding steps on NVIDIA RTX 3090 GPU. We provide all our experiments on a single transformer block as the whole

<!-- page 8 -->

| Model | Method | PQ | WG | HS | A-e | A-c | LA | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LLAMA2-7B | FP16 | 79.11 | 69.06 | 75.99 | 74.58 | 46.25 | 73.90 | 69.82 |
| QuaRot | 76.77 | 63.77 | 72.16 | 69.87 | 40.87 | 70.39 | 65.64 |  |
| LLAMA2-13B | FP16 | 80.47 | 72.22 | 79.39 | 77.48 | 49.23 | 76.75 | 72.59 |
| QuaRot | 78.89 | 70.24 | 76.37 | 72.98 | 46.59 | 73.67 | 69.79 |  |
| LLAMA2-70B | FP16 | 82.70 | 77.98 | 83.84 | 80.98 | 57.34 | 79.58 | 77.07 |
| QuaRot | 82.43 | 76.24 | 81.82 | 80.43 | 56.23 | 78.73 | 75.98 |  |

![RP29_Ashkboos_2024 fig04](../figures/RP29_Ashkboos_2024_fig04.jpg)
*Figure 4: Performance of the QuaRot kernel on a single transformer block of LLAMA-2 models using NVIDIA RTX 3090 GPU. **Left**: For the speedup results, we evaluate using sequence length 2048 with different batch sizes. **Right**: Peak memory saving during decoding of 50 tokens with different prefill sequence lengths using batch size 16.*

model does not fit on our GPU cluster for large batch sizes. We provide more performance analysis of our kernels (as well as complete results) in Appendix A.10.

**Prefill Stage Performance Increases.** For the compute-bound prefill stage, we present the speedups of using QuaRot on 2048 sequence length with different batch sizes in Figure 4 **Left**. On LLAMA2-7B model, we get 1.97x-2.16x speedup over the FP16 implementation using our QuaRot kernel. The speedup increases with batch sizes as the computation will become a bottleneck in larger batch sizes. on LLAMA2-70B model, we get up to 3.33x speedup. Note that our performance results could be improved by optimizing our kernels (e.g., fusing the quantization operations into the MatMul).

**Decoding Stages Memory Saving.** Finally, we evaluate the memory improvement which is the main bottleneck of the decoding stage. Figure 4 **Right** shows the peak memory saving on LLAMA-2 models. We provide results for LLAMA2-7B and LLAMA2-70B models. In both models, we get at least 3.63x peak memory saving compared to FP16 case during the decoding stage. Note that the KV cache is larger in LLAMA2-7B model as the LLAMA2-70B uses grouped-query attention [Ainslie et al., 2023]. In the LLAMA2-7B model, the memory saving increases with the sequence length, resulting in up to 3.75x memory saving. on LLAMA2-70B model, we get 3.89x savings in almost all the cases. We expect these values to be larger for the whole model (instead of just the single layer

<!-- page 9 -->

| Method | LLAMA-2 |  |  |
| --- | --- | --- | --- |
| 7B | 13B | 70B |  |
| Baseline | 5.47 | 4.88 | 3.32 |
| QuaRot | 6.10 | 5.40 | 3.79 |
| QuaRot-256G | 5.98 | 5.28 | 3.63 |
| QuaRot-128G | 5.93 | 5.26 | 3.61 |
| QuaRot-64G | 5.88 | 5.25 | 3.58 |

here) since as the number of layers increases the effect of constant size objects in memory becomes much less significant.

## 5.3 Ablation Studies

To evaluate different aspects of QuaRot, we evaluate the use of Round-to-Nearest Weight Quantization, Group-wise Quantization (with different group sizes), and KV cache Quantization with different bit-width combinations (Appendix [A.3)](#page-13-0). In addition, we investigate the role of applying Hadamard transformation on the Weight-only Quantization schemes (Appendix [A.4)](#page-13-1) as well as using Random Orthogonal Matrices (Appendix [A.5)](#page-14-0) instead of Hadamard matrices. Finally, we evaluate the accuracy of our quantized models when we apply FP16 Hadamard Transformation (Appendix [A.7)](#page-14-1).

Round-to-Nearest Weight Quantization. GPTQ is our default choice for weight quantization in QuaRot. Here, we study the role of quantizing the weights using Round-to-Nearest (RTN). Table [3](#page-8-0) shows that applying RTN weight quantization fully maintains the FP16 model accuracy in 8 bits. We note that RTN does not need any calibration set or hyper-parameter during the quantization. Comparing Table [3](#page-8-0) and [2,](#page-7-1) we conclude that in 4 bits, the gap between QuaRot-RTN and QuaRot-GPTQ decreases when the model size is increased (2.27 on LLAMA2-7B and 0.34 on LLAMA2-70B ) showing that GPTQ is a better option in smaller models. For more detailed results see Appendix [A.6.](#page-14-2)

Group-wise Quantization. Table [4](#page-8-1) shows the accuracy of applying QuaRot with various group-sizes for the activations and weights. The results show a clear trade-off between the accuracy and the group-sizes: smaller group-sizes give better accuracy (but require more bits to store scales for each group and more complex matrix-multiplication kernels).

# 6 Conclusion

We introduce QuaRot: a method which uses Hadamard matrices to eliminate outliers in the activations and KV cache of pre-trained LLMs, enabling end-to-end 4-bit quantization for the first time (to the best of our knowledge). Quantizing LLAMA2-70B to 4 bits with QuaRot maintains 99% of the

<!-- page 10 -->

downstream task performance of the FP16 baseline, with a 2.16× speedup on RTX 3090 GPUs during the prefill stage (and up to 3.39× memory saving during the decoding stage). Quantizing all LLAMA-2 models to 6 and 8 bits is lossless.

Opportunities to build on QuaRot include quantizing the residuals and extending the method to mixture-of-experts architectures. In terms of hardware, end-to-end INT4 inference with QuaRot could be exploited to give similar speedups as that of the recently announced NVIDIA B200 GPU architecture, while being much cheaper to implement compared to the floating point (FP4) format.

## References

- Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, Alon Benhaim, Misha Bilenko, Johan Bjorck, Sébastien Bubeck, Martin Cai, Caio César Teodoro Mendes, Weizhu Chen, Vishrav Chaudhary, Parul Chopra, Allie Del Giorno, Gustavo de Rosa, Matthew Dixon, Ronen Eldan, Dan Iter, Amit Garg, Abhishek Goswami, Suriya Gunasekar, Emman Haider, Junheng Hao, Russell J. Hewett, Jamie Huynh, Mojan Javaheripi, Xin Jin, Piero Kauffmann, Nikos Karampatziakis, Dongwoo Kim, Mahoud Khademi, Lev Kurilenko, James R. Lee, Yin Tat Lee, Yuanzhi Li, Chen Liang, Weishung Liu, Eric Lin, Zeqi Lin, Piyush Madan, Arindam Mitra, Hardik Modi, Anh Nguyen, Brandon Norick, Barun Patra, Daniel Perez-Becker, Thomas Portet, Reid Pryzant, Heyang Qin, Marko Radmilac, Corby Rosset, Sambudha Roy, Olatunji Ruwase, Olli Saarikivi, Amin Saied, Adil Salim, Michael Santacroce, Shital Shah, Ning Shang, Hiteshi Sharma, Xia Song, Masahiro Tanaka, Xin Wang, Rachel Ward, Guanhua Wang, Philipp Witte, Michael Wyatt, Can Xu, Jiahang Xu, Sonali Yadav, Fan Yang, Ziyi Yang, Donghan Yu, Chengruidong Zhang, Cyril Zhang, Jianwen Zhang, Li Lyna Zhang, Yi Zhang, Yue Zhang, Yunan Zhang, and Xiren Zhou. Phi-3 technical report: A highly capable language model locally on your phone, 2024.
- Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. *arXiv preprint arXiv:2305.13245*, 2023.
- Saleh Ashkboos, Ilia Markov, Elias Frantar, Tingxuan Zhong, Xincheng Wang, Jie Ren, Torsten Hoefler, and Dan Alistarh. Towards end-to-end 4-bit inference on generative large language models. *arXiv preprint arXiv:2310.09259*, 2023.
- Saleh Ashkboos, Maximilian L Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, and James Hensman. Slicegpt: Compress large language models by deleting rows and columns. *arXiv* *preprint arXiv:2401.15024*, 2024.
- Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language. In *Thirty-Fourth AAAI Conference on Artificial* *Intelligence*, 2020.
- Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, and Christopher M De Sa. Quip: 2-bit quantization of large language models with guarantees. *Advances in Neural Information Processing Systems*, 36, 2024.
- Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. *ArXiv*, abs/1803.05457, 2018. URL [https://api.semanticscholar.org/CorpusID:](https://api.semanticscholar.org/CorpusID:3922816) [3922816](https://api.semanticscholar.org/CorpusID:3922816).
- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In *Advances in Neural Information Processing* *Systems*, 2022.
- Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale. *Advances in Neural Information Processing Systems*, 35: 30318–30332, 2022.
- Tim Dettmers, Ruslan Svirschevski, Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashkboos, Alexander Borzunov, Torsten Hoefler, and Dan Alistarh. Spqr: A sparse-quantized representation for near-lossless llm weight compression. *arXiv preprint arXiv:2306.03078*, 2023.

<!-- page 11 -->

- Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, and Dan Alistarh. Extreme compression of large language models via additive quantization. *arXiv preprint* *arXiv:2401.06118*, 2024.
- Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate post-training quantization for generative pre-trained transformers. *arXiv preprint arXiv:2210.17323*, 2022.
- Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, et al. A framework for few-shot language model evaluation. *Version v0. 0.1. Sept*, 2021.
- Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W Mahoney, Yakun Sophia Shao, Kurt Keutzer, and Amir Gholami. Kvquant: Towards 10 million context length llm inference with kv cache quantization. *arXiv preprint arXiv:2401.18079*, 2024.
- Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, and Song Han. Awq: Activationaware weight quantization for llm compression and acceleration. *arXiv preprint arXiv:2306.00978*, 2023.
- Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. Kivi: A tuning-free asymmetric 2bit quantization for kv cache. *arXiv preprint* *arXiv:2402.02750*, 2024.
- Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models, 2016.
- NVIDIA. Nvidia cutlass library, 2023. URL [https://github.com/NVIDIA/cutlass/](https://github.com/NVIDIA/cutlass/).
- Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. PyTorch: An imperative style, high-performance deep learning library. *Advances in neural information processing systems*, 32, 2019.
- Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.
- Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. *Communications of the ACM*, 64(9):99–106, 2021.
- Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, and Ping Luo. Omniquant: Omnidirectionally calibrated quantization for large language models. *arXiv preprint arXiv:2308.13137*, 2023.
- Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Beidi Chen, Percy Liang, Christopher Ré, Ion Stoica, and Ce Zhang. Flexgen: High-throughput generative inference of large language models with a single gpu. In *International Conference on Machine Learning*, pages 31094–31116. PMLR, 2023.
- Neil J A Sloane. A library of hadamard matrices, 2024. URL [http://neilsloane.com/](http://neilsloane.com/hadamard/) [hadamard/](http://neilsloane.com/hadamard/).
- Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. *CoRR*, abs/2104.09864, 2021. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2104.09864) [2104.09864](https://arxiv.org/abs/2104.09864).
- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra,

<!-- page 12 -->

- Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023.
- Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, and Christopher De Sa. Quip#: Even better llm quantization with hadamard incoherence and lattice codebooks. *arXiv preprint* *arXiv:2402.04396*, 2024.
- Xiuying Wei, Yunchen Zhang, Xiangguo Zhang, Ruihao Gong, Shanghang Zhang, Qi Zhang, Fengwei Yu, and Xianglong Liu. Outlier suppression: Pushing the limit of low-bit transformer language models. *Advances in Neural Information Processing Systems*, 35:17402–17414, 2022.
- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. Huggingface's transformers: State-of-the-art natural language processing. *arXiv preprint arXiv:1910.03771*, 2019.
- Haocheng Xi, Changhao Li, Jianfei Chen, and Jun Zhu. Training transformers with 4-bit integers. *Advances in Neural Information Processing Systems*, 36:49146–49168, 2023.
- Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models. In *International* *Conference on Machine Learning*, pages 38087–38099. PMLR, 2023.
- Zihao Ye. FlashInfer: Kernel Library for LLM Serving. [https://github.com/flashinfer-ai/](https://github.com/flashinfer-ai/flashinfer) [flashinfer](https://github.com/flashinfer-ai/flashinfer), 2023.
- Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? *arXiv preprint arXiv:1905.07830*, 2019.
- Yilong Zhao, Chien-Yu Lin, Kan Zhu, Zihao Ye, Lequn Chen, Size Zheng, Luis Ceze, Arvind Krishnamurthy, Tianqi Chen, and Baris Kasikci. Atom: Low-bit quantization for efficient and accurate llm serving. *arXiv preprint arXiv:2310.19102*, 2023.

<!-- page 13 -->

## A Appendix

## A.1 OuaRot on the Attention Module

Figure 5 shows the original attention module in large language models with RoPE. The input of the attention module is already rotated using the randomized Hadamard matrix **Q** (see Section 4) and in the first step, we fuse the inverse of such matrices into the input linear layers of the attention. In the next step, we fuse the exact Hadamard matrices on each block of the columns (proportional to each head) on the V_projection layer to make sure that the Values will be rotated at the output of that layer. In the next step, we apply exact Hadamard transformations on the Keys and Queries and quantize the KV after RoPE operation (note that the Keys and Queries Hadamard transformations will be canceled during the attention operation). Finally, we apply another Hadamard transformation between heads before Out_projection layer and fuse the inverse into the weights. Figure 6 shows the result of applying QuaRot on the attention module.

![RP29_Ashkboos_2024 fig05](../figures/RP29_Ashkboos_2024_fig05.jpg)
*Figure 5: Flow diagram of a self-attention block as used in most LMs, including the pre-positioned RMSNorm. Solid arrows represent flow during training, prefill and inference of each token. Dashed arrows show access to and from the KV cache, used at generation-time. The RoPE block computes relative positional embeddings.*

![RP29_Ashkboos_2024 fig06](../figures/RP29_Ashkboos_2024_fig06.jpg)
*Figure 6: QuaRot applied to an attention component. The RMSNorm scaling \alpha is absorbed into the input weight matrices, and the hidden state has been rotated by \mathbf{Q} in the same way as for the FFN block (see previous figure). Colored labels show the bit-width of each flow, and dashed lines show the flow to/from the KV cache.*

## A.2 Clipping Ratio Ablation

We use the clipping ratio for both weights and activations during the quantization. During the weight quantization, we apply a linear search over the MSE error to extract the best clipping ratio for each

<!-- page 14 -->

column of the weight matrix. However, this is not possible as we quantize the inputs on the fly during the inference and we need to use a constant clipping ratio for such quantization. We conclude that using 0.95 and 0.9 are suitable during asymmetric (KV cache) and symmetric (inputs) quantization which matches the finding from [[Zhao et al.,](#page-11-0) [2023]](#page-11-0).

## A.3 KV Cache Quantization Ablation

We keep the rest of the model (including weights and activations) in high precision and apply our group-wise asymmetric quantization (with group-size 128) with various precision to keys and values. Table [6](#page-13-2) shows the results of using various precision during KV cache quantization. The results show a negligible (at most 0.21) perplexity degradation up to 3-bit KV cache (0.07 for LLAMA2-70B model). In addition, by comparing the 3 and 4-bit quantization, we can see that compared to the values, keys are more sensitive to quantization as keeping the keys in 4-bits and values in 3-bits has 0.03 perplexity loss (0.18 for 3-bit keys and 4-bit values) on the LLAMA2-7B model. This matches the previous study on KV cache quantization [[Hooper et al.,](#page-10-4) [2024,](#page-10-4) [Liu et al.,](#page-10-5) [2024]](#page-10-5). The results show that using 3-bit KV-caches results in a better accuracy (5.68 on LLAMA2-7B model) compared to keeping the keys in 4-bits and quantizing the values using 2-bits (with 5.75 perplexity on LLAMA2-7B model).

## A.4 Weight-only Quantization Ablation

QuaRot improves the quality of quantized models by removing the outlier features during the Hadamard transformations. As we fuse the Hadamard matrices into the weights, we study the role of these transformations for weight-only quantization (we keep the rest of the data-types in FP16). Table [7](#page-14-3) shows the WikiText-2 perplexity results with asymmetric quantization. Using GPTQ quantization, QuaRot improves the perplexity by up to 2.65 in 4 bits. In addition, applying QuaRot improves the quality more in lower precision (2-3 bits) in all models. QuaRot also improves the RTN quantization up to 0.24 perplexity points. GPTQ still has a lower perplexity in 2-3 bits. However, applying QuaRot improves the quality of GPTQ in 2 bits to a non-trivial value (5.6 on the LLAMA2-70B model).

<!-- page 15 -->

## A.5 Random Orthogonal Matrices Ablation

QuaRot fuses Hadamard transformations into weight matrices to eliminate outliers. However, due to the computational invariance property in LLMs, any orthogonal matrix can be fused to the model and we only need to apply an online 1\frac{1}{2} Hadamard transformations in each layer (see Section 4). Here, we study the use of random orthogonal matrices in QuaRot. We start with a uniformly random matrix and apply QR decomposition to make it orthogonal before fusing it into the weights.

Table 8 shows the results of applying random orthogonal matrices on LLAMA-2 models. Random orthogonal matrices are not as good as random Hadamard transformations and we have up 1.35 perplexity gap on LLAMA2-7B. However, as the model size increases, the gap decreases, resulting in a perplexity change of 0.28 in the LLAMA2-70B model. Note that using the above matrices does not change the computation as we still use a fast Hadamard kernel for the down-projection and out-projection layers.

## A.6 Round-to-Nearest Weight Quantization: Detailed Results

Table 9 shows the detailed results of QuaRot with GPTQ and round-to-nearest (RTN) weight quantization for both 6 and 8 bits on various tasks for LLAMA-2 models.

## A.7 FP16 Hadamard Transformation Ablation

We use FP32 online Hadamard transformation across all our experiments. Table 10 shows the results of using FP16 Hadamard transformation during the inference (for *down-projection* and *out-projection* layers). On LLAMA2-7B model, the results show <0.1 perplexity change on WikiText-2 and <0.6% averaged accuracy change on the zero-shot tasks, which we consider as noise. On LLAMA2-13B model, different Hadamard precisions have the same perplexities with 0.07% difference in the averaged zero-shot results. We conclude that the model will not be changed using different Hadamard precision.

<!-- page 16 -->

## A.8 LLAMA-3 Results

In this section, we show the accuracy of applying QuaRot for quantizing the LLAMA3-8B and LLAMA3-70B models. Table [11](#page-16-1) shows the WikiText-2 perplexity of quantizing the LLAMA-3 models with QuaRot using 4-bit quantization. Compared to Table [1,](#page-7-0) we conclude that LLAMA-3 is more sensitive to quantization as we can see a higher gap between the quantized and FP16 models. Table [12](#page-16-2) shows the accuracy results of those models on zero-shot tasks.

<!-- page 17 -->

| Model | Method | PQ | WG | HS | A-e | A-c | LA | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LLAMA3-8B | FP16 | 80.74 | 72.77 | 79.06 | 77.82 | 53.33 | 75.63 | 73.22 |
| QuaRot | 75.14 | 65.82 | 72.94 | 68.01 | 43.34 | 65.81 | 65.18 |  |
| LLAMA3-70B | FP16 | 84.66 | 80.51 | 84.89 | 85.86 | 64.25 | 79.47 | 79.94 |
| QuaRot | 78.07 | 69.30 | 77.33 | 73.44 | 47.53 | 69.57 | 69.21 |  |

## A.9 Phi-3-mini-4k-instruct Results

In this section, we show the accuracy of applying QuaRot for quantizing the Phi-3-mini-4k-instruct model [[Abdin et al.,](#page-9-9) [2024]](#page-9-9). Table [13](#page-16-3) shows the accuracy results of the model in terms of perplexity and on zero-shot tasks.

## A.10 Performance Analysis

We implement the attention mechanism using three routines: 1) Init: During the prefill stage, this routine initializes the cache from all the key and value vectors in the prefill. The attention output during prefill is computed directly using Flash Attention [[Dao et al.,](#page-9-5) [2022]](#page-9-5) since we already have access to dequantized keys and values. 2) Append: During decoding, this routine is called first to quantize the current keys and values and append them to the cache. 3) Decode: Finally, this routine is called during decoding with the current query vector. The routine computes the attention output using a quantized implementation of flash attention which can load the quantized cache and compute the final value vector.

4-bit Linear and Attention Layers. We benchmark our 4-bit linear layer which involves 4-bit matrix multiplication. For a given input of FP16, the layer optionally computes the Hadamard operation, then calls the quantization kernel to quantize and save the input in a sub-byte format. In the next step, the quantized weights and input are passed to the CUTLASS 4-bit GEMM kernel. Finally, the output is dequantized and cast back to FP16. Figure [7](#page-17-0) shows the speedup of our 4-bit layer for different layer sizes where the layer sizes match the FFN linear layer sizes in LLAMA-2 models.

<!-- page 18 -->

![RP29_Ashkboos_2024 fig07](../figures/RP29_Ashkboos_2024_fig07.jpg)
*Figure 7: Performance of 16-bit and 4-bit linear layer for 2048 sequence lengths with and without online Hadamard transformation on a NVIDIA RTX 3090 GPU, averaged over 1000 runs. The matrix sizes correspond to the linear layer sizes in LLAMA-2 FFN blocks (i.e. Wdown). Here the batch size is 1, but the performance ratio holds for larger batches (see Table [14)](#page-18-0).*

Our 4-bit linear layer gets 3.2x speedup relative to FP16 in the LLAMA2-7B model, and 4.3x on the LLAMA2-70B model. These numbers are for a batch size of 1, we find that scaling is approximately linear with batch size: more results in Table [14.](#page-18-0) We include the runtime with and without Hadamard operations, as Wup and Wgate do not require Hadamard transforms, whilst Wdown does. We see that the Hadamard transform adds very little overhead to the forward pass at most 7% overhead.

We also compare the speed of performing append and decode routines for a single token given a cache of size 2047. This is equivalent to the cost of decoding the 2048-th token in a sequence. The comparison between the speed of FP16 and INT4 for different batch sizes and layer sizes is reported in Table [15.](#page-19-0) For the layer size used in LLAMA2-7B , our 4-bit implementation gets up to 1.72x improvement in speed for the larger batch sizes (e.g. from 16 onwards). The 4-bit cache is slower than FP16 for smaller batch sizes (e.g. up to 8). Note that this is intuitive as the main benefit of the 4-bit cache is reducing the I/O cost. A speed up is only visible if this reduction is more significant than the quantization overhead which happens for either larger batch sizes or longer sequences.

Table [14](#page-18-0) shows the results of benchmarking our 4-bit linear layer. The layer sizes are extracted based on the linear layer sizes in LLAMA-2 models (for out-projection and down-projections). We apply both FP16 and FP32 Hadamard transformations and show the runtime on NVIDIA RTX GPU using 2048 sequence lengths. Table [15](#page-19-0) shows the results of decoding a single token in the attention layer when we apply KV-cache quantization. We extract the size of the attention layer based on the LLAMA-2 models.

<!-- page 19 -->

<!-- page 20 -->

Tables [16](#page-19-1) and [17](#page-20-0) show the detailed speedups and memory saving of a single transformer block for QuaRot on LLAMA2-7B model using NVIDIA RTX 3090 GPU.

<!-- page 21 -->
