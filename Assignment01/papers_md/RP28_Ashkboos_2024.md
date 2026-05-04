<!-- RP28_Ashkboos_2024 | source: papers_json/RP28_Ashkboos_2024/ -->

## SLICEGPT: COMPRESS LARGE LANGUAGE MODELS BY DELETING ROWS AND COLUMNS

Saleh Ashkboos†∗ ETH Zurich

saleh.ashkboos@inf.ethz.ch

Marcelo Gennari do Nascimento

Microsoft

marceloge@microsoft.com

James Hensman Microsoft Research jameshensman@microsoft.com

Maximilian L. Croci† Microsoft Research t-mcroci@microsoft.com

Torsten Hoefler ETH Zurich torsten.hoefler@inf.ethz.ch

## ABSTRACT

Large language models have become the cornerstone of natural language processing, but their use comes with substantial costs in terms of compute and memory resources. Sparsification provides a solution to alleviate these resource constraints, and recent works have shown that trained models can be sparsified post-hoc. Existing sparsification techniques face challenges as they need additional data structures and offer constrained speedup with current hardware. In this paper we present SliceGPT, a new post-training sparsification scheme which replaces each weight matrix with a smaller (dense) matrix, reducing the embedding dimension of the network. Through extensive experimentation we show that SliceGPT can remove up to 25% of the model parameters (including embeddings) for LLAMA-2 70B, OPT 66B and Phi-2 models while maintaining 99%, 99% and 90% zero-shot task performance of the dense model respectively. Our sliced models run on fewer GPUs and run faster without any additional code optimization: on 24GB consumer GPUs we reduce the total compute for inference on LLAMA-2 70B to 64% of that of the dense model; on 40GB A100 GPUs we reduce it to 66%. We offer a new insight, computational invariance in transformer networks, which enables SliceGPT and we hope it will inspire and enable future avenues to reduce memory and computation demands for pre-trained models. Code is available at: [https://github.com/microsoft/TransformerCompression](https://github.com/microsoft/TransformerCompression) .

# 1 INTRODUCTION

Large language models (LLMs) are neural networks with billions of parameters, trained on trillions of tokens [(Zhao et al.,](#page-13-0) [2023)](#page-13-0). The cost of training an LLM has caused a shift to re-using pre-trained models for multiple tasks, the *foundation model* paradigm. The size of LLMs makes deploying a pre-trained model an expensive undertaking. Many models require multiple GPUs to be able to compute a prediction, and because the models are autoregressive, multiple forward passes of the neural network are needed to generate text responses. It is therefore of widespread interest to reduce the computational requirements of these models, usually performed via post-training techniques referred to as *model compression*.

A majority of model compression techniques fall into one of four categories: distillation, tensor decomposition (which includes low-rank factorization), pruning and quantization [(Hoefler et al.,](#page-11-0) [2021;](#page-11-0) [Gholami et al.,](#page-11-1) [2021;](#page-11-1) [Zhu et al.,](#page-13-1) [2023;](#page-13-1) [Gupta & Agrawal,](#page-11-2) [2021)](#page-11-2). In this work we focus on pruning,

> ^∗^Work completed as an intern at Microsoft.

> ^†^Equal contribution.

<!-- page 2 -->

## ^X^ ^W^ Unstructured sparsity ^X^ ^W^ 2:4 Structured sparsity XQ ^Q^^⊤^^W^ Slicing (ours)

*Figure 1: Matrix multiplication of the signal X and a weight matrix W under different types of sparsity. Left: unstructured sparsity, where some elements of W are zero, and X is dense. Middle: 2:4 structured sparsity, where each block of four weight matrix entries contains two zeros, and X is dense. Right: SliceGPT, where after introducing transformation Q, all the sparsity is arranged to the bottom rows of W and the corresponding columns of X are removed.*

though we hope that our methodology may influence future work on other areas. Whilst pruning methods have been around for some time, many approaches require recovery fine-tuning (RFT) after pruning to maintain performance, making the overall process an expensive and hard-to-scale task. With SliceGPT we compress large models using a single GPU in just a few hours and maintain competitive performance on generation and downstream tasks even without RFT.

Pruning methods work by setting some elements of the weight matrices in an LLM to zero, and (optionally) updating the surrounding elements of the matrix to compensate. The result is a sparse pattern which means that some floating point operations can be skipped in the matrix multiplications required in the forward pass of the neural network. The relative speedup of the operations depends on the level of sparsity and the sparsity pattern: more structured sparsity is associated with more computational gain. In contrast to other pruning methods, SliceGPT prunes away (slices off!) entire rows or columns of the weight matrices. Before slicing, we perform a single transformation of the network which leaves the predictions invariant, but allows the slicing to have only a small effect.

The result is that weight matrices are smaller, and the signals passed between blocks of the neural network are smaller too: we reduce the *embedding dimension* of the neural network.

Figure [1](#page-1-0) compares our approach with existing sparsity methods. Our contributions are as follows:

- 1. We introduce the idea of *computational invariance*: we show that we can apply orthogonalmatrix transformations to each weight matrix in a transformer without changing the model.
- 2. We use this to edit each block in a transformer architecture, such that we are projecting the signal matrix[1](#page-1-1) between blocks onto its own principal components. We remove columns or rows of the transformed weight matrices to reduce the model size. We call the transformation and removal of weights SliceGPT.
- 3. We conduct multiple experiments on OPT [(Zhang et al.,](#page-13-2) [2022)](#page-13-2) and LLAMA-2 [(Touvron](#page-12-0) [et al.,](#page-12-0) [2023)](#page-12-0) LLMs, demonstrating that SliceGPT is able to compress these models by up to 30% with superior perplexity to the state of the art 2:4 scheme. On downstream tasks we additionally experiment with Phi-2 and show that all models can be sliced by up to 30% while maintaining >90% of the dense performance.

# 2 BACKGROUND

In this section, we first describe some necessary background on transformer architectures, which allows us to introduce notation which we will use to prove our main results. Then we describe related work on sparsification for compressing such architectures.

> ^1^The signal matrix is sometimes referred as activation matrix.

<!-- page 3 -->

## 2.1 Transformer Networks

Transformer networks (Vaswani et al., 2017) are a class of neural networks that have been shown to be effective at a wide range of tasks including language modeling. The transformer architecture is composed of a series of layers, each of which is composed of a multi-head self-attention block followed by a feed-forward network block. Between each block, there is a LayerNorm (Ba et al., 2016) (or RMSNorm (Zhang & Sennrich, 2019)) block. Figure 2 illustrates part of a transformer network: an attention block connected to a Feed Forward Network (FFN) block through a LayerNorm block, with residual connections. The following describes the operations of each component (ignoring dropout, which is not applied post-training).

**Embeddings** Let D be the embedding dimension of our transformer, N be the sequence length. The transformer model takes as input a sequence of token IDs and position IDs, and uses them to index the embedding matrices, producing the initial signal \mathbf{X} with shape N \times D. In what follows we consider, without loss of generality, a single embedding matrix \mathbf{W}_{\text{embd}} indexed by input sequence s.

**LayerNorm** After embeddings, the signal matrix is passed through a LayerNorm operation, which subtracts the mean from each row of the matrix, divides the row by its standard deviation, rescales (columnwise), and adds an offset. We write the LayerNorm block as

$$
LayerNorm(\mathbf{X}) = RMSNorm(\mathbf{X}\mathbf{M})diag(\boldsymbol{\alpha})\sqrt{D} + \mathbf{1}_{N}\boldsymbol{\beta}^{\top} 
(1)
$$

where RMSNorm(X) applies ^2x \leftarrow x/\|x\| to each row of X. The vector parameter \alpha and offset (vector) parameter \beta are learned independently at each LayerNorm instance. The constant matrix \mathbf{M} = \mathbf{I} - \frac{1}{D}\mathbf{1}\mathbf{1}^{\top} is a D \times D matrix which subtracts the mean from each row of X.

Attention Blocks The attention block has four matrices: \mathbf{W}_k, \mathbf{W}_q, \mathbf{W}_v and \mathbf{W}_o, each of dimension D \times D. The input signal arriving into the block is projected into the Key (\mathbf{X}\mathbf{W}_k), Query (\mathbf{X}\mathbf{W}_q), and Value (\mathbf{X}\mathbf{W}_v) matrices, which are then split into multiple *heads*. A nonlinear operation is applied at each head before the signals are combined and multiplied by the output weight matrix \mathbf{W}_o. Since the first three weight matrices are applied separately to the inputs, we can concatenate them and perform a single matrix multiplication (denoted by the white box around these matrices in Figure 2). We can consider the concatenation of these matrices to be a single linear layer, which we denote \mathbf{W}_{\text{in}}. We also refer to the output matrix as \mathbf{W}_{\text{out}}. We treat the attention block as \sigma(\mathbf{X}\mathbf{W}_{\text{in}} + \mathbf{b}_{\text{in}})\mathbf{W}_{\text{out}} + \mathbf{b}_{\text{out}}^3, where \sigma represents the multi-head attention operation.

FFN Blocks The other type of block that appears in transformer architectures is a Feed Forward Network (FFN) block. In many cases, this is a Multi-layer Perceptron (MLP), which consists of a linear layer \mathbf{W}_1, followed by an element-wise operation \sigma, followed by a second linear layer: \sigma(\mathbf{X}\mathbf{W}_1+b_1)\mathbf{W}_2+b_2. Some architectures have adopted the gated format, where an additional matrix is used, and the operation is (\sigma(\mathbf{X}\mathbf{W}_1+b_1)\circ(\mathbf{X}\mathbf{W}_2))\mathbf{W}_3, where \circ is an element-wise product. Much like the first three linear layers in the attention module, we can consider the concatenation of \mathbf{W}_1 and \mathbf{W}_2 to be a single linear operation, and denote it \mathbf{W}_{in}. We can therefore denote the operation of MLP or gated FFN layers as \sigma(\mathbf{X}\mathbf{W}_{in})\mathbf{W}_{out}, where \sigma takes a different meaning to that in an attention.

**Language Modelling (LM) Head** All of the transformer networks to which we apply SliceGPT in this paper have a decoder-only structure following (Radford et al., 2018): after multiple layers applying alternating attention and FFN blocks, a head block computes logits which are used to compute the loss during training and token prediction on deployment. The head operation is \mathbf{XW}_{\text{head}} + b_{\text{head}}, where \mathbf{X} is the output of the last transformer block.

Forward pass Once the model is trained and all of the parameters are set, the computations required in a transformer network to produce predictions involve passing signal matrices from one block to the next until the head node is reached. Since we are able to define both FFN and attention blocks in the form \sigma(\mathbf{X}\mathbf{W}_{in} + \mathbf{b}_{in})\mathbf{W}_{out} + \mathbf{b}_{out}, where we understand that \sigma represents either a point-wise or multi-head-attention nonlinearity, we are able to describe the forward pass using Algorithm 1.

> ^&^lt;sup>2</sup>In some implementations an RMSNorm block may contain scale parameters. We consider these to be special instances of LayerNorm and handle them accordingly.

> ^&^lt;sup>3</sup>For ease of notation here and throughout this paper, we abuse notation slightly and omit the broadcasting of the bias terms across the sequence length dimension. The complete notation for the operation of an attention block is \sigma(\mathbf{X}\mathbf{W}_{\text{in}} + \mathbf{1}_N \boldsymbol{b}_{\text{in}}^{\mathsf{T}})\mathbf{W}_{\text{out}} + \mathbf{1}_N \boldsymbol{b}_{\text{out}}^{\mathsf{T}}.

<!-- page 4 -->

## Algorithm 1 The forward pass of a transformer network

```
Require: \{\mathbf{W}_{\text{in}}^{\ell}, \boldsymbol{b}_{\text{in}}^{\ell}, \mathbf{W}_{\text{out}}^{\ell} \boldsymbol{b}_{\text{out}}^{\ell}\}_{\ell=1}^{L}
 // weights and biases of FFN and attention blocks
Require: \{\sigma_\ell\}_{\ell=1}^L
 // nonlinearity associated with each block
Require: \{\text{Norm}_{\ell}\}_{\ell=0}^{L}
 // LayerNorm or RMSNorm instances to perform between blocks
Require: \mathbf{W}_{\text{embd}}, \mathbf{W}_{\text{head}}, \boldsymbol{b}_{\text{head}}
 // embedding and head matrices
Require: s
 // input sequence
 1: \mathbf{X} \leftarrow \mathbf{W}_{\text{embd}}[\boldsymbol{s},:]
 // index embeddings
 2:\; \mathbf{X} \leftarrow Norm_0(\mathbf{X})
 // normalize
 3: for \ell = 1 \dots L do
 \mathbf{Z} \leftarrow \sigma_{\ell} (\mathbf{X} \mathbf{W}_{\text{in}}^{\ell} + \boldsymbol{b}_{\text{in}}^{\ell}) \mathbf{W}_{\text{out}}^{\ell} + \boldsymbol{b}_{\text{out}}^{\ell}
 // apply FFN or attention
 \mathbf{X} \leftarrow \text{Norm}_{\ell}(\mathbf{X} + \mathbf{Z})
 // normalize and apply residual connection
 6. end for
 7: return \mathbf{X}\mathbf{W}_{\text{head}} + \boldsymbol{b}_{\text{head}}
 // apply model head
```

## 2.2 RELATED WORK

In the simplest setting, one can employ magnitude-based sparsification, which involves setting the smallest weights in the model to zero (Han et al., 2016; Zhu & Gupta, 2017; Gale et al., 2019). Although magnitude sparsification is scalable, its application to LLMs gives too strong a degradation in performance (Frantar & Alistarh, 2023). Optimal Brain Surgeon (OBS) (Hassibi et al., 1993; LeCun et al., 1989), a more sophisticated method, systematically removes weights that have the least impact on the loss function. The method compensates for the error introduced by weight removal by updating the un-pruned weights using the inverse of the Hessian matrix. Unfortunately, OBS is impractical for models with a few million parameters due to the need to calculate and store the inverse of the Hessian matrix. To address the computational limitation posed by OBS, recent research has explored two approaches: approximating the inverse of the Hessian matrix such as WoodFisher (Singh & Alistarh, 2020) or applying it separately to each layer such as in Optimal Brain Compression (OBC, Frantar & Alistarh, 2022), known as layer-wise pruning. While these techniques have proven effective for mediumsized networks, they are not practical for large language models, where individual layer weight matrices typically contain more than 10^8 param-

GPTQ (Frantar et al., 2022) has solved this issue by quantizing (representing the parameter using lower precision) the weight matrix of LLMs using a column-by-column scheme and updating all not-yet-quantized weights in the next columns.

![RP28_Ashkboos_2024 fig01](../figures/RP28_Ashkboos_2024_fig01.jpg)
*Figure 2: A single layer in a transformer network. The signals (inputs) arising from the previous blocks of the networks arrive at the bottom of the figure, before being passed through attention, LayerNorm, and FFN. The attention and FFN blocks both have input and output linear operations (blue) which we denote in the text as \mathbf{W}_{in}, \mathbf{W}_{out}. The linear operations of LayerNorm \mathbf{M} and \mathbf{diag}(\alpha) are highlighted. This and subsequent figures do not show biases.*

SparseGPT (Frantar & Alistarh, 2023) applied the same idea for pruning and sparsifies the LLMs using unstructured and semi-structured pruning, and Sun et al. (2023) simplified the idea by using only the diagonal of the Hessian. Since achieving end-to-end speed improvements through unstructured pruning is a demanding task, they also attempted a similar technique to induce sparsity with semi-structured patterns like 2:4 and 4:8 (Mishra et al., 2021). However, implementing such structures does not maintain the accuracy of the model.

<!-- page 5 -->

Another approach to compression is low-rank approximation, where each weight matrix is replaced with the product of two matrices with a smaller inner dimension, usually followed by a fine-tuning step (Hu et al., 2021; Mahabadi et al., 2021; Noach & Goldberg, 2020; Tukan et al., 2020). To achieve compression, the inner dimension must be smaller than half of the original dimension. In contrast, our method replaces each weight matrix with a single smaller one, reducing the embedding dimension without the need for fine-tuning.

We propose to delete rows and columns of weight matrices, which is similar to pruning of filters and channels in the convnet literature. There, sparsity-inducing regularization is added to batch-norm factors (Liu et al., 2017) or network structures (Huang & Wang, 2018), and the network is trained or fine-tuned, resulting in the pruning of channels or parts of the network. Perhaps the most analogous methods to ours are ThiNet (Luo et al., 2017; He et al., 2017), which apply linear operations between layers (as will we), interleaved with more fine-tuning with regularization. In this literature, the model sizes are typically several orders of magnitude smaller than in LLMs, for example the VGG16 network has 138M parameters, comparable with the very smallest OPT model that we consider. The huge size of LLMs makes methods that involve extensive fine-tuning unappealing, especially when outer-loops are needed to select regularization parameters.

Recently, some works have been proposed that apply structured pruning to LLMs, followed by continued training (or fine-tuning) to recover the performance that is lost. For example LLM-pruner (Ma et al., 2023a) removes connected structures from an LLM before further training. Contemporarily with our work, LLM Surgeon (van der Ouderaa et al., 2023) interweaves recovery fine-tuning with pruning. We provide results for SliceGPT as a single-shot method and with post-slicing recovery fine-tuning.

# 3 SLICEGPT

Our SliceGPT method relies on a computational invariance that is inherent in the transformer architecture. By this, we mean that it is possible to apply an orthogonal transformation to the output of one component, so long as it is undone in the next. Our key insight is that the RMSNorm operation which is performed between blocks of the network does not affect the transformation: the operations commute. In this section, we first describe how the invariance occurs in RMSNorm-connected transformer networks, then we note how networks trained with LayerNorm connections can be converted to RMSNorm. Next, we describe our method to compute transformations at each layer using Principal Component Analysis (PCA), such that the signal between blocks is projected onto its principal components. Finally, we describe how deleting the minor principal components corresponds to slicing away rows or columns of the modified network.

## 3.1 COMPUTATIONAL INVARIANCE IN TRANSFORMER NETWORKS

Let \mathbf{Q} denote an orthogonal matrix: we have \mathbf{Q}^{\top}\mathbf{Q} = \mathbf{Q}\mathbf{Q}^{\top} = \mathbf{I}. Note that multiplying a vector \mathbf{x} by \mathbf{Q} does not change the norm of the vector, since \|\mathbf{Q}\mathbf{x}\| = \sqrt{\mathbf{x}^{\top}\mathbf{Q}^{\top}\mathbf{Q}\mathbf{x}} = \sqrt{\mathbf{x}^{\top}\mathbf{x}} = \|\mathbf{x}\|. In this work, the dimensions of \mathbf{Q} will always match the embedding dimension of the transformer D.

Suppose that X_{\ell} is the output of one block of the transformer, which is then processed by RMSNorm, and then inputted to the subsequent block as RMSNorm(X_{\ell}). If we insert linear layers with the orthogonal matrix Q before RMSNorm and Q^{\top} after RMSNorm, the network remains unchanged, since each row of the signal matrix is multiplied by Q, normalized and multiplied by Q^{\top}. We have

$$
RMSNorm(\mathbf{X}_{\ell}\mathbf{Q})\mathbf{Q}^{\top} = RMSNorm(\mathbf{X}_{\ell}). \tag{2}
$$

A proof of this relation appears in Appendix A.1. Now, since each attention or FFN block of the network has a linear operation on both the input and output, we can absorb the additional operations {\bf Q} into the linear layers of the blocks. Since the network contains residual connections, we must also apply {\bf Q} to the output of all previous layers (all the way back to the embedding) and to all subsequent layers (all the way up to the LM Head).

An *invariant* function is one for which a transformation to the input does not result in a change to the output. In our case, we can apply any orthogonal transformation \mathbf{Q} to the weights of the transformer without changing the result, so the *computation* can be performed in any transformed state. We refer to this as a *computational invariance*, and define it in the following theorem.

<!-- page 6 -->

**Theorem 1.** Let \mathbf{W}_{in}^{\ell} and \mathbf{W}_{out}^{\ell} be the weight matrices of the linear layers of the \ell-th block of an RMSNorm-connected transformer network, and b_{in}^{\ell}, b_{out}^{\ell} be the corresponding biases, if any, and let W_{embd} and W_{head} be the embedding and head matrices. Let Q be an orthogonal matrix of dimension D. Then the following network is equivalent to the original transformer network:

$$
\tilde{\mathbf{W}}_{embd} = \mathbf{W}_{embd} \mathbf{Q}, \qquad (3) \qquad \tilde{\mathbf{b}}_{out}^{\ell} = \mathbf{Q}^{\top} \mathbf{b}_{out}^{\ell}, \qquad (6) 
\tilde{\mathbf{W}}_{in}^{\ell} = \mathbf{Q}^{\top} \mathbf{W}_{in}^{\ell}, \qquad (4) \qquad \tilde{\mathbf{W}}_{head} = \mathbf{Q}^{\top} \mathbf{W}_{head}. \qquad (7)
$$

$$
\mathbf{\tilde{W}}_{im}^{\ell} = \mathbf{W}_{embd}^{\mathsf{T}} \mathbf{W}_{in}^{\ell}, (3) \mathbf{\tilde{b}}_{out}^{\ell} = \mathbf{Q}^{\mathsf{T}} \mathbf{W}_{head}^{\ell}, (6) \tilde{\mathbf{W}}_{head}^{\ell} = \mathbf{Q}^{\mathsf{T}} \mathbf{W}_{head}. (7)
$$

$$ \tilde{\mathbf{W}}_{out}^{\ell} = \mathbf{W}_{out}^{\ell} \mathbf{Q} \,, \tag{5} $$

The input and head biases are copied: \tilde{b}_{in}^{\ell} = b_{in}^{\ell}, \tilde{b}_{head} = b_{head}.

*Proof.* We can show that the transformed network computes the same results as the original by stepping through Algorithm 1. Suppose that on line 1, the original network has computed X, then the modified network has computed X = XQ, using Equation 3. Applying RMSNorm on line 2, we see that the operation of the two networks matches: by Equation 2 we have RMSNorm(\dot{\mathbf{X}}) = RMSNorm(\mathbf{XQ}) = RMSNorm(\mathbf{X})\mathbf{Q}. Applying the nonlinearity on line 4, we see that \tilde{\mathbf{X}}\tilde{\mathbf{W}}_{in}^{\ell} = XW_{in}^{\ell}, using Equation 4 and it follows that \tilde{Z} = ZQ. On line 5 the residual connection means we have (\ddot{X} + \ddot{Z}) = (X + Z)Q, and applying RMSNorm results in assignment of \ddot{X} = XQ. This follows through to the end of the loop. Finally, on line 7, the transformations are undone as XW_{head} = XW_{head} using Equation 7.

## 3.2 LAYERNORM TRANSFORMERS CAN BE CONVERTED TO RMSNORM

The computational invariance of the transformer network applies only to RMSNorm-connected networks. Before working on those with Layer-Norm, we convert the network to RMSNorm by absorbing the linear blocks of LayerNorm into the adjacent blocks. Figure 3 shows such a transformation on the transformer network (see Figure 2). In each block, we multiply the output matrix W_{out} by the mean-subtraction matrix M, which accounts for the mean subtraction that would happen in the subsequent LayerNorm. The input matrices W_{in} are pre-multiplied by the scales of the preceding LayerNorm blocks. The embedding matrix \mathbf{W}_{embd} must be mean-subtracted, and \mathbf{W}_{head} must be re-scaled by the last Layer-Norm scales. This is a straightforward change in the order of operations and does not affect the network output.

## 3.3 A TRANSFORMATION PER BLOCK

Now that every LayerNorm in the transformer has been converted to RMSNorm, we can select any Q to modify the model. Our initial plan was to collect signals from the model, construct an orthogonal matrix using those signals and to delete parts of the network. We quickly saw that the signals at different blocks of the network were not aligned, and that we would need to apply a different orthogonal matrix at each block, \mathbf{Q}_{\ell}.

Allowing the orthogonal matrix used in each block to differ can be shown to leave the model

![RP28_Ashkboos_2024 fig02](../figures/RP28_Ashkboos_2024_fig02.jpg)
*Figure 3: Converting a transformer network from LayerNorm to RMSNorm: the scale matrix \operatorname{diag}(\alpha) is absorbed into the subsequent matrix W<sub>in</sub>. Figure shows the block in combined colors. We use (\alpha) for brevity. The mean-subtraction matrix \mathbf{M} is applied to each matrix \mathbf{W}_{out}. Layernorm becomes RMSNorm, up to a constant \sqrt{D} (not shown). Here, the scaling (\alpha') comes from the previous block.*

<!-- page 7 -->

![RP28_Ashkboos_2024 fig03](../figures/RP28_Ashkboos_2024_fig03.jpg)
*Figure 4: With the network converted to RMSNorm (see Figure 3), we apply the computational-invariance idea. The input weight matrices \operatorname{diag}(\alpha)W_{in} are pre-multiplied by \mathbf{Q}^{\top}. The output matrices \mathbf{W}_{out}\mathbf{M} are post-multiplied by \mathbf{Q}. In the skip-connection, a new linear layer is added \mathbf{Q}_{\ell}^{\top}\mathbf{Q}_{\ell+1}. After these modifications, the matrices can be sliced (hatched areas).*

unchanged using the same proof as Theorem 1, with the exception of line 5 of Algorithm 1. Here we see that the residual connection and the output of the block must have the same rotation. To fix this, we modify the residual connection by applying the linear transformation \mathbf{Q}_{\ell-1}^{\top}\mathbf{Q}_{\ell} to the residual. Figure 4 shows how different rotations can be applied to different blocks with the additional linear operation in the residual connection. Unlike the modifications to the weight matrices, these additional operations cannot be pre-computed and add a small (D \times D) overhead to the model. Nonetheless, they are needed to allow slicing the model (Section 3.4) and we see real speedup overall (Section 4).

To compute the matrices \mathbf{Q}_{\ell}, we use PCA. We select a calibration dataset from the training set, run it through the model (after converting LayerNorm operations into RMSNorm), and extract the orthogonal matrix of the layer. We use the output of the transformed network to calculate the orthogonal matrices of the next layers. More precisely, if \mathbf{X}_{\ell,i} is the output of the \ell^{\text{th}} RMSNorm block for the i^{\text{th}} sequence in the calibration dataset, we compute

$$
\mathbf{C}_{\ell} = \sum_{i} \mathbf{X}_{\ell,i}^{\top} \mathbf{X}_{\ell,i} \tag{8}
$$

and set \mathbf{Q}_\ell to the be the eigenvectors of \mathbf{C}_\ell, sorted by decreasing eigenvalues.

## 3.4 SLICING

The goal of Principal Component Analysis is usually to take a data matrix X and compute a lower dimensional representation Z, and an approximate reconstruction \tilde{X}:

$$
\mathbf{Z} = \mathbf{XQD}, \qquad \tilde{\mathbf{X}} = \mathbf{ZD}^{\mathsf{T}} \mathbf{Q}^{\mathsf{T}}. (9)
$$

where \mathbf{Q} is the eigenvectors of \mathbf{X}^{\top}\mathbf{X}, and \mathbf{D} is a D \times D_{\text{small}} deletion matrix (containing D_{\text{small}} columns of the D \times D identity matrix), which removes some of the columns of the matrix to the left. The reconstruction is L_2 optimal, in the sense that \mathbf{Q}\mathbf{D} is a linear mapping that minimizes \|\mathbf{X} - \tilde{\mathbf{X}}\|^2.

When we apply PCA to the signal matrix \mathbf{X} between blocks, we never materialize the N \times D signal matrix, but we apply the deletion matrix \mathbf{D} to the operations preceding and succeeding the construction of that matrix, which have already been multiplied by \mathbf{Q} in the above. We delete rows of \mathbf{W}_{\text{in}} and columns of \mathbf{W}_{\text{out}} and \mathbf{W}_{\text{embd}}. We also delete both rows *and* columns of the matrix \mathbf{Q}_{\ell-1}^{\top}\mathbf{Q}_{\ell} that we have inserted into the residual connection (see Figure 4).

# 4 EXPERIMENTAL VALIDATION

**Setup** We use Hugging Face Transformers (Wolf et al., 2019) to implement our code with PyTorch (Paszke et al., 2019). The computation of **Q** is performed on a single H100 GPU with 80GB of memory, taking approximately 3.5 hours to complete for the LLAMA-2 70B model. During the PCA calculation, we use double precision for computing the eigenvectors of the covariance matrix. We find that using single precision for eigenvector calculations in PyTorch leads to a discrepancy in the final accuracy, as detailed in Appendix A.2.

<!-- page 8 -->

We experiment with two different calibration sets: the WikiText-2 training dataset [(Merity et al.,](#page-11-16) [2016)](#page-11-16) and the Alpaca training dataset [(Taori et al.,](#page-12-10) [2023)](#page-12-10). An ablation study on the calibration set size and sequence length is presented in Appendix [A.3.](#page-14-2)

Models, Tasks, and GPUs We evaluate all our experiments on OPT [(Zhang et al.,](#page-13-2) [2022)](#page-13-2), LLAMA-2 [(Touvron et al.,](#page-12-0) [2023)](#page-12-0) model families, and additionally evaluate Phi-2 (in our zero-shot task) experiments. We exclude OPT 175B, as it is outperformed by smaller LLAMA-2 models. Nonetheless, we anticipate that this larger model will yield improved results, as larger models typically offer more promising opportunities for compression (see Section [4.1)](#page-7-0). We evaluate our scheme on both language generation as well as popular zero-shot tasks. To demonstrate the comprehensive speedup achieved by SliceGPT we use: Quadro RTX6000 GPUs with 24GB of memory as a representative example of consumer-level GPUs; 40GB A100s and 80GB H100s to provide datacenter-level benchmarks.

Baseline Setup We initially planned to compare our results against a scheme that pruned columns (or rows) with the smallest norm but found that this baseline was very poor, with the WikiText-2 perplexity of the model soaring into the 1000s after pruning just a few columns. Instead, we compare SliceGPT against SparseGPT [(Frantar & Alistarh,](#page-10-1) [2023)](#page-10-1) employing a 2:4 sparsity ratio, as this is the only sparsity scheme which achieves speedup [(Mishra et al.,](#page-11-8) [2021)](#page-11-8).

## 4.1 RESULTS

Generation Task We begin by showcasing our findings using the WikiText-2 dataset. In this context, we evaluate the performance of both the OPT and LLAMA-2 model families across different sizes when using this dataset for slicing. Table [1](#page-7-1) shows the perplexity obtained by various slicing levels. SliceGPT exhibits superior performance when applied to OPT models compared to LLAMA-2 models which matches our intuition from the spectrum analysis of those models (see Appendix [A.4](#page-15-0) for our discussion). The performance of SliceGPT improves as the model size increases. Comparing SliceGPT with SparseGPT, we see that that SparseGPT 2:4 performs worse than SliceGPT with 25% slicing in all LLAMA-2 models. For OPT, we see that 30% sliced models beat 2:4 sparsity for all model sizes except 2.7B.

| Table 1: OPT and LLAMA-2 perplexity results on WikiText2. The calibration set size and sequence |  |  |
| --- | --- | --- |
| length are 1024 and 2048, respectively. |  |  |

| Method | OPT | LLAMA-2 |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 125M | 1.3B | 2.7B | 6.7B | 13B | 30B | 66B | 7B | 13B | 70B |  |
| Dense | 27.64 | 14.61 | 12.46 | 10.85 | 10.12 | 9.56 | 9.33 | 5.47 | 4.88 | 3.32 |
| SparseGPT 2:4 | 45.07 | 29.61 | 14.90 | 13.00 | 11.80 | 10.53 | 10.22 | 8.69 | 7.07 | 4.98 |
| SliceGPT (10%) | 29.34 | 15.10 | 12.75 | 10.92 | 10.27 | 9.65 | 9.43 | 5.89 | 5.21 | 3.69 |
| SliceGPT (20%) | 34.26 | 16.43 | 13.73 | 11.48 | 10.66 | 9.87 | 9.57 | 6.64 | 5.81 | 4.25 |
| SliceGPT (25%) | 37.74 | 17.46 | 14.56 | 11.90 | 10.94 | 10.04 | 9.68 | 7.24 | 6.30 | 4.60 |
| SliceGPT (30%) | 43.98 | 19.09 | 15.83 | 12.51 | 11.33 | 10.27 | 9.85 | 8.12 | 6.99 | 5.05 |

Zero-shot Tasks We assess SliceGPT across five well-known zero-shot tasks: PIQA [(Bisk et al.,](#page-10-3) [2020)](#page-10-3); WinoGrande [(Sakaguchi et al.,](#page-12-11) [2021)](#page-12-11); HellaSwag [(Zellers et al.,](#page-12-12) [2019)](#page-12-12); ARC-e and ARCc [(Clark et al.,](#page-10-4) [2018)](#page-10-4). Following similar work [(Frantar & Alistarh,](#page-10-1) [2023;](#page-10-1) [Dettmers et al.,](#page-10-5) [2022;](#page-10-5) [Frantar et al.,](#page-11-7) [2022;](#page-11-7) [Dettmers et al.,](#page-10-6) [2023)](#page-10-6), we use the LM Evaluation Harness [(Gao et al.,](#page-11-17) [2021)](#page-11-17) with default parameters in our evaluations.

Figure [5](#page-8-0) shows the average scores achieved by the sliced models across these tasks. The top row of the plot shows the mean accuracy when WikiText-2 is used for calibration, and the bottom row shows the accuracy when Alpaca is used for calibration. We observe a similar pattern to the generation task in the results: the OPT models are more amenable to compression than the LLAMA-2 models, and the reduction in accuracy is less pronounced in the larger models. Here we also include the Phi-2 model: we see that sliced versions of the Phi-2 model are comparable with sliced versions of the LLAMA-2 7B model. The largest OPT and LLAMA-2 models can be compressed very effectively, with just a few percentage points loss when removing 30% of the 66B OPT model.

We additionally experiment here with recovery fine-tuning (RFT). We apply a small amount of RFT to sliced LLAMA-2 and Phi-2 models using LoRA [(Hu et al.,](#page-11-9) [2021)](#page-11-9), following the idea from [Ma](#page-11-15)

<!-- page 9 -->

et al. (2023a). For models sliced with WikiText-2 we use approximately 1k sequences, for those sliced with the Alpaca dataset we use 5k. For all RFT we use lora\_r = 32, lora\_alpha = 10 and sequence length 1024, and use defaults for all other hyperparameters in the Hugging Face PEFT package (Mangrulkar et al., 2022).

Figure 6 shows the results. We see a marked difference between RFT on WikiText-2 and Alpaca datasets, with the Alpaca dataset giving much higher performing models. We attribute this difference to the similarity between Alpaca and the benchmark tasks. For the largest Llama-2 70B model sliced at 30%, with RFT on Alpaca we are able to achieve an average accuracy of 74.3%, compared to 76.6% on the dense model. The sliced model has approximately 51.6B parameters and considerably improved throughput as we demonstrate later.

We see that Phi-2 is not able to recover the drop in accuracy from slicing using only the WikiText-2 dataset, but using Alpaca we are able to recover several percentage points. The average accuracy of Phi-2 with 25% slicing and RFT is 65.2%, compared to 72.2% with the dense model. The sliced model has approximately 2.2B parameters and retains 90.3% of the accuracy of the 2.8B model. This shows that even small LMs can benefit from post-training pruning. Tables of accuracies across each task are provided in Appendix A.5.

![RP28_Ashkboos_2024 fig04](../figures/RP28_Ashkboos_2024_fig04.jpg)
*Figure 5: Mean zero-shot accuracy on OPT, LLAMA-2 and Phi-2 across multiple tasks after slicing with the WikiText-2 (top) and Alpaca (bottom) datasets for calibration.*

![RP28_Ashkboos_2024 fig05](../figures/RP28_Ashkboos_2024_fig05.jpg)
*Figure 6: Mean zero-shot accuracy on LLAMA-2 and Phi-2 across multiple tasks after slicing and recovery fine-tuning (RFT). Left: WikiText-2 used for calibration and RFT. Right: Alpaca used for calibration and RFT. Despite an extensive search, we were not able to find RFT parameters that enabled improved performance in the OPT models.*

**Benchmarking Throughput** Unlike conventional sparsity methods, which introduce sparsity in \mathbf{W}_{in} and \mathbf{W}_{out}, SliceGPT also introduces (structured) sparsity in \mathbf{X}: entire columns of \mathbf{X} are sliced off, reducing the embedding dimension. This enhances both the computational complexity (in flops) and data movement within our compressed model.

<!-- page 10 -->

The token throughput of models sliced at 25% and 50% are compared to the dense model on 80GB H100 GPUs. We set the sequence length to 128 and find the maximum throughput by doubling the batch size until the GPUs run out of memory or the throughput drops off. The 25% sliced models achieve up to 1.55× throughput improvement over the dense model. At 50% slicing the largest models require only one GPU instead of two, with large increases in throughput: 3.13× and 1.87×. This means that for a fixed number of GPUs, these models achieve 6.26× and 3.75× throughput of a dense model. We note that the WikiText2 perplexity of SliceGPT at 50% is worse than SparseGPT 2:4, but the throughput is much higher than could be achieved with a sparse method that does not slice X. For models of size 13B, the performance increase from batch-size increasing is less pronounced because the models take up little of the GPU memory. On consumer grade GPUs (with less memory) the throughput for these smaller models is likely to be improved. For full details see Appendix [A.6.](#page-20-0)

Inference Time Next we study the end-to-end runtime of a model compressed with SliceGPT. Table [2](#page-9-0) compares the time of generating a single token in OPT 66B and LLAMA-2 70B models on Quadro RTX6000 and A100 GPUs. We observe a speedup of 16-17% on RTX6000 GPUs when employing 25% slicing, and 11-13% on A100s. We reduce the number of GPUs used in both cases, providing energy and cost savings relative to deployment of the dense model. For LLAMA-2 70B, the compute required using RTX6000 GPUs is reduced to 64%, from 1764 GPUms to 1075 GPUms[4](#page-9-1) . We attribute this improvement to our approach of substituting weight matrices with smaller ones and using dense kernels in our compressed models, which is infeasible with other pruning schemes.

End-to-end performance gains are not feasible with our baseline SparseGPT 2:4 at the time of writing. Instead, we compare SliceGPT with SparseGPT 2:4 by comparing the relative timing of each operation involved in a transformer layer. We find that SliceGPT (25%) is competitive with SparseGPT (2:4) in terms of speedup and perplexity for large models. For further details see Appendix [A.7.](#page-20-1)

Compute cost All LLAMA-2 , OPT and Phi-2 models can be sliced on a single GPU in 1 to 3 hours. With recovery fine-tuning we compress all LMs in 1 to 5 hours total, as shown in Table [3.](#page-9-2)

<!-- page 11 -->

# 5 CONCLUSION AND FUTURE WORK

We've introduced SliceGPT which allows for structured pruning for large language models. We reduce the cost of inference of LLAMA-2 70B on 40GB A100 GPUs to 66% of that of the dense model without any additional code optimization, requiring fewer GPUs (from 4 to 3) while maintaining better held-out perplexity than SparseGPT 2:4. On 24GB RTX6000 GPUs, the cost of inference is reduced to 64%, requiring 2 fewer GPUs (from 7 to 5). On zero-shot downstream tasks, slicing OPT 66B, LLAMA-2 70B and Phi-2 at 25% maintains 99%, 96% and 87% of the dense performance. With recovery fine-tuning 25%-sliced LLAMA-2 70B and Phi-2 increase to 99% and 90% respectively.

Opportunities remain to build on our method. Smaller but dense LMs perform better than LMs with 13B parameters or less pruned to similar sizes, though we do not expect this to remain the case for long. Our pruned models have more parameters than those pruned with SparseGPT but our method allows for larger batch sizes to be loaded into GPU memory, and has no overhead for sparsity structure: perhaps a combined method could obtain the best of both. Other methods of computing Q could improve the results. To further decrease the inference time and GPU count, complementary methods including quantization [(Xiao et al.,](#page-12-13) [2023;](#page-12-13) [Dettmers et al.,](#page-10-5) [2022;](#page-10-5) [Ashkboos et al.,](#page-10-7) [2023;](#page-10-7) [Dettmers et al.,](#page-10-6) [2023;](#page-10-6) [Frantar et al.,](#page-11-7) [2022)](#page-11-7), and structural pruning (e.g. [Ma et al.,](#page-11-19) [2023b)](#page-11-19) could be used.

We hope that our observation of computational invariance can help future research in improving the efficiency of deep learning models, and perhaps inspire new theoretical insights.

## ACKNOWLEDGEMENTS

We thank Dmitry Kats, Pashmina Cameron, Pavel Myshkov and Liana Mikaelyan for their invaluable contributions to the source code. We additionally thank Pashmina Cameron for her helpful feedback when reviewing early versions of the paper.

## REFERENCES

- Saleh Ashkboos, Ilia Markov, Elias Frantar, Tingxuan Zhong, Xincheng Wang, Jie Ren, Torsten Hoefler, and Dan Alistarh. Towards end-to-end 4-bit inference on generative large language models. *arXiv preprint arXiv:2310.09259*, 2023.
- Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. *arXiv preprint* *arXiv:1607.06450*, 2016.
- Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language. In *Thirty-Fourth AAAI Conference on Artificial* *Intelligence*, 2020.
- Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. *ArXiv*, abs/1803.05457, 2018. URL [https://api.semanticscholar.org/CorpusID:](https://api.semanticscholar.org/CorpusID:3922816) [3922816](https://api.semanticscholar.org/CorpusID:3922816).
- Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. LLM. int8 (): 8-bit matrix multiplication for transformers at scale. *arXiv preprint arXiv:2208.07339*, 2022.
- Tim Dettmers, Ruslan Svirschevski, Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashkboos, Alexander Borzunov, Torsten Hoefler, and Dan Alistarh. Spqr: A sparse-quantized representation for near-lossless LLM weight compression. *arXiv preprint arXiv:2306.03078*, 2023.
- Elias Frantar and Dan Alistarh. Optimal brain compression: A framework for accurate post-training quantization and pruning. *Advances in Neural Information Processing Systems*, 35:4475–4488, 2022.
- Elias Frantar and Dan Alistarh. SparseGPT: Massive language models can be accurately pruned in one-shot. 2023.

<!-- page 12 -->

- Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate post-training quantization for generative pre-trained transformers. *arXiv preprint arXiv:2210.17323*, 2022.
- Trevor Gale, Erich Elsen, and Sara Hooker. The state of sparsity in deep neural networks, 2019.
- Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, et al. A framework for few-shot language model evaluation. *Version v0. 0.1. Sept*, 2021.
- Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W. Mahoney, and Kurt Keutzer. A survey of quantization methods for efficient neural network inference. *CoRR*, abs/2103.13630, 2021. URL [https://arxiv.org/abs/2103.13630](https://arxiv.org/abs/2103.13630).
- Manish Gupta and Puneet Agrawal. Compression of deep learning models for text: A survey, 2021.
- Song Han, Huizi Mao, and William J. Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding, 2016.
- Babak Hassibi, David G Stork, and Gregory J Wolff. Optimal brain surgeon and general network pruning. In *IEEE international conference on neural networks*, pp. 293–299. IEEE, 1993.
- Yihui He, Xiangyu Zhang, and Jian Sun. Channel pruning for accelerating very deep neural networks. In *Proceedings of the IEEE international conference on computer vision*, pp. 1389–1397, 2017.
- Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, and Alexandra Peste. Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks. *CoRR*, abs/2102.00554, 2021. URL [https://arxiv.org/abs/2102.00554](https://arxiv.org/abs/2102.00554).
- Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021.
- Zehao Huang and Naiyan Wang. Data-driven sparse structure selection for deep neural networks. In *Proceedings of the European conference on computer vision (ECCV)*, pp. 304–320, 2018.
- Yann LeCun, John Denker, and Sara Solla. Optimal brain damage. *Advances in neural information* *processing systems*, 2, 1989.
- Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, and Changshui Zhang. Learning efficient convolutional networks through network slimming. In *Proceedings of the IEEE* *international conference on computer vision*, pp. 2736–2744, 2017.
- Jian-Hao Luo, Jianxin Wu, and Weiyao Lin. Thinet: A filter level pruning method for deep neural network compression. In *Proceedings of the IEEE international conference on computer vision*, pp. 5058–5066, 2017.
- Xinyin Ma, Gongfan Fang, and Xinchao Wang. Llm-pruner: On the structural pruning of large language models. *arXiv preprint arXiv:2305.11627*, 2023a. URL [https://arxiv.org/pdf/](https://arxiv.org/pdf/2305.11627.pdf) [2305.11627.pdf](https://arxiv.org/pdf/2305.11627.pdf).
- Xinyin Ma, Gongfan Fang, and Xinchao Wang. LLM-pruner: On the structural pruning of large language models, 2023b.
- Rabeeh Karimi Mahabadi, James Henderson, and Sebastian Ruder. Compacter: Efficient low-rank hypercomplex adapter layers, 2021.
- Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, and Benjamin Bossan. Peft: State-of-the-art parameter-efficient fine-tuning methods. [https://github.](https://github.com/huggingface/peft) [com/huggingface/peft](https://github.com/huggingface/peft), 2022.
- Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. *arXiv preprint arXiv:1609.07843*, 2016.
- Asit Mishra, Jorge Albericio Latorre, Jeff Pool, Darko Stosic, Dusan Stosic, Ganesh Venkatesh, Chong Yu, and Paulius Micikevicius. Accelerating sparse deep neural networks. *arXiv preprint* *arXiv:2104.08378*, 2021.

<!-- page 13 -->

- Matan Ben Noach and Yoav Goldberg. Compressing pre-trained language models by matrix decomposition. In *Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for* *Computational Linguistics and the 10th International Joint Conference on Natural Language Pro**cessing*, pp. 884–889, Suzhou, China, December 2020. Association for Computational Linguistics. URL [https://aclanthology.org/2020.aacl-main.88](https://aclanthology.org/2020.aacl-main.88).
- Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. PyTorch: An imperative style, high-performance deep learning library. *Advances in neural information processing systems*, 32, 2019.
- Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.
- Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. *Communications of the ACM*, 64(9):99–106, 2021.
- Sidak Pal Singh and Dan Alistarh. Woodfisher: Efficient second-order approximation for neural network compression. *Advances in Neural Information Processing Systems*, 33:18098–18109, 2020.
- Mingjie Sun, Zhuang Liu, Anna Bair, and J Zico Kolter. A simple and effective pruning approach for large language models. *arXiv preprint arXiv:2306.11695*, 2023.
- Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. [https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca), 2023.
- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023.
- Murad Tukan, Alaa Maalouf, Matan Weksler, and Dan Feldman. Compressed deep networks: Goodbye SVD, hello robust low-rank approximation. *arXiv preprint arXiv:2009.05647*, 2020.
- Tycho FA van der Ouderaa, Markus Nagel, Mart van Baalen, Yuki M Asano, and Tijmen Blankevoort. The llm surgeon. *arXiv preprint arXiv:2312.17244*, 2023.
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information processing* *systems*, 30, 2017.
- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. Huggingface's transformers: State-of-the-art natural language processing. *arXiv preprint arXiv:1910.03771*, 2019.
- Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models. In *International* *Conference on Machine Learning*, pp. 38087–38099. PMLR, 2023.
- Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? *arXiv preprint arXiv:1905.07830*, 2019.

<!-- page 14 -->

- Biao Zhang and Rico Sennrich. Root mean square layer normalization. *Advances in Neural* *Information Processing Systems*, 32, 2019.
- Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. *arXiv preprint arXiv:2205.01068*, 2022.
- Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. *arXiv* *preprint arXiv:2303.18223*, 2023.
- Michael Zhu and Suyog Gupta. To prune, or not to prune: exploring the efficacy of pruning for model compression, 2017.
- Xunyu Zhu, Jian Li, Yong Liu, Can Ma, and Weiping Wang. A survey on model compression for large language models. *arXiv preprint arXiv:2308.07633*, 2023.

<!-- page 15 -->

## A APPENDIX

## A.1 PROOF OF EQUATION 2

An orthogonal matrix \mathbf{Q} is a square matrix that satisfies the relation \mathbf{Q}^{\top}\mathbf{Q} = \mathbf{Q}\mathbf{Q}^{\top} = \mathbf{I}. The norm of a vector is the square-root of the sum of squares of the elements: \|\mathbf{x}\| = \sqrt{\sum_i x_i^2} = \sqrt{\mathbf{x}^{\top}\mathbf{x}}. Multiplying a vector by \mathbf{Q} does not change the norm since \|\mathbf{Q}\mathbf{x}\| = \sqrt{\mathbf{x}^{\top}\mathbf{Q}^{\top}\mathbf{Q}\mathbf{x}} = \|\mathbf{x}\|.

The RMSNorm operation divides each row of the input matrix \mathbf{X} by its norm. By the basic rules of linear algebra, if \mathbf{x} is a row of \mathbf{X}, then \mathbf{Q}^{\top}\mathbf{x} is the corresponding row of \mathbf{X}\mathbf{Q}. Applying RMSNorm to \mathbf{X}\mathbf{Q}, said row will now be equal to \frac{1}{\|\mathbf{x}\|}\mathbf{Q}^{\top}\mathbf{x}. After RMSnorm, we can multiply by \mathbf{Q}^{\top}, our row is now equal to \frac{1}{\|\mathbf{x}\|}\mathbf{Q}\mathbf{Q}^{\top}\mathbf{x} = \frac{1}{\|\mathbf{x}\|}\mathbf{x}. Thus we have the relation

$$ RMSNorm(\mathbf{XQ})\mathbf{Q}^{\top} = RMSNorm(\mathbf{X}). \tag{10} $$

## A.2 SINGLE PRECISION EIGENVALUE CALCULATION

As previously noted in Section 4, we employ double precision when performing the PCA algorithm. This choice is made in order to mitigate potential numerical errors that may arise during the computation of the orthogonal matrix in SliceGPT. Nevertheless, it is intriguing to investigate the impact of employing lower precision for PCA calculations on the ultimate accuracy.

Table 4 shows the perplexity of all our models when we apply FP32 PCA in our scheme. It shows that the accuracy of larger models could be affected by numerical errors during the PCA calculation phase. It should be noted that we use PyTorch (torch.linalg) for calculating the eigenvectors and eigenvalues.

## A.3 SENSITIVITY TO THE CALIBRATION SET SIZE AND SEQUENCE LENGTH

We present an ablation study to examine the role of the WikiText-2 calibration set. We focus on the generation task with 25% sparsity using OPT 6.7B and LLAMA-2 7B models.

![RP28_Ashkboos_2024 fig06](../figures/RP28_Ashkboos_2024_fig06.jpg)
*Figure 7: The effect of the calibration set size and sequence length on perplexity of WikiText2.*

<!-- page 16 -->

Figure 7 (left) shows the result of varying the size of the calibration set on the perplexity. It shows that sample sizes of at least 128 provide sensible choices for our calibration set.

Next we explore the effect of using different sequence lengths N in the calibration set. Given a fixed number of B samples, the PCA input matrix is computed using NB embedding vectors, and understanding the tradeoff between having a larger B or a larger N is interesting. Figure 7 (right) shows the results of varying the sequence length in the calibration set from 128 to 4096: we conclude that having a larger sequence length can result in better perplexity.

Using these insights, we use a calibration set size of 1024 and sequence length 2048 in our main experiments (Table 1). In Table 5 below we evaluate the perplexity of OPT and LLAMA-2 models on WikiText-2 with a smaller calibration set size, which confirms the trend that decreasing this degrades the perplexity across all models and sizes.

## A.4 SPECTRUM ANALYSIS OF LLAMA-2 AND OPT MODELS

The figure below shows the eigenvalue distribution for the OPT 6.7B and LLAMA-2 7B models. Although both models have a comparable parameter count, the LLAMA-2 model has a more tightly compressed distribution in its embeddings spectrum. This observation shows that there are no dominant principal components with significantly more information, making the pruning of these components a more challenging task.

![RP28_Ashkboos_2024 fig07](../figures/RP28_Ashkboos_2024_fig07.jpg)
*Figure 8: Normalized (by maximum) spectrum of the MLP inputs (log scale) using 64 samples. Except for the first layer in the LLAMA-2 model, the eigenvalue distributions for both models show faster decay in early layers compared to later ones. This suggests that a greater amount of slicing could be applied after the orthogonal transformation in these early layers.*

We can use the above insights to slice different layers by different amounts. Instead of specifying the slicing level upfront, we set the fraction of the total variance to discard during each PCA calculation, which sets the number of rows and columns to slice off from each matrix. For each model, we run three experiments with varying target variances to obtain a total reduction on the network close to 25%.

<!-- page 17 -->

The results are shown in Table [6](#page-16-0) below. Varying the slicing level by layer improves the WikiText-2 perplexity in OPT models, but has the opposite effect in LLAMA-2 models.

<!-- page 18 -->

## A.5 DETAILED ZERO-SHOT RESULTS

In this section, we provide the detailed results of the zero-shot tasks we presented in the paper.

<!-- page 19 -->

<!-- page 20 -->

<!-- page 21 -->

## A.6 BENCHMARKING THROUGHPUT EXPERIMENT

## A.7 BENCHMARKING INFERENCE TIME OF SLICEGPT AGAINST SPARSEGPT

We use the CuSparseLT 0.5 library to run sparse matrix multiplications on an 80 GB A100 GPU, replicating the size of the matrix-matrix multiplications in three different-sized LLAMA-2 models. We used PyTorch to run similar matrix multiplications for the dense equivalent, and for SliceGPT (which is also straightforward dense matmul, but smaller). We chose a sequence length of 2048, and took the matrix sizes from the HuggingFace config files. We took the median runtime over 10^3^ attempts.

Each LLAMA-2 layer requires a gated FFN with one up projection, one down projection, and a gated projection. In attention, the architecture of the model means that the query matrix multiplication is a different size to the key and value matrix multiplications. The following table shows the time taken in ms to run each matrix multiplication in the model, plus a "total" time and a relative speedup.

We also benchmarked the OPT architecture in the same way. In this case, the matrix multiplications associated with Key, Value, Query and Out are all the same size, and there are just two matrix multiplications in the MLP section (FC1 and FC2).

<!-- page 22 -->
