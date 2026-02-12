from __future__ import annotations

import os

import regex as re
from collections import Counter
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn import Module

from tests.toolFun.Tokenizer import get_stats, merge_ids,BPETokenizer
from tests.toolFun.transformer import Linear, Embedding, RMSNorm, SwiGLU, MultiHeadAttention, Rope, DotAttention
from tests.toolFun.Optimizer import Adamw,Cosine,GradientClip


def run_linear(
    d_in: int,#输入维度的大小
    d_out: int,#输出维度的大小
    weights: Float[Tensor, " d_out d_in"],#要使用的线性权重
    in_features: Float[Tensor, " ... d_in"],#要应用该函数的输出张量
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    # 1. 检查维度是否匹配（可选，但推荐）
    assert weights.shape == (d_out, d_in)
    assert in_features.shape[-1] == d_in
    liner = Linear(d_in,d_out)
    # 这里的 key "W" 必须对应你在 __init__ 中 self.W = ... 的名字
    state_dict = {"W": weights}
    # 3. 加载权重
    # strict=True (默认) 会检查名字是否完全匹配，多一个少一个都会报错
    liner.load_state_dict(state_dict)

    return liner.forward(in_features)



def run_embedding(
    vocab_size: int,#词汇表中嵌入向量的数量
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],#要从中获取的嵌入向量
    token_ids: Int[Tensor, " ..."],#要从嵌入层获取的词元 ID 集合
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    enbeding = Embedding(vocab_size, d_model)
    state = {"embed_table":weights}
    enbeding.load_state_dict(state)
    return enbeding.forward(token_ids)


def run_swiglu(
    d_model: int,#前馈输入和输出的维度
        d_ff: int,#SwiGLU 网络内部上投影的维度。
    w1_weight: Float[Tensor, " d_ff d_model"],#存储的 W1 权重,门控投影 。负责把输入映射到高维，并经过激活函数。它决定了“让多少信息通过”。
    w2_weight: Float[Tensor, " d_model d_ff"],#上投影 。负责把输入映射到高维，但不进行激活（或者是线性的）。它包含了主要的信息内容。
    w3_weight: Float[Tensor, " d_ff d_model"],#下投影.负责把处理好的高维特征，重新压缩回原来的维度 d_model
    in_features: Float[Tensor, " ... d_model"],#前馈层的输入嵌入。
) -> Float[Tensor, " ... d_model"]:

    swiglu = SwiGLU(d_model,d_ff)
    state = {"W1":w1_weight,"W2":w2_weight,"W3":w3_weight}
    swiglu.load_state_dict(state)
    return swiglu.forward(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:

    mult_atten = MultiHeadAttention(d_model,num_heads)
    # PyTorch Linear 权重为 (out, in)，计算为 x @ weight.T；本实现用 x @ w，故需加载 weight.T
    state = {
        "wq": q_proj_weight.T,
        "wk": k_proj_weight.T,
        "wv": v_proj_weight.T,
        "out_proj": o_proj_weight.T,
    }
    mult_atten.load_state_dict(state)
    return mult_atten.forward(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,#每个 token 在序列中的“绝对位置索引
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    B,T,D = in_features.shape
    assert d_model % num_heads == 0,"d_model must be divisible by num_heads"
    head_dim = d_model // num_heads
    # 1. 初始化 RoPE 模块
    # 注意：这里的 d_k 是 head_dim
    rope_module = Rope(theta=theta, d_k=head_dim, max_seq_len=max_seq_len)
    #投影qkv
    q = in_features @ q_proj_weight.T
    k = in_features @ k_proj_weight.T
    v = in_features @ v_proj_weight.T
    #将d_model拆分为head_dim * heads并转置
    q = q.view(B,T,num_heads,head_dim).transpose(1, 2)
    k = k.view(B,T,num_heads,head_dim).transpose(1, 2)
    v = v.view(B,T,num_heads,head_dim).transpose(1, 2)
    #关键插入点：旋转 (Apply RoPE)
    # 6. 🔴 应用 RoPE
    # 注意：RoPE 类通常需要处理 Heads 维度广播，确保你的 RoPE forward 支持 [Batch, Heads, Seq, Dim] 输入
    q_rope = rope_module.forward(q,token_positions)
    k_rope = rope_module.forward(k,token_positions)
    # 7. 计算注意力 (Scaled Dot-Product)，decoder-only 需因果掩码
    causal_mask = torch.tril(torch.ones(T, T, device=in_features.device, dtype=q_rope.dtype)).view(1, 1, T, T)
    attn_score = DotAttention().forward(q_rope, k_rope, v, mask=causal_mask)
    out = attn_score.transpose(1,2).contiguous()
    out = out.view(B,T,d_model)
    return out @ o_proj_weight.T

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = Rope(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
    return rope.forward(in_query_or_key,token_positions)


def run_transformer_block(
    d_model: int,#Transformer 模块输入的维度。
    num_heads: int,#多头注意力机制中使用的头数
    d_ff: int,#前馈内层的维度
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],

    in_features: Float[Tensor, " batch sequence_length d_model"],

) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    给定预归一化 Transformer 模块的权重和输入特征，
    返回在输入特征上运行 Transformer 模块的输出。
    此函数应使用 RoPE。
    根据您的实现，您可能只需将相关参数
    传递给您的 TransformerBlock 构造函数，或者您可能需要初始化您自己的 RoPE类并传递该类。
    参数：
    d_model (int)：Transformer 模块输入的维度。
    num_heads (int)：多头注意力机制中使用的头数。`d_model` 必须
    能被 `num_heads` 整除。
    d_ff (int)：前馈内层的维度。
    max_seq_len (int)：如果您的实现支持预缓存，则为预缓存的最大序列长度。
    theta (float)：RoPE 参数。
    weights (dict[str, Tensor])：
    参考实现的状态字典。

    此字典的键为：
    - `attn.q_proj.weight`
    所有 `num_heads` 个注意力头的查询投影。形状为 (d_model, d_model)。行按形状为 (num_heads, d_k) 的矩阵排序----就是reshape，
    因此 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。

    - `attn.k_proj.weight`
    所有 `num_heads` 个注意力头的键投影。形状为 (d_model, d_model)。行按形状为 (num_heads, d_k) 的矩阵排序，
    因此 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。

    - `attn.v_proj.weight`
    所有注意力头的值投影`num_heads` 个注意力头。形状为 (d_model, d_model)。行按形状为 (num_heads, d_v) 的矩阵排序，
    因此 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。

    - `attn.output_proj.weight`
    多头自注意力输出投影的权重。形状为 (d_model, d_model)。

    - `ln1.weight`
    第一个 RMSNorm 的仿射变换权重。
    应用于 Transformer 模块。
    形状为 (d_model,)。

    - `ffn.w1.weight`
    前馈神经网络 (FFN) 中第一个线性变换的权重。形状为 (d_model, d_ff)。

    - `ffn.w2.weight`
    前馈神经网络 (FFN) 中第二个线性变换的权重。形状为 (d_ff, d_model)。

    - `ffn.w3.weight`
    前馈神经网络 (FFN) 中第三个线性变换的权重。形状为 (d_model, d_ff)。

    - `ln2.weight`
    第二个 RMSNorm 的仿射变换权重应用于 Transformer 模块。形状为 (d_model,)。

    in_features (Float[Tensor, "batch sequence_length d_model"]):
    用于运行您的实现的张量。
    返回值：
    Float[Tensor, "batch sequence_length d_model"] 张量，包含使用 RoPE 对输入特征运行 Transformer 模块的输出。

    """
    # 获取当前输入的实际维度
    batch_size, seq_len, _ = in_features.shape
    #残差链接预留
    residual = in_features
    # 1. 修正位置索引生成
    # 必须使用 seq_len 而不是 max_seq_len
    # 必须指定 device=in_features.device
    token_pos = torch.arange(seq_len, device=in_features.device)
    # 扩展为 [Batch, Seq] 以匹配 Attention 的输入要求
    token_pos = token_pos.unsqueeze(0).expand(batch_size, seq_len)
    #对输入进行RMSNorm并传入带RoPE 的多头注意力
    normed_x1 = run_rmsnorm(d_model, eps=1e-5,weights=weights["ln1.weight"],in_features=in_features)
    attn_out = run_multihead_self_attention_with_rope(d_model,num_heads,max_seq_len,theta,
                                               weights["attn.q_proj.weight"],
                                               weights["attn.k_proj.weight"],
                                               weights["attn.v_proj.weight"],
                                               weights["attn.output_proj.weight"],
                                               normed_x1,token_pos)
    #残差连接
    x = residual + attn_out
    residual = x
    #第二次rmsnorm
    normed_x2 = run_rmsnorm(d_model, eps=1e-5,weights=weights["ln2.weight"],in_features=x)
    swiglu_x = run_swiglu(d_model,d_ff,weights["ffn.w1.weight"],weights["ffn.w2.weight"],weights["ffn.w3.weight"],normed_x2)
    out_t = residual + swiglu_x
    return out_t


def run_transformer_lm(
    vocab_size: int,#待预测的输出词汇表中唯一词项的数量。
    context_length: int,#一次处理的最大词元数
    d_model: int,
    num_layers: int,#要使用的 Transformer 层数。
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """给定 Transformer 语言模型的权重和输入索引，
返回对输入索引进行前向传播后的输出。
此函数应使用 RoPE。
参数：
vocab_size (int)：待预测的输出词汇表中唯一词项的数量。
context_length (int)：一次处理的最大词元数。
d_model (int)：模型嵌入和子层输出的维度。
num_layers (int)：要使用的 Transformer 层数。
num_heads (int)：多头注意力机制中使用的注意力头数量。`d_model` 必须能被 `num_heads` 整除。
d_ff (int)：前馈内层的维度（参见 3.3 节）。
rope_theta (float)：RoPE 的 Theta 参数。

weights (dict[str, Tensor])：
参考实现的状态字典。 {num_layers} 指的是一个介于 0 和 num_layers - 1 之间的整数（层索引）。
此字典的键如下：
- `token_embeddings.weight`
词元嵌入矩阵。形状为 (vocab_size, d_model)。

- `layers.{num_layers}.attn.q_proj.weight`
所有 `num_heads` 个注意力头的查询投影。形状为 (num_heads * (d_model / num_heads), d_model)。行按形状为 (num_heads, d_k) 的矩阵排序，
因此 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。

- `layers.{num_layers}.attn.k_proj.weight`
所有 `num_heads` 个注意力头的键投影。形状为 (num_heads * (d_model / num_heads), d_model)。行按形状为 (num_heads, d_k) 的矩阵排序，
因此 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。

- `layers.{num_layers}.attn.v_proj.weight`
所有 `num_heads` 个注意力头的权重值投影。形状为 (num_heads * (d_model / num_heads), d_model)。行按形状为 (num_heads, d_v) 的矩阵排序，
因此 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。

- `layers.{num_layers}.attn.output_proj.weight`
多头自注意力输出投影的权重形状为 ((d_model / num_heads) * num_heads, d_model)。

- `layers.{num_layers}.ln1.weight`
Transformer 模块中第一个 RMSNorm 的仿射变换权重形状为 (d_model,)。

- `layers.{num_layers}.ffn.w1.weight`
前馈神经网络 (FFN) 中第一个线性变换的权重。形状为 (d_model, d_ff)。

- `layers.{num_layers}.ffn.w2.weight`
前馈神经网络 (FFN) 中第二个线性变换的权重。形状为 (d_ff, d_model)。

- `layers.{num_layers}.ffn.w3.weight`
前馈神经网络 (FFN) 中第三个线性变换的权重。形状为 (d_model, d_ff）。

- `layers.{num_layers}.ln2.weight`
第二个 RMSNorm 变换的仿射权重应用于 Transformer 模块。形状为 (d_model,)。

- `ln_final.weight`
应用于最终 Transformer 模块输出的 RMSNorm 变换的仿射权重。形状为 (d_model,)。

- `lm_head.weight`
语言模型输出嵌入的权重。形状为 (vocab_size, d_model)。
in_indices (Int[Tensor, "batch_size sequence_length"]) 用于运行语言模型的输入索引张量。形状为 (batch_size, sequence_length)，其中

`sequence_length` 至多为 `context_length`。
返回值：

Float[Tensor, "batch_size sequence_length vocab_size"]: 包含每个词元预测的未归一化

下一个词分布的张量。

"""

    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rms = RMSNorm(d_model, eps)
    state = {"weights":weights}
    rms.load_state_dict(state)
    return rms.forward(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], #inputs[i][j] 是第 i 个样本的第 j 个类别的未归一化 logit 值
    targets: Int[Tensor, " batch_size"]#形状为 (batch_size,) 的张量，包含正确类别的索引
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    batch_size = inputs.shape[0]
    # 沿着dim = 1找到最大
    #使用 .values 获取数值，使用 keepdim=True 保持形状为 (batch_size, 1) 以便广播
    line_max = inputs.max(dim=1,keepdim=True).values
    # Exp & Sum: (inputs - max).exp().sum(dim=1).log()
    # (B, V) - (B, 1) -> (B, V) -> sum -> (B,) -> log -> (B,)
    exp_sum = (inputs - line_max).exp().sum(dim=1).log()
    # Restore: 加上刚才减去的 max，得到完整的 Log-Sum-Exp。
    # 修正点：line_max 是 (B, 1)，exp_sum_log 是 (B,)。
    log_sum_exp = exp_sum + line_max.squeeze(1)
    # 4. 获取目标类别的 Logits (分子部分)
    # 使用高级索引 (Advanced Indexing)
    # inputs[i, targets[i]] 取出每个样本对应真实标签的 logit
    # torch.arange(batch_size) 生成 [0, 1, ..., batch-1]
    """
    inputs[row_index:Tensor, col_index:Tensor]
    那么 PyTorch 的规则是：逐元素配对索引
    inputs[row_index[i], col_index[i]]   for each i
    row_index = torch.arange(batch_size)  # [0, 1, 2]
    col_index = targets                   # [2, 3, 4]  ====>target_logits = tensor([
    inputs[0, 2],   # sample 0 的真实类别 logit
    inputs[1, 3],   # sample 1 的真实类别 logit
    inputs[2, 4],   # sample 2 的真实类别 logit
])

    """
    target_logits = inputs[torch.arange(batch_size), targets]
    # 5. 计算损失并求平均
    # Loss = LSE - Target_Logit
    losses = log_sum_exp - target_logits
    return torch.mean(losses)



def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    GradientClip(parameters,max_l2_norm).__call__()


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return Adamw



def run_get_lr_cosine_schedule(
    it: int,#要获取学习率的迭代次数
    max_learning_rate: float,#余弦学习率策略（带预热）的最大学习率
    min_learning_rate: float,#余弦学习率策略（带预热）的最小/最终学习率
    warmup_iters: int,#线性预热学习率所需的迭代次数
    cosine_cycle_iters: int,#余弦退火迭代次数
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    cosine = Cosine(max_learning_rate, min_learning_rate,warmup_iters, cosine_cycle_iters)
    return cosine.__call__(it)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],#分词器词汇表，一个从 int（词汇表中标记的 ID）到 bytes（标记字节）的映射
    merges: list[tuple[bytes, bytes]],#BPE 合并列表。列表中的每个元素都是一个字节元组 (<token1>, <token2>)，
    special_tokens: list[str] | None = None,#分词器使用的特殊字符串标记列表。这些字符串永远不会被拆分成多个标记，始终保持为一个标记。
) -> Any:#一个使用提供的词汇表、合并列表和特殊标记的 BPE 分词器。
    # 实例化上面的类
    tokenizer = BPETokenizer(vocab, merges, special_tokens)

    # 返回这个实例
    return tokenizer


def run_train_bpe(
    input_path: str | os.PathLike, #str 包含 BPE 分词器训练数据的文本文件的路径。
    vocab_size: int,    #一个正整数，定义最大最终词汇表大小（包括初始字节词汇表、合并生成的词汇项以及任何特殊标记）。
    special_tokens: list[str],#要添加到词汇表的字符串列表
    **kwargs,
) -> tuple[ dict[int, bytes], list[tuple[bytes, bytes]] ]:
    #dict[int, bytes] 分词器词汇表，从 int（词汇表中的分词 ID）到 bytes（分词字节）的映射。
    #训练生成的 BPE 合并列表。每个列表项都是一个字节元组 (<token1>, <token2>)，表示 <token1> 已与 <token2> 合并。合并应按创建顺序排序。
    # 1. 读取文本并转换为字节 ID 列表
    with open(input_path, "r",encoding='utf-8') as f:
        text = f.read()

        """
        改进：预处理特殊字符，若有特殊字符，按照特殊字符切分为多段
        """
    if special_tokens:
        # 对特殊字符进行转义（如 | -> \|），并用 | 连接成正则
        # 结果类似: "\<\|endoftext\|\>|\<\|pad\|\>"
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        text_segments = re.split(pattern, text)
    else:
        text_segments = [text]
        #使用GPT - 2风格的正则进行预分词
        # 这会将文本拆分为单词列表，例如 ["The", " world", " is", ...]
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    #统计单词出现的频次
    vocab_counts = Counter()
    for text in text_segments:
        #根据特殊字符切分后的各个语句统计单词
        if not text:continue
        #对片段进行切分
        words = re.findall(PAT, text)
        #统计单词内的频率
        for word in words:
            word_bytes = tuple(word.encode("utf-8"))
            vocab_counts[word_bytes] += 1
        #初始化基础词表
    vocab = {idx: bytes([idx]) for idx in range(256)}
     #预留合并的词表个数
    merges_num = vocab_size - 256 - len(special_tokens)
    merges_indices = []
    #主循环
    for i in range (merges_num):
        stats = get_stats(vocab_counts)
        if not stats:
            break
        # 找到频率最高的 pair；同频时按 GPT-2 约定用字节字典序断序 (vocab[p[0]], vocab[p[1]])
        max_pair = max(stats, key=lambda p: (stats[p], vocab[p[0]], vocab[p[1]]))
        merges_indices.append(max_pair)
        new_id = 256 + i
        vocab[new_id] = vocab[max_pair[0]] + vocab[max_pair[1]]
        vocab_counts = merge_ids(vocab_counts,max_pair,new_id)
    #后处理，将特殊字符放入词表
    current_idx = 256 + len(merges_indices)
    for token in special_tokens:
        vocab[current_idx] = token.encode("utf-8")
        current_idx += 1
    #格式化输出
    final_merges = []
    for p0,p1 in merges_indices:
        final_merges.append((vocab[p0], vocab[p1]))
    return vocab, final_merges























