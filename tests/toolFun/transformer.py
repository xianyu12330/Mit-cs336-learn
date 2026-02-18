
import torch
from sympy import false
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features #输入的维度
                 , out_features, #输出的维度
                 device=None, #存储参数的设备-》cpu/gpu
                 dtype=None#参数的数据类型
                 ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.W = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        # 对权重进行 Xavier 初始化,std为标准差
        std = 2 / (in_features + out_features)**0.5
        """将输入张量的值设置为截断的正态分布的随机数
        paddle.nn.init.trunc_normal_(tensor , mean=0.0正态分布的均值，默认值为 0.0
        , std=1.0, 正态分布的标准差，默认值为 1.0
        a=-2.0, 截断的下限，默认值为-2.0
        b=2.0截断的上限，默认值为 2.0
        )
        """
        nn.init.trunc_normal_(self.W, std=std,a=-3 * std,b=3 * std)


    def forward(self, x:torch.Tensor)->torch.Tensor:
        return x @ self.W.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, #词汇表的大小
                 embedding_dim:int,#嵌入向量的维数
                 device=None, #存储的设备gpu/cpu
                 dtype=None #参数数据类型
                 ):
        super().__init__()
        #词汇表大小
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        """创建一个用于存储“所有单词向量”的矩阵
        矩阵的大小应该是 [num_embeddings, 行数,词汇表有多大,每一行对应一个词
        embedding_dim   每个词用多少维度的向量来表示（例如 512 维）
        ]
        """
        std = 1
        self.embed_table = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embed_table, std=std,a=-3 * std,b=3 * std)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.embed_table[x]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, #模型的隐藏维度
                 eps: float = 1e-5,#数值稳定性的 Epsilon 值
                 device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        #通常初始化为 1（即初始时不改变数据的幅度）
        #作用：根据任务的实际需要，把某些特征“拉长”或“缩短”（恢复表达能力）。看哪个特征更重要
        self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # 题目要求对于不同的精度要先转换为 float32 再进行归一化，最后再转换回原来的精度
        in_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        norm = x_f32 / (x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        x_norm = norm.to(in_dtype)
        return x_norm * self.weights

class SwiGLU(nn.Module):
    def __init__(self, d_model: int,d_ff:int = None,device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_ff:
            self.d_ff = d_ff
        else:
            ff_dim = int(d_model * 8 / 3)
            # 下面这个公式是标准的向上取整到 64 倍数的写法： ((x + 63) // 64) * 64
            self.d_ff = ((ff_dim + 63) // 64) * 64
        #w1,w2,w3->[d_model,d_ff]
        self.W1 = nn.Parameter(torch.empty(self.d_ff, d_model, device=device, dtype=dtype))
        self.W2 = nn.Parameter(torch.empty(d_model, self.d_ff, device=device, dtype=dtype))
        self.W3 = nn.Parameter(torch.empty(self.d_ff, d_model, device=device, dtype=dtype))

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # x 的形状: [Batch, Seq, d_model]
        #1.门控投影(Gate) - W1
        # 公式: gate = x @ W1.T
        # 形状: [..., d_model] @ [d_model, d_ff] -> [..., d_ff]
        gate = x @ self.W1.T
        ## 2. 上投影 (Value) - W3
        # 公式: value = x @ W3.T
        # 形状: [..., d_model] @ [d_model, d_ff] -> [..., d_ff]
        value = x @ self.W3.T

        # 3. 激活函数 (SiLU): SiLU(x) = x * sigmoid(x)，不是单纯的 sigmoid(x)
        gate = gate * torch.sigmoid(gate)  # 等价于 F.silu(gate)
        # 4. 逐元素相乘 (Element-wise Multiply)
        # SwiGLU 核心逻辑
        hidden = gate * value

        # 5. 下投影 (Output) - W2
        # 公式: out = hidden @ W2.T
        # 形状: [..., d_ff] @ [d_ff, d_model] -> [..., d_model]
        return hidden @ self.W2.T

class Rope(nn.Module):
    def __init__(self, theta: float,#rope的Θ值
                 d_k: int,#查询向量和关键向量的维度
                 max_seq_len: int,#输入的最大序列长度
                 device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        #生成频率计算公式，生成一个长度为 d_k / 2 的向量-----[d_k//2,]-----[f0,f1,f2......]
        freq = 1.0 / (self.theta **((torch.arange(0,self.d_k,2)).float()/self.d_k))
        #获得一句话每个词的顺序,RoPE 的位置变量 pos[0号token，1号token，。。。。。n号token]
        #一维张量[max_seq_len,]-----[0, 1, 2, ..., max_seq_len-1]
        seq_idx = torch.arange(max_seq_len, device=device, dtype=freq.dtype)
        #将Θ与顺序融合,旋转角公式
        #[max_seq_len,d_k//2]
        idx_theta = torch.einsum("m,d->md",seq_idx,freq)
        #
        idx_theta2 = torch.cat((idx_theta, idx_theta), dim=-1)
        #[max_seq_len,d_k//2]
        self.register_buffer('cos_cache',idx_theta2.cos())
        self.register_buffer('sin_cache',idx_theta2.sin())


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
       x:[B, Seq_Len, d_k]
       token_positions:[B, max_seq_len]
        x 是输入的稠密向量，token_positions 是 token 的位置信息
        输出 shape = 索引 tensor shape + 被索引张量的剩余维度
        根据max_seq_len查字典，输入了 [Batch, Seq_Len] 个位置-》拿回 [Batch, Seq_Len] 个 cos 向量-》每个向量长度 = d_k//2
        """# 查表后 cos/sin 形状: [..., Seq, d_k]，取前 d_k/2 维与 x1/x2 配对
        #RoPE 对 (x_{2i}, x_{2i+1}) 成对旋转，cos/sin 只需 d_k/2 维

        cos = self.cos_cache[token_positions][..., : self.d_k // 2]  # [..., d_k/2]
        sin = self.sin_cache[token_positions][..., : self.d_k // 2]
        # token_positions 为 1D 时需加 batch 维；x 为 (B, Seq, d_k) 或 (B, num_heads, Seq, d_k) 时需能与 cos 广播
        if cos.dim() == 2:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        if cos.dim() == 3:
            # (1, T, d_k/2) -> (1, 1, T, d_k/2) 以便与 (B, num_heads, T, head_dim/2) 广播
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        # 奇数、偶数位置
        x1 = x[..., 0::2]  # 偶数 [..., d_k/2]
        x2 = x[..., 1::2]  # 奇数 [..., d_k/2]
        """旋转: x' = x*cos - y*sin, y' = x*sin + y*cos"""
        output_x = x1 * cos - x2 * sin
        output_y = x2 * cos + x1 * sin
        out = torch.stack([output_x, output_y], dim=-1)
        return out.flatten(-2)

class DotAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q: torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask:torch.Tensor |None = None)->torch.Tensor:
        dim = q.shape[-1]
        k_t = k.transpose(-2,-1)
        score = torch.matmul(q,k_t) / (dim ** 0.5)
        #最终生成的新张量包含了在掩码位置上被替换的值，其余位置保持原样
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        attn_weight = torch.softmax(score, dim=-1)
        return torch.matmul(attn_weight, v)

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,heads:int):
        super().__init__()
        self.heads = heads #头的数量
        self.d_model = d_model #总维度
        # 检查能否整除
        assert d_model % heads == 0, "d_model must be divisible by num_heads"
        # 每个头的维度
        self.head_dim = d_model // heads
        """定义线性层 wq,wk,wv
        直接定义 3 个大矩阵，形状都是 [d_model, d_model]。计算完后再拆分维度。
        """
        self.wq = nn.Parameter(torch.empty(d_model, d_model))
        self.wk = nn.Parameter(torch.empty(d_model, d_model))
        self.wv = nn.Parameter(torch.empty(d_model, d_model))
        self.out_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.attention = DotAttention()

    def forward(self,x:torch.Tensor,mask:torch.Tensor | None = None)->torch.Tensor:
        #1.投影 (Project)：输入 X分别乘以 W_Q, W_K, W_V。
        # x @ W: [B, T, D] @ [D, D] -> [B, T, D]
        B, T, D = x.shape
        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv
        #2.将d_model拆分为head_dim * heads

        q = q.view(B,T,self.heads,self.head_dim)
        k = k.view(B,T,self.heads,self.head_dim)
        v = v.view(B,T,self.heads,self.head_dim)
        #3.把 num_heads 维度移到前面，让每个头独立计算
        """
        转换完后[batch, seq_len, heads, head_dim]->[batch, heads, seq_len, head_dim]
        Q[batch, head] = [seq_len, head_dim]
        """
        query = q.transpose(1,2)
        key = k.transpose(1,2)
        value = v.transpose(1,2)
        # 4. 因果掩码：decoder-only 模型，位置 i 只能 attend 到 j<=i
        if mask is None:
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=query.dtype)).view(1, 1, T, T)
        else:
            causal_mask = mask
        atten_out = self.attention.forward(query, key, value, mask=causal_mask)
        out = atten_out.transpose(1,2)
        # 关键修正：必须使其连续，否则 view/reshape 会报错
        out = out.contiguous()
        out = out.reshape(B,T,self.d_model)
        return out @ self.out_proj

class MultiHeadAttentionWithRoPE(nn.Module):
    """带 RoPE 的多头自注意力，对应 adapters.run_multihead_self_attention_with_rope 的逻辑。"""

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.rope = Rope(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len)
        self.attn = DotAttention()

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if token_positions is None:
            token_positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=q.dtype)).view(1, 1, T, T)
        out = self.attn(q, k, v, mask=causal_mask)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.output_proj(out)

    def load_ref_state_dict(self, state: dict):
        """参考 state_dict 与 PyTorch 一致：(out, in)，计算为 x @ W.T，直接加载。"""
        self.q_proj.W.data = state["attn.q_proj.weight"].clone()
        self.k_proj.W.data = state["attn.k_proj.weight"].clone()
        self.v_proj.W.data = state["attn.v_proj.weight"].clone()
        self.output_proj.W.data = state["attn.output_proj.weight"].clone()

class TransformerBlock(nn.Module):
    """预归一化 Transformer 块：ln1 -> MHA+RoPE -> 残差 -> ln2 -> SwiGLU -> 残差。"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, eps: float = 1e-5):
        super().__init__()
        self.ln1 = RMSNorm(d_model, eps=eps)
        self.attn = MultiHeadAttentionWithRoPE(d_model, num_heads, max_seq_len, theta)
        self.ln2 = RMSNorm(d_model, eps=eps)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        x = self.attn(x, token_positions)
        x = residual + x
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        return residual + x

    def load_ref_state_dict(self, state: dict):
        self.ln1.weights.data = state["ln1.weight"]
        self.attn.load_ref_state_dict(state)
        self.ln2.weights.data = state["ln2.weight"]
        # ref: w1 (d_model, d_ff), w2 (d_ff, d_model), w3 (d_model, d_ff); 我们 W1 (d_ff, d_model) 等
        self.ffn.W1.data = state["ffn.w1.weight"].clone()
        self.ffn.W2.data = state["ffn.w2.weight"].clone()
        self.ffn.W3.data = state["ffn.w3.weight"].clone()

class TransformerLM(nn.Module):
    """Transformer 语言模型：embedding -> N × TransformerBlock -> ln_final -> lm_head。"""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(idx)
        B, T, _ = x.shape
        token_positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)

    def load_ref_state_dict(self, state: dict):
        self.token_embeddings.embed_table.data = state["token_embeddings.weight"]
        for i in range(self.num_layers):
            prefix = f"layers.{i}."
            block_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            self.layers[i].load_ref_state_dict(block_state)
        self.ln_final.weights.data = state["ln_final.weight"]
        self.lm_head.W.data = state["lm_head.weight"]

class CrossEntropyLoss(nn.Module):
    """对 logits 与 target 索引计算平均交叉熵（数值稳定的 log-sum-exp）。"""

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        line_max = inputs.max(dim=1, keepdim=True).values
        exp_sum = (inputs - line_max).exp().sum(dim=1).log()
        log_sum_exp = exp_sum + line_max.squeeze(1)
        target_logits = inputs[torch.arange(batch_size, device=inputs.device), targets]
        return (log_sum_exp - target_logits).mean()

class SoftMax(nn.Module):
    def forward(self, inputs: torch.Tensor,dim:int = -1) -> torch.Tensor:
       #使用 keepdim=True 防止后续广播出错
        line_max = inputs.max(dim=dim, keepdim=True).values
        # 减去最大值（这一步很安全，不会改变 Softmax 的结果）
        curr_feature = inputs - line_max
        exp_feature = curr_feature.exp()
        #求和
        sum_exp_feature = exp_feature.sum(dim=dim, keepdim=True)
        #归一化
        return exp_feature / sum_exp_feature

#top-p过滤逻辑
class TopP(nn.Module):
    def top_p_filter(self,logits:torch.Tensor,p:float) -> torch.Tensor:
        #对词表分值进行降序排序
        sort_logits,sort_idx = torch.sort(logits,dim=-1,descending=True)
        soft_ = SoftMax()
        #计算累计的概率分布,每个对应位置以及之前的总和如[0.65, 0.24, 0.11] -》[0.65, 0.89, 1.00
        cumulative_probs = torch.cumsum(soft_.forward(sort_logits,dim=-1),dim=-1)
        #创建掩码，去掉累计超过p的token
        #把所有超过p的位置标记为true
        sort_idx_remove = cumulative_probs > p
        #确保至少保留第一个词（最高概率），并且我们要保留第一个“使概率超过 p”的那个词。做法是把标记位向右移动一格。
        sort_idx_remove[...,1:] = sort_idx_remove[...,:-1].clone()
        sort_idx_remove[...,0] = false

        #被移除的token设置为负无穷
        #将sort_idx_remove中的值，按照sort_idx指定的位置，放回原始顺序的位置上
        #例 [A, B, C]-》[False, True, True]
        idx_to_remove = sort_idx_remove.scatter(1,sort_idx,sort_idx_remove)
        #[2.0, -inf, -inf]  # 只保留A，屏蔽B和C
        logits = logits.masked_fill_(idx_to_remove,float("-inf"))
        return logits

            








