"""
Lecture 03: Transformer 架构完整实现
从零实现Transformer的各个组件

主要内容:
1. Scaled Dot-Product Attention
2. Multi-Head Attention
3. Position Encoding (Sinusoidal & RoPE)
4. Feed-Forward Network
5. 完整Transformer Block
6. KV Cache实现

支持设备: CUDA / MPS (Apple Silicon) / CPU
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from utils.device_utils import DeviceManager, get_device


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention

    公式: Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Args:
        dropout: attention权重的dropout概率
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,       # [B, ..., L_q, d_k]
        key: torch.Tensor,         # [B, ..., L_k, d_k]
        value: torch.Tensor,       # [B, ..., L_k, d_v]
        mask: Optional[torch.Tensor] = None,  # [B, ..., L_q, L_k]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Returns:
            output: [B, ..., L_q, d_v]
            attention_weights: [B, ..., L_q, L_k]
        """
        d_k = query.size(-1)

        # Step 1: Compute attention scores
        # QK^T: [B, ..., L_q, d_k] @ [B, ..., d_k, L_k] → [B, ..., L_q, L_k]
        scores = torch.matmul(query, key.transpose(-2, -1))

        # Step 2: Scale by √d_k
        scores = scores / math.sqrt(d_k)

        # Step 3: Apply mask (if provided)
        if mask is not None:
            # mask为True的位置会被设为-inf
            scores = scores.masked_fill(mask, float('-inf'))

        # Step 4: Softmax normalization
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Step 5: Compute output
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention

    将Q, K, V投影到多个head，分别计算attention后合并

    Args:
        d_model: 模型维度
        num_heads: attention头数
        dropout: dropout概率
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个head的维度

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Attention module
        self.attention = ScaledDotProductAttention(dropout)

    def forward(
        self,
        query: torch.Tensor,  # [B, L_q, d_model]
        key: torch.Tensor,    # [B, L_k, d_model]
        value: torch.Tensor,  # [B, L_k, d_model]
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Returns:
            output: [B, L_q, d_model]
            attention_weights: [B, num_heads, L_q, L_k]
        """
        batch_size = query.size(0)

        # Step 1: Linear projections
        Q = self.W_q(query)  # [B, L_q, d_model]
        K = self.W_k(key)    # [B, L_k, d_model]
        V = self.W_v(value)  # [B, L_k, d_model]

        # Step 2: Reshape for multi-head
        # [B, L, d_model] → [B, L, num_heads, d_k] → [B, num_heads, L, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Step 3: Apply attention
        # 如果有mask，需要扩展到[B, num_heads, L_q, L_k]
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, L_q, L_k]

        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # Step 4: Concatenate heads
        # [B, num_heads, L_q, d_k] → [B, L_q, num_heads, d_k] → [B, L_q, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # Step 5: Final projection
        output = self.W_o(attn_output)

        return output, attn_weights


class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦位置编码 (原始Transformer论文)

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model]

        Returns:
            x + positional encoding: [B, L, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RotaryPositionalEncoding(nn.Module):
    """
    RoPE (Rotary Position Embedding)

    现代LLM常用的位置编码方式（如LLaMA, GPT-NeoX）

    核心思想: 通过旋转向量来编码相对位置
    """

    def __init__(self, d_model: int, max_len: int = 5000, base: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base

        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

        # 预计算sin和cos
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        """构建sin/cos缓存"""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, d_model]
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """旋转操作的辅助函数"""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        """
        应用RoPE到query和key

        Args:
            q: [B, num_heads, L, d_k]
            k: [B, num_heads, L, d_k]
            position_ids: [B, L]

        Returns:
            rotated (q, k)
        """
        seq_len = q.size(2)

        if seq_len > self.max_len:
            self._build_cache(seq_len)

        cos = self.cos_cached[:, :seq_len, ...]
        sin = self.sin_cached[:, :seq_len, ...]

        # 扩展维度匹配multi-head
        cos = cos.unsqueeze(1)  # [1, 1, L, d_model]
        sin = sin.unsqueeze(1)  # [1, 1, L, d_model]

        # 应用旋转
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network

    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    通常d_ff = 4 * d_model
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = 'gelu'
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.SiLU()  # swish

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model]

        Returns:
            output: [B, L, d_model]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    完整的Transformer Block

    结构:
    1. LayerNorm + Multi-Head Attention + Residual
    2. LayerNorm + Feed-Forward + Residual

    使用Pre-Norm（先normalize再计算）
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        use_rope: bool = True
    ):
        super().__init__()

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Multi-Head Attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Position encoding (RoPE)
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEncoding(d_model // num_heads)

        # Feed-Forward Network
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model]
            mask: [B, L, L] causal mask

        Returns:
            output: [B, L, d_model]
        """
        # Pre-Norm Self-Attention
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x, mask)
        x = residual + self.dropout(attn_output)

        # Pre-Norm FFN
        residual = x
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = residual + self.dropout(ffn_output)

        return x


class KVCache:
    """
    KV Cache实现 - 用于自回归生成加速

    避免每次生成新token时重复计算之前token的K和V
    """

    def __init__(self, batch_size: int, max_seq_len: int, num_heads: int, head_dim: int, device):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        # 缓存shape: [batch_size, num_heads, max_seq_len, head_dim]
        self.cache_k = torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device)
        self.cache_v = torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device)

        self.current_len = 0

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新KV Cache

        Args:
            k: 新的key [B, num_heads, new_len, head_dim]
            v: 新的value [B, num_heads, new_len, head_dim]

        Returns:
            完整的k和v（包括缓存的历史）
        """
        new_len = k.size(2)

        # 更新缓存
        self.cache_k[:, :, self.current_len:self.current_len + new_len, :] = k
        self.cache_v[:, :, self.current_len:self.current_len + new_len, :] = v

        self.current_len += new_len

        # 返回完整历史
        return (
            self.cache_k[:, :, :self.current_len, :],
            self.cache_v[:, :, :self.current_len, :]
        )

    def reset(self):
        """重置缓存"""
        self.cache_k.zero_()
        self.cache_v.zero_()
        self.current_len = 0


def create_causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """
    创建因果注意力掩码（下三角为False，上三角为True）

    Returns:
        mask: [seq_len, seq_len] - True表示位置会被mask
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


def demonstrate_attention():
    """演示Attention机制"""

    dm = DeviceManager()

    print("\n" + "=" * 70)
    print("🔍 Attention Mechanism Demo")
    print("=" * 70)

    batch_size = 2
    seq_len = 4
    d_model = 64
    num_heads = 4

    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model, device=dm.device)

    # 1. Scaled Dot-Product Attention
    print("\n1. Scaled Dot-Product Attention:")
    sdp_attn = ScaledDotProductAttention().to(dm.device)

    # 简单的QKV（相同输入，Self-Attention）
    q = k = v = x
    output, weights = sdp_attn(q, k, v)

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {weights.shape}")
    print(f"   Weights sum (should be 1): {weights[0, 0].sum().item():.4f}")

    # 2. Multi-Head Attention
    print("\n2. Multi-Head Attention:")
    mha = MultiHeadAttention(d_model, num_heads).to(dm.device)

    output, weights = mha(x, x, x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {weights.shape}")

    # 3. 带Causal Mask的Attention
    print("\n3. Causal Masked Attention:")
    mask = create_causal_mask(seq_len, device=dm.device)
    output, weights = mha(x, x, x, mask.unsqueeze(0))

    print(f"   Causal mask (True = masked):")
    for row in mask:
        print(f"     {row.int().tolist()}")

    print(f"\n   Attention weights (should be lower-triangular):")
    weights_sample = weights[0, 0].detach().cpu()
    for i, row in enumerate(weights_sample):
        print(f"     {[f'{v:.2f}' for v in row.tolist()]}")


def demonstrate_position_encoding():
    """演示位置编码"""

    dm = DeviceManager()

    print("\n" + "=" * 70)
    print("📍 Position Encoding Demo")
    print("=" * 70)

    d_model = 64
    seq_len = 10

    # 1. Sinusoidal Position Encoding
    print("\n1. Sinusoidal Position Encoding:")
    sinusoidal_pe = SinusoidalPositionalEncoding(d_model).to(dm.device)

    x = torch.zeros(1, seq_len, d_model, device=dm.device)
    x_with_pe = sinusoidal_pe(x)

    print(f"   Position encoding shape: {x_with_pe.shape}")
    print(f"   PE[0, 0:5] (first 5 dims at pos 0): {x_with_pe[0, 0, :5].tolist()}")
    print(f"   PE[0, :, 0] (dim 0 across positions): {x_with_pe[0, :, 0].tolist()}")

    # 2. RoPE
    print("\n2. Rotary Position Encoding (RoPE):")
    head_dim = d_model // 4  # 4 heads
    rope = RotaryPositionalEncoding(head_dim).to(dm.device)

    q = torch.randn(1, 4, seq_len, head_dim, device=dm.device)  # 4 heads
    k = torch.randn(1, 4, seq_len, head_dim, device=dm.device)

    q_rotated, k_rotated = rope(q, k)
    print(f"   Q shape: {q.shape} → {q_rotated.shape}")
    print(f"   K shape: {k.shape} → {k_rotated.shape}")

    # 3. 位置编码的可视化特性
    print("\n📊 Position Encoding Properties:")
    print("""
    Sinusoidal:
    - 固定的、非学习的编码
    - 不同位置的编码正交性好
    - 可以泛化到比训练时更长的序列

    RoPE:
    - 编码相对位置信息
    - 在attention计算时应用（而非加到embedding）
    - LLaMA、GPT-NeoX等模型使用
    - 更好的长文本外推能力
    """)


def demonstrate_transformer_block():
    """演示完整Transformer Block"""

    dm = DeviceManager()

    print("\n" + "=" * 70)
    print("🏗️ Transformer Block Demo")
    print("=" * 70)

    batch_size = 2
    seq_len = 32
    d_model = 256
    num_heads = 8
    d_ff = 1024

    # 创建Transformer Block
    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1,
        use_rope=False
    ).to(dm.device)

    # 输入
    x = torch.randn(batch_size, seq_len, d_model, device=dm.device)
    mask = create_causal_mask(seq_len, device=dm.device).unsqueeze(0)

    # 前向传播
    output = block(x, mask)

    print(f"\n   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")

    # 参数统计
    total_params = sum(p.numel() for p in block.parameters())
    print(f"\n   Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # 分解参数
    attn_params = sum(p.numel() for p in block.attention.parameters())
    ffn_params = sum(p.numel() for p in block.ffn.parameters())
    norm_params = sum(p.numel() for p in block.norm1.parameters())
    norm_params += sum(p.numel() for p in block.norm2.parameters())

    print(f"   - Attention: {attn_params:,}")
    print(f"   - FFN: {ffn_params:,}")
    print(f"   - LayerNorm: {norm_params:,}")


def demonstrate_kv_cache():
    """演示KV Cache"""

    dm = DeviceManager()

    print("\n" + "=" * 70)
    print("⚡ KV Cache Demo")
    print("=" * 70)

    batch_size = 1
    max_seq_len = 32
    num_heads = 4
    head_dim = 16

    # 创建KV Cache
    kv_cache = KVCache(batch_size, max_seq_len, num_heads, head_dim, dm.device)

    print("\n   Simulating autoregressive generation with KV Cache:")

    # 模拟生成过程
    for step in range(5):
        # 新token的K和V
        new_k = torch.randn(batch_size, num_heads, 1, head_dim, device=dm.device)
        new_v = torch.randn(batch_size, num_heads, 1, head_dim, device=dm.device)

        # 更新缓存
        full_k, full_v = kv_cache.update(new_k, new_v)

        print(f"   Step {step + 1}: New K/V shape = {new_k.shape}")
        print(f"           Full K/V shape = {full_k.shape}")

    print("\n📊 KV Cache Benefits:")
    print("""
    Without KV Cache (每次从头计算):
    - Step 1: 计算1个token的KV
    - Step 2: 计算2个token的KV
    - Step n: 计算n个token的KV
    - 总计: O(n²) 计算量

    With KV Cache:
    - Step 1: 计算1个token的KV，存入cache
    - Step 2: 计算1个新token的KV，与cache合并
    - Step n: 计算1个新token的KV
    - 总计: O(n) 计算量

    ⚡ 速度提升巨大，尤其是长序列生成！
    """)


def run_all_demos():
    """运行所有演示"""

    # 1. Attention机制
    demonstrate_attention()

    # 2. 位置编码
    demonstrate_position_encoding()

    # 3. Transformer Block
    demonstrate_transformer_block()

    # 4. KV Cache
    demonstrate_kv_cache()


if __name__ == '__main__':
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                                                                    ║")
    print("║           CS336 Lecture 03 - Transformer Architecture             ║")
    print("║                                                                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    run_all_demos()

    print("\n" + "=" * 70)
    print("✅ Transformer Demo完成！")
    print("=" * 70)
    print("\n💡 Key Takeaways:")
    print("  - Attention enables dynamic weighting of input positions")
    print("  - Multi-head allows learning different relationship patterns")
    print("  - Position encoding adds sequential information")
    print("  - KV Cache is crucial for efficient generation")
    print("\n")
