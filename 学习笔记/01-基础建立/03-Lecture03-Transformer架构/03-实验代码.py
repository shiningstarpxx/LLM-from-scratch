"""
Lecture 03: Transformer Architecture 实验代码

本文件包含Transformer架构的核心组件实现和对比实验
对应深度讨论记录中的各个主题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple


# =============================================================================
# 实验1: Scaled Dot-Product Attention (Q3验证)
# =============================================================================

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    标准Scaled Dot-Product Attention

    验证Q3: 为什么需要除以sqrt(d_k)？
    """
    d_k = Q.size(-1)

    # 计算scores
    scores = Q @ K.transpose(-2, -1)

    # 是否缩放
    if scale:
        scores = scores / (d_k ** 0.5)

    # 应用mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax
    attention_weights = F.softmax(scores, dim=-1)

    # 加权求和
    output = attention_weights @ V

    return output, attention_weights


def experiment_scaling_factor():
    """实验：验证scaling factor的必要性"""
    print("=" * 60)
    print("实验1: Scaling Factor的作用")
    print("=" * 60)

    # 不同维度
    d_k_values = [64, 256, 512, 1024]

    for d_k in d_k_values:
        Q = torch.randn(1, 10, d_k)
        K = torch.randn(1, 10, d_k)
        V = torch.randn(1, 10, d_k)

        # 不缩放
        _, attn_no_scale = scaled_dot_product_attention(Q, K, V, scale=False)

        # 缩放
        _, attn_scaled = scaled_dot_product_attention(Q, K, V, scale=True)

        # 计算attention权重的熵（衡量分布的均匀程度）
        def entropy(weights):
            eps = 1e-10
            return -torch.sum(weights * torch.log(weights + eps), dim=-1).mean()

        print(f"\nd_k = {d_k}")
        print(f"  不缩放: 熵 = {entropy(attn_no_scale):.4f}")
        print(f"  缩放后: 熵 = {entropy(attn_scaled):.4f}")
        print(f"  最大权重: 不缩放={attn_no_scale.max():.4f}, "
              f"缩放={attn_scaled.max():.4f}")


# =============================================================================
# 实验2: Multi-Head Attention参数量验证 (Q8)
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention实现"""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 注意：投影矩阵大小都是 [d_model, d_model]
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = X.shape

        # 线性投影
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        # Reshape: [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        output, _ = scaled_dot_product_attention(Q, K, V, mask)

        # 拼接
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # 输出投影
        output = self.W_O(output)

        return output


def experiment_multihead_parameters():
    """实验：验证Multi-Head参数量"""
    print("\n" + "=" * 60)
    print("实验2: Multi-Head Attention参数量")
    print("=" * 60)

    d_model = 512

    # 不同heads数
    for num_heads in [1, 4, 8, 16]:
        model = MultiHeadAttention(d_model, num_heads)

        total_params = sum(p.numel() for p in model.parameters())

        print(f"\nheads = {num_heads:2d}")
        print(f"  d_k = {d_model // num_heads}")
        print(f"  总参数: {total_params:,}")
        print(f"  理论值: {4 * d_model * d_model:,}")


# =============================================================================
# 实验3: Pre-LN vs Post-LN (Q18)
# =============================================================================

class PostLNTransformerBlock(nn.Module):
    """Post-LN Transformer Block (原始)"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Post-LN: x = LN(x + Attention(x))
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ffn(x))
        return x


class PreLNTransformerBlock(nn.Module):
    """Pre-LN Transformer Block (现代)"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # Pre-LN: x = x + Attention(LN(x))
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


def experiment_preln_vs_postln():
    """实验：对比Pre-LN和Post-LN的数值稳定性"""
    print("\n" + "=" * 60)
    print("实验3: Pre-LN vs Post-LN")
    print("=" * 60)

    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 12

    x = torch.randn(1, 10, d_model)

    # Post-LN
    x_post = x.clone()
    post_norms = [x_post.norm().item()]

    for i in range(num_layers):
        block = PostLNTransformerBlock(d_model, num_heads, d_ff)
        x_post = block(x_post)
        post_norms.append(x_post.norm().item())

    # Pre-LN
    x_pre = x.clone()
    pre_norms = [x_pre.norm().item()]

    for i in range(num_layers):
        block = PreLNTransformerBlock(d_model, num_heads, d_ff)
        x_pre = block(x_pre)
        pre_norms.append(x_pre.norm().item())

    print(f"\n数值范围对比:")
    print(f"  Post-LN: 初始={post_norms[0]:.2f}, 最终={post_norms[-1]:.2f}")
    print(f"  Pre-LN:  初始={pre_norms[0]:.2f}, 最终={pre_norms[-1]:.2f}")

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.plot(post_norms, label='Post-LN', marker='o')
    plt.plot(pre_norms, label='Pre-LN', marker='s')
    plt.xlabel('Layer')
    plt.ylabel('Norm of output')
    plt.title('Pre-LN vs Post-LN: Numerical Stability')
    plt.legend()
    plt.grid(True)
    plt.savefig('preln_vs_postln.png', dpi=150, bbox_inches='tight')
    print("\n  图表已保存: preln_vs_postln.png")


# =============================================================================
# 实验4: Causal Masking (Q20)
# =============================================================================

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """创建因果mask矩阵"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return ~mask  # 反转：True=可见，False=mask


def experiment_causal_masking():
    """实验：可视化Causal Mask"""
    print("\n" + "=" * 60)
    print("实验4: Causal Masking")
    print("=" * 60)

    seq_len = 8
    d_k = 64

    Q = torch.randn(1, seq_len, d_k)
    K = torch.randn(1, seq_len, d_k)
    V = torch.randn(1, seq_len, d_k)

    # 创建mask
    mask = create_causal_mask(seq_len)

    print(f"\nCausal Mask (seq_len={seq_len}):")
    print(mask.int())

    # 计算attention
    _, attention_weights = scaled_dot_product_attention(
        Q, K, V, mask=mask.unsqueeze(0)
    )

    print(f"\nAttention Weights:")
    print(attention_weights[0].detach().numpy())

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Mask
    axes[0].imshow(mask.int(), cmap='RdYlGn', interpolation='nearest')
    axes[0].set_title('Causal Mask (Green=Visible, Red=Masked)')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')

    # Attention weights
    im = axes[1].imshow(attention_weights[0].detach().numpy(),
                        cmap='viridis', interpolation='nearest')
    axes[1].set_title('Attention Weights')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig('causal_masking.png', dpi=150, bbox_inches='tight')
    print("\n  图表已保存: causal_masking.png")


# =============================================================================
# 实验5: KV缓存优化 (Q21)
# =============================================================================

class AttentionWithKVCache:
    """带KV缓存的Attention（推理优化）"""

    def __init__(self, d_model: int, d_k: int):
        self.d_model = d_model
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_k)

        # 缓存
        self.K_cache = None
        self.V_cache = None

    def forward(self, x_new: torch.Tensor, use_cache: bool = True):
        """
        x_new: [batch, 1, d_model] 新生成的token
        """
        # 计算新token的Q、K、V
        Q_new = self.W_Q(x_new)
        K_new = self.W_K(x_new)
        V_new = self.W_V(x_new)

        if use_cache and self.K_cache is not None:
            # 使用缓存
            K = torch.cat([self.K_cache, K_new], dim=1)
            V = torch.cat([self.V_cache, V_new], dim=1)
        else:
            K = K_new
            V = V_new

        # 更新缓存
        if use_cache:
            self.K_cache = K
            self.V_cache = V

        # 计算attention
        output, _ = scaled_dot_product_attention(Q_new, K, V)

        return output

    def reset_cache(self):
        self.K_cache = None
        self.V_cache = None


def experiment_kv_cache():
    """实验：KV缓存的加速效果"""
    print("\n" + "=" * 60)
    print("实验5: KV缓存优化")
    print("=" * 60)

    d_model = 512
    d_k = 64
    seq_len = 100

    model = AttentionWithKVCache(d_model, d_k)

    # 无缓存
    import time

    model.reset_cache()
    start = time.time()
    for i in range(seq_len):
        x = torch.randn(1, i+1, d_model)
        # 重新计算所有
        _ = model.forward(x[:, -1:], use_cache=False)
    time_no_cache = time.time() - start

    # 有缓存
    model.reset_cache()
    start = time.time()
    for i in range(seq_len):
        x = torch.randn(1, 1, d_model)
        _ = model.forward(x, use_cache=True)
    time_with_cache = time.time() - start

    print(f"\n生成{seq_len}个tokens:")
    print(f"  无KV缓存: {time_no_cache:.4f}s")
    print(f"  有KV缓存: {time_with_cache:.4f}s")
    print(f"  加速比: {time_no_cache / time_with_cache:.2f}x")


# =============================================================================
# 实验6: Linear Attention复杂度对比 (Q22)
# =============================================================================

def linear_attention(Q, K, V, feature_map=None):
    """
    Linear Attention: φ(Q) @ (φ(K).T @ V)
    复杂度: O(nd²)
    """
    if feature_map is None:
        # 简单的特征映射: elu(x) + 1
        feature_map = lambda x: F.elu(x) + 1

    Q_prime = feature_map(Q)
    K_prime = feature_map(K)

    # 关键：先算 K.T @ V (O(d²n))
    KV = K_prime.transpose(-2, -1) @ V  # [batch, d_k, d_v]

    # 再算 Q @ KV (O(nd²))
    output = Q_prime @ KV

    # 归一化
    normalizer = Q_prime @ K_prime.sum(dim=-2, keepdim=True).transpose(-2, -1)
    output = output / (normalizer + 1e-6)

    return output


def experiment_linear_attention_complexity():
    """实验：对比标准Attention和Linear Attention的复杂度"""
    print("\n" + "=" * 60)
    print("实验6: Linear Attention复杂度对比")
    print("=" * 60)

    d_k = 64
    seq_lengths = [256, 512, 1024, 2048, 4096]

    import time

    print(f"\n{'Seq Len':<10}{'Standard (ms)':<15}{'Linear (ms)':<15}{'Speedup':<10}")
    print("-" * 50)

    for n in seq_lengths:
        Q = torch.randn(1, n, d_k)
        K = torch.randn(1, n, d_k)
        V = torch.randn(1, n, d_k)

        # 标准Attention
        start = time.time()
        for _ in range(10):
            _ = scaled_dot_product_attention(Q, K, V)
        time_standard = (time.time() - start) * 100  # ms

        # Linear Attention
        start = time.time()
        for _ in range(10):
            _ = linear_attention(Q, K, V)
        time_linear = (time.time() - start) * 100  # ms

        speedup = time_standard / time_linear if time_linear > 0 else 0

        print(f"{n:<10}{time_standard:<15.2f}{time_linear:<15.2f}{speedup:<10.2f}x")


# =============================================================================
# 主函数
# =============================================================================

def main():
    """运行所有实验"""
    print("\n" + "=" * 60)
    print("Lecture 03: Transformer Architecture 实验代码")
    print("=" * 60)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 运行实验
    experiment_scaling_factor()
    experiment_multihead_parameters()
    experiment_preln_vs_postln()
    experiment_causal_masking()
    experiment_kv_cache()
    experiment_linear_attention_complexity()

    print("\n" + "=" * 60)
    print("所有实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
