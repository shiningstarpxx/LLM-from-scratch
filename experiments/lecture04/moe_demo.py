"""
Lecture 04: Mixture of Experts (MoE) 完整实现
从零实现MoE的各个组件

主要内容:
1. Basic Router (Top-k Selection)
2. Expert Network
3. Load Balancing Loss
4. Shared Expert (DeepSeek-V3 style)
5. Expert Capacity & Auxiliary Losses
6. 完整MoE Layer

支持设备: CUDA / MPS (Apple Silicon) / CPU
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from collections import defaultdict

from utils.device_utils import DeviceManager, get_device


class Expert(nn.Module):
    """
    单个Expert网络 - 标准FFN结构

    结构: Linear → Activation → Linear
    """

    def __init__(self, d_model: int, d_ff: int, activation: str = 'gelu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model] 或 [num_tokens, d_model]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class TopKRouter(nn.Module):
    """
    Top-K Router - MoE的核心组件

    功能: 为每个token选择top-k个experts

    创新历史:
    - Switch Transformer: k=1（极端稀疏）
    - GShard: k=2（平衡性能与稀疏性）
    - DeepSeek-V3: k=8（更多expert参与）
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        k: int = 2,
        noise_std: float = 0.0,  # 训练时添加噪声
        temperature: float = 1.0
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.noise_std = noise_std
        self.temperature = temperature

        # Router网络: 简单的线性映射
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            gates: [batch_size, seq_len, k] - 选中expert的权重
            indices: [batch_size, seq_len, k] - 选中expert的索引
            router_logits: [batch_size, seq_len, num_experts] - 完整logits（用于auxiliary loss）
        """
        # Step 1: 计算router logits
        router_logits = self.gate(x)  # [B, L, num_experts]

        # Step 2: 训练时添加噪声（可选，提升exploration）
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise

        # Step 3: 温度缩放
        router_logits = router_logits / self.temperature

        # Step 4: Softmax得到概率
        router_probs = F.softmax(router_logits, dim=-1)  # [B, L, num_experts]

        # Step 5: Top-K选择
        top_k_gates, top_k_indices = torch.topk(router_probs, self.k, dim=-1)

        # Step 6: 重新归一化选中的gates（使它们和为1）
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=-1, keepdim=True) + 1e-6)

        return top_k_gates, top_k_indices, router_logits


class LoadBalancingLoss:
    """
    负载均衡损失 - 确保experts被均匀使用

    目标: 避免"rich-get-richer"问题（少数expert被过度使用）

    计算方法:
    L_balance = α * num_experts * Σ(f_i * P_i)

    其中:
    - f_i: expert i被选中的token比例
    - P_i: router分配给expert i的平均概率
    - α: 损失权重（通常0.01）
    """

    @staticmethod
    def compute(
        router_logits: torch.Tensor,  # [B, L, num_experts]
        top_k_indices: torch.Tensor,   # [B, L, k]
        alpha: float = 0.01
    ) -> torch.Tensor:
        """
        计算负载均衡损失

        Args:
            router_logits: router的原始输出
            top_k_indices: 选中的expert索引
            alpha: 损失权重

        Returns:
            load_balance_loss: 标量
        """
        num_experts = router_logits.size(-1)
        batch_size, seq_len, k = top_k_indices.shape

        # 计算每个expert的平均路由概率 P_i
        router_probs = F.softmax(router_logits, dim=-1)  # [B, L, num_experts]
        mean_probs = router_probs.mean(dim=[0, 1])  # [num_experts]

        # 计算每个expert被选中的比例 f_i
        # 将top_k_indices展平并统计
        flat_indices = top_k_indices.reshape(-1)  # [B*L*k]
        expert_counts = torch.bincount(flat_indices, minlength=num_experts).float()
        total_tokens = batch_size * seq_len * k
        freq = expert_counts / total_tokens  # [num_experts]

        # 负载均衡损失
        loss = alpha * num_experts * (freq * mean_probs).sum()

        return loss


class RouterZLoss:
    """
    Router Z-Loss - 稳定训练的辅助损失

    目的: 防止router logits过大导致softmax饱和

    公式: L_z = (1/B) * Σ(log(Σexp(logits)))²

    来自ST-MoE论文
    """

    @staticmethod
    def compute(router_logits: torch.Tensor, beta: float = 0.001) -> torch.Tensor:
        """
        计算Z-loss

        Args:
            router_logits: [B, L, num_experts]
            beta: 损失权重

        Returns:
            z_loss: 标量
        """
        # log(sum(exp(logits))) 然后平方
        log_sum_exp = torch.logsumexp(router_logits, dim=-1)  # [B, L]
        z_loss = beta * (log_sum_exp ** 2).mean()

        return z_loss


class MoELayer(nn.Module):
    """
    完整的MoE Layer

    结构:
    1. Router选择top-k experts
    2. 分发tokens到对应experts
    3. 加权合并expert输出
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        k: int = 2,
        dropout: float = 0.0,
        use_load_balance_loss: bool = True,
        use_z_loss: bool = True,
        load_balance_alpha: float = 0.01,
        z_loss_beta: float = 0.001
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.k = k
        self.use_load_balance_loss = use_load_balance_loss
        self.use_z_loss = use_z_loss
        self.load_balance_alpha = load_balance_alpha
        self.z_loss_beta = z_loss_beta

        # Router
        self.router = TopKRouter(d_model, num_experts, k)

        # Experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(num_experts)
        ])

        self.dropout = nn.Dropout(dropout)

        # 用于记录统计信息
        self.expert_usage = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播

        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            output: [batch_size, seq_len, d_model]
            aux_losses: 字典，包含各种辅助损失
        """
        batch_size, seq_len, d_model = x.shape
        aux_losses = {}

        # Step 1: Router决定每个token去哪些experts
        gates, indices, router_logits = self.router(x)
        # gates: [B, L, k]
        # indices: [B, L, k]

        # Step 2: 计算辅助损失
        if self.training:
            if self.use_load_balance_loss:
                lb_loss = LoadBalancingLoss.compute(
                    router_logits, indices, self.load_balance_alpha
                )
                aux_losses['load_balance_loss'] = lb_loss

            if self.use_z_loss:
                z_loss = RouterZLoss.compute(router_logits, self.z_loss_beta)
                aux_losses['z_loss'] = z_loss

        # Step 3: 分发tokens到experts并合并结果
        # 简化实现：逐token处理（生产环境应使用更高效的batch处理）
        output = torch.zeros_like(x)

        # 统计expert使用情况
        expert_counts = defaultdict(int)

        for b in range(batch_size):
            for s in range(seq_len):
                token = x[b, s]  # [d_model]
                token_output = torch.zeros(d_model, device=x.device, dtype=x.dtype)

                for i in range(self.k):
                    expert_idx = indices[b, s, i].item()
                    gate_value = gates[b, s, i]

                    expert_output = self.experts[expert_idx](token.unsqueeze(0))
                    token_output = token_output + gate_value * expert_output.squeeze(0)

                    expert_counts[expert_idx] += 1

                output[b, s] = token_output

        self.expert_usage = dict(expert_counts)
        output = self.dropout(output)

        return output, aux_losses


class SharedExpertMoELayer(nn.Module):
    """
    带Shared Expert的MoE Layer (DeepSeek-V3 style)

    创新点:
    1. 1个Shared Expert: 处理所有token，学习通用特征
    2. N个Sparse Experts: Top-k选择，学习专业特征

    优势:
    - 避免cold start问题
    - 缓解"rich-get-richer"
    - 训练更稳定
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,  # sparse experts数量
        k: int = 2,
        shared_expert_ratio: float = 0.25,  # shared expert相对大小
        dropout: float = 0.0,
        use_load_balance_loss: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.k = k

        # Shared Expert - 容量可以不同于sparse experts
        shared_d_ff = int(d_ff * shared_expert_ratio)
        self.shared_expert = Expert(d_model, shared_d_ff)

        # Router (只路由sparse experts)
        self.router = TopKRouter(d_model, num_experts, k)

        # Sparse Experts
        self.sparse_experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(num_experts)
        ])

        self.dropout = nn.Dropout(dropout)
        self.use_load_balance_loss = use_load_balance_loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播

        output = shared_expert(x) + Σ(gate_i * sparse_expert_i(x))
        """
        batch_size, seq_len, d_model = x.shape
        aux_losses = {}

        # Step 1: Shared Expert (所有token)
        shared_output = self.shared_expert(x)  # [B, L, d_model]

        # Step 2: Router选择sparse experts
        gates, indices, router_logits = self.router(x)

        # Step 3: 计算辅助损失
        if self.training and self.use_load_balance_loss:
            lb_loss = LoadBalancingLoss.compute(
                router_logits, indices, alpha=0.01
            )
            aux_losses['load_balance_loss'] = lb_loss

        # Step 4: Sparse Expert输出
        sparse_output = torch.zeros_like(x)

        for b in range(batch_size):
            for s in range(seq_len):
                token = x[b, s]

                for i in range(self.k):
                    expert_idx = indices[b, s, i].item()
                    gate_value = gates[b, s, i]

                    expert_out = self.sparse_experts[expert_idx](token.unsqueeze(0))
                    sparse_output[b, s] += gate_value * expert_out.squeeze(0)

        # Step 5: 合并Shared + Sparse
        output = shared_output + sparse_output
        output = self.dropout(output)

        return output, aux_losses


class MoETransformerBlock(nn.Module):
    """
    使用MoE的Transformer Block

    与标准Transformer Block的区别:
    - FFN层替换为MoE层
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        d_ff: int = 3072,
        num_experts: int = 8,
        k: int = 2,
        dropout: float = 0.1,
        use_shared_expert: bool = True
    ):
        super().__init__()

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # MoE Layer (替代普通FFN)
        if use_shared_expert:
            self.moe = SharedExpertMoELayer(
                d_model, d_ff, num_experts, k, dropout=dropout
            )
        else:
            self.moe = MoELayer(
                d_model, d_ff, num_experts, k, dropout=dropout
            )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播

        Returns:
            output: [B, L, d_model]
            aux_losses: MoE辅助损失
        """
        # Pre-Norm Attention
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = residual + self.dropout(attn_output)

        # Pre-Norm MoE
        residual = x
        x = self.norm2(x)
        moe_output, aux_losses = self.moe(x)
        x = residual + moe_output

        return x, aux_losses


def demonstrate_basic_moe():
    """演示基础MoE"""

    dm = DeviceManager()

    print("\n" + "=" * 70)
    print("🔀 Basic MoE Demo")
    print("=" * 70)

    batch_size = 2
    seq_len = 8
    d_model = 64
    d_ff = 256
    num_experts = 4
    k = 2

    # 创建MoE层
    moe = MoELayer(d_model, d_ff, num_experts, k).to(dm.device)

    # 输入
    x = torch.randn(batch_size, seq_len, d_model, device=dm.device)

    # 前向传播
    moe.train()
    output, aux_losses = moe(x)

    print(f"\n1. MoE Layer Configuration:")
    print(f"   - d_model: {d_model}")
    print(f"   - d_ff: {d_ff}")
    print(f"   - num_experts: {num_experts}")
    print(f"   - k (top-k): {k}")

    print(f"\n2. Input/Output shapes:")
    print(f"   - Input: {x.shape}")
    print(f"   - Output: {output.shape}")

    print(f"\n3. Auxiliary Losses:")
    for name, loss in aux_losses.items():
        print(f"   - {name}: {loss.item():.6f}")

    print(f"\n4. Expert Usage (tokens routed to each expert):")
    for expert_id, count in sorted(moe.expert_usage.items()):
        print(f"   - Expert {expert_id}: {count} tokens ({count/(batch_size*seq_len*k)*100:.1f}%)")


def demonstrate_shared_expert():
    """演示Shared Expert MoE"""

    dm = DeviceManager()

    print("\n" + "=" * 70)
    print("🌟 Shared Expert MoE Demo (DeepSeek-V3 Style)")
    print("=" * 70)

    batch_size = 2
    seq_len = 8
    d_model = 64
    d_ff = 256
    num_experts = 4
    k = 2

    # 创建带Shared Expert的MoE
    shared_moe = SharedExpertMoELayer(
        d_model, d_ff, num_experts, k,
        shared_expert_ratio=0.25
    ).to(dm.device)

    x = torch.randn(batch_size, seq_len, d_model, device=dm.device)

    shared_moe.train()
    output, aux_losses = shared_moe(x)

    print(f"\n1. Shared Expert MoE Configuration:")
    print(f"   - Shared Expert: Always active (d_ff={int(d_ff*0.25)})")
    print(f"   - Sparse Experts: {num_experts} × (d_ff={d_ff})")
    print(f"   - Top-k: {k}")

    print(f"\n2. Key Benefits:")
    print("""
    ✅ 避免Cold Start: Shared Expert始终提供基础输出
    ✅ 缓解Rich-get-richer: 所有token都经过Shared Expert
    ✅ 训练稳定: 即使routing不好也有保底输出
    ✅ 分工明确: Shared处理通用特征，Sparse处理专业特征
    """)

    # 参数对比
    total_params = sum(p.numel() for p in shared_moe.parameters())
    shared_params = sum(p.numel() for p in shared_moe.shared_expert.parameters())
    sparse_params = sum(p.numel() for p in shared_moe.sparse_experts.parameters())

    print(f"3. Parameter Distribution:")
    print(f"   - Shared Expert: {shared_params:,} ({shared_params/total_params*100:.1f}%)")
    print(f"   - Sparse Experts: {sparse_params:,} ({sparse_params/total_params*100:.1f}%)")
    print(f"   - Total: {total_params:,}")


def demonstrate_load_balancing():
    """演示负载均衡"""

    dm = DeviceManager()

    print("\n" + "=" * 70)
    print("⚖️ Load Balancing Demo")
    print("=" * 70)

    print("\n1. 问题: Rich-get-richer（马太效应）")
    print("""
    - 初始化时，某些experts表现略好
    - 这些experts被更多选中
    - 获得更多训练机会
    - 变得更好，被选中更多
    - 最终：少数expert处理大部分tokens
    """)

    # 模拟不均衡的routing
    batch_size, seq_len, num_experts = 4, 16, 8
    k = 2

    # 偏向expert 0和1
    biased_logits = torch.randn(batch_size, seq_len, num_experts, device=dm.device)
    biased_logits[:, :, 0] += 2.0  # Expert 0偏好
    biased_logits[:, :, 1] += 1.5  # Expert 1偏好

    router_probs = F.softmax(biased_logits, dim=-1)
    _, top_k_indices = torch.topk(router_probs, k, dim=-1)

    print("\n2. 不均衡的Expert使用 (模拟):")
    flat_indices = top_k_indices.reshape(-1)
    for i in range(num_experts):
        count = (flat_indices == i).sum().item()
        bar = "█" * int(count / 2)
        print(f"   Expert {i}: {count:3d} tokens {bar}")

    # 计算Load Balancing Loss
    lb_loss = LoadBalancingLoss.compute(biased_logits, top_k_indices, alpha=0.01)
    print(f"\n3. Load Balancing Loss: {lb_loss.item():.6f}")
    print("   (较大值表示不均衡)")

    # 均衡的情况
    balanced_logits = torch.randn(batch_size, seq_len, num_experts, device=dm.device) * 0.1
    router_probs = F.softmax(balanced_logits, dim=-1)
    _, balanced_indices = torch.topk(router_probs, k, dim=-1)

    print("\n4. 均衡的Expert使用 (模拟):")
    flat_indices = balanced_indices.reshape(-1)
    for i in range(num_experts):
        count = (flat_indices == i).sum().item()
        bar = "█" * int(count / 2)
        print(f"   Expert {i}: {count:3d} tokens {bar}")

    lb_loss_balanced = LoadBalancingLoss.compute(balanced_logits, balanced_indices, alpha=0.01)
    print(f"\n5. Balanced Load Balancing Loss: {lb_loss_balanced.item():.6f}")
    print("   (较小值表示均衡)")


def demonstrate_moe_transformer():
    """演示MoE Transformer Block"""

    dm = DeviceManager()

    print("\n" + "=" * 70)
    print("🏗️ MoE Transformer Block Demo")
    print("=" * 70)

    batch_size = 2
    seq_len = 16
    d_model = 128
    num_heads = 4
    d_ff = 512
    num_experts = 4
    k = 2

    # 创建MoE Transformer Block
    moe_block = MoETransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_experts=num_experts,
        k=k,
        use_shared_expert=True
    ).to(dm.device)

    x = torch.randn(batch_size, seq_len, d_model, device=dm.device)

    moe_block.train()
    output, aux_losses = moe_block(x)

    print(f"\n1. MoE Transformer Block Configuration:")
    print(f"   - d_model: {d_model}")
    print(f"   - num_heads: {num_heads}")
    print(f"   - d_ff: {d_ff}")
    print(f"   - num_experts: {num_experts}")
    print(f"   - k: {k}")
    print(f"   - use_shared_expert: True")

    print(f"\n2. Input/Output:")
    print(f"   - Input: {x.shape}")
    print(f"   - Output: {output.shape}")

    # 参数统计
    total_params = sum(p.numel() for p in moe_block.parameters())
    attn_params = sum(p.numel() for p in moe_block.attention.parameters())
    moe_params = sum(p.numel() for p in moe_block.moe.parameters())

    print(f"\n3. Parameters:")
    print(f"   - Total: {total_params:,}")
    print(f"   - Attention: {attn_params:,}")
    print(f"   - MoE: {moe_params:,}")

    # vs Dense对比
    dense_ffn_params = d_model * d_ff * 2 + d_model + d_ff  # 两个Linear层+bias
    print(f"\n4. MoE vs Dense FFN:")
    print(f"   - Dense FFN params: {dense_ffn_params:,}")
    print(f"   - MoE params: {moe_params:,}")
    print(f"   - Ratio: {moe_params/dense_ffn_params:.1f}x more params")
    print(f"   - But active params per token: ~{moe_params/(num_experts/k):,.0f}")


def demonstrate_moe_efficiency():
    """演示MoE效率分析"""

    print("\n" + "=" * 70)
    print("📊 MoE Efficiency Analysis")
    print("=" * 70)

    print("\n1. MoE核心优势: 参数与计算解耦")
    print("""
    Dense Model:
    - 参数数量 = 计算量
    - 更多参数 → 更多计算

    MoE Model:
    - 参数数量 >> 激活参数数量
    - 每个token只激活top-k experts
    - 更多参数，但计算量与k相关
    """)

    # 具体数字对比
    d_model = 768
    d_ff = 3072
    num_experts = 64
    k = 2

    dense_params = 2 * d_model * d_ff  # 忽略bias
    moe_params = num_experts * 2 * d_model * d_ff

    dense_flops = 2 * dense_params
    moe_flops = k * 2 * 2 * d_model * d_ff  # k个experts的forward

    print(f"\n2. 参数对比 (d={d_model}, ff={d_ff}, E={num_experts}, k={k}):")
    print(f"   Dense FFN: {dense_params:,} params")
    print(f"   MoE (64 experts): {moe_params:,} params")
    print(f"   Ratio: {moe_params/dense_params:.0f}x more params")

    print(f"\n3. 计算量对比 (per token):")
    print(f"   Dense FFN: {dense_flops:,} FLOPs")
    print(f"   MoE (k=2): {moe_flops:,} FLOPs")
    print(f"   Computation ratio: {moe_flops/dense_flops:.1f}x")

    print(f"\n4. 模型规模对比:")

    models = [
        ("GPT-3", 175, 175, "Dense"),
        ("Switch-C", 1600, 1.6, "MoE k=1"),
        ("GPT-4 (rumored)", 1800, 220, "MoE k=2"),
        ("DeepSeek-V3", 671, 37, "MoE k=8"),
    ]

    print(f"   {'Model':<18} {'Total Params':<15} {'Active Params':<15} {'Type':<12}")
    print(f"   {'-'*60}")
    for name, total, active, type_ in models:
        print(f"   {name:<18} {total:>10}B    {active:>10}B    {type_:<12}")

    print("\n5. 为什么MoE更高效？")
    print("""
    ✅ 条件计算: 只激活需要的experts
    ✅ 专家专业化: 每个expert学习不同特征
    ✅ 参数利用率: 相同计算量下容纳更多知识
    ✅ 训练效率: 可以更大batch size
    """)


def run_all_demos():
    """运行所有演示"""

    # 1. 基础MoE
    demonstrate_basic_moe()

    # 2. Shared Expert
    demonstrate_shared_expert()

    # 3. 负载均衡
    demonstrate_load_balancing()

    # 4. MoE Transformer
    demonstrate_moe_transformer()

    # 5. 效率分析
    demonstrate_moe_efficiency()


if __name__ == '__main__':
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                                                                    ║")
    print("║           CS336 Lecture 04 - Mixture of Experts (MoE)             ║")
    print("║                                                                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    run_all_demos()

    print("\n" + "=" * 70)
    print("✅ MoE Demo完成！")
    print("=" * 70)
    print("\n💡 Key Takeaways:")
    print("  - MoE allows scaling parameters without proportional compute increase")
    print("  - Load balancing is crucial for effective training")
    print("  - Shared Expert improves stability (DeepSeek-V3)")
    print("  - Auxiliary losses (Load Balance, Z-Loss) are essential")
    print("\n")
