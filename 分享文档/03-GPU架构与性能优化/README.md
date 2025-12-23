# GPU架构与性能优化

> **一句话摘要**: 从GPU内存层次到Roofline模型，理解深度学习性能优化的硬件基础，掌握FlashAttention等关键优化技术的原理。

## 核心概念

### 关键术语
| 术语 | 定义 | 重要性 |
|------|------|--------|
| SM (Streaming Multiprocessor) | GPU的基本计算单元，包含多个CUDA核心 | 理解GPU并行架构 |
| HBM (High Bandwidth Memory) | 高带宽显存，GPU的主要存储 | 决定数据传输瓶颈 |
| Shared Memory | 片上高速缓存，SM内线程共享 | 优化的关键资源 |
| Arithmetic Intensity (AI) | 计算量与内存访问的比值 (FLOP/Byte) | 判断瓶颈的核心指标 |
| Roofline Model | 分析计算 vs 内存瓶颈的可视化模型 | 性能优化的理论框架 |

### 概念图谱
```
GPU性能优化
├── 硬件架构
│   ├── 计算单元 (SM, CUDA Core, Tensor Core)
│   ├── 内存层次 (Register → SRAM → HBM)
│   └── 线程模型 (Thread → Warp → Block → Grid)
├── 性能分析
│   ├── Roofline Model
│   ├── Arithmetic Intensity
│   └── Memory-Bound vs Compute-Bound
└── 优化技术
    ├── Tiling (空间优化)
    ├── Fusion (时间优化)
    └── FlashAttention (综合应用)
```

## 技术深度

### 1. GPU内存层次结构

**6层内存金字塔**:
```
                 ┌─────────────────┐
                 │    Register     │  ← 1 cycle, KB级
                 │   (~256KB/SM)   │
                 └────────┬────────┘
                          │
                 ┌────────┴────────┐
                 │  Shared Memory  │  ← 20-30 cycles, 164KB/SM
                 │    / L1 Cache   │
                 └────────┬────────┘
                          │
                 ┌────────┴────────┐
                 │     L2 Cache    │  ← 200 cycles, 40MB
                 └────────┬────────┘
                          │
                 ┌────────┴────────┐
                 │       HBM       │  ← 380 cycles, 40-80GB
                 │   (2 TB/s)      │
                 └────────┬────────┘
                          │
                 ┌────────┴────────┐
                 │   Host Memory   │  ← 10,000 cycles
                 │   (PCIe/NVLink) │
                 └────────┬────────┘
                          │
                 ┌────────┴────────┐
                 │      SSD        │  ← 100,000+ cycles
                 └─────────────────┘
```

**A100关键参数**:
| 资源 | 容量 | 带宽 | 延迟 |
|------|------|------|------|
| Register | 256KB/SM | ~19.5 TB/s | 1 cycle |
| Shared Memory | 164KB/SM | ~19.5 TB/s | 20-30 cycles |
| L2 Cache | 40MB | ~5 TB/s | 200 cycles |
| HBM | 40-80GB | 1.5-2 TB/s | 380 cycles |

**核心洞察**:
```
SRAM vs HBM 速度差: 20x!
- SRAM (Shared Memory): 20 cycles
- HBM: 400 cycles

这就是为什么FlashAttention有效的根本原因！
```

### 2. Roofline Model

**核心公式**:
$$
\text{Performance} = \min(\text{Peak FLOPS}, \text{Bandwidth} \times \text{AI})
$$

其中 AI (Arithmetic Intensity) = FLOP / Bytes

**A100 Roofline**:
```
                    │
    312 TFLOP/s ────┼───────────────────────────  ← Compute Roof
                    │                        /
    Performance     │                      /
    (TFLOP/s)       │                    /
                    │                  /
                    │                /  ← Memory Roof (斜率 = 带宽)
                    │              /
                    │            /
                    │          /
                    │        /
                    │      /
                    │    /
                    │  /
                    │/
                    └──────────────────────────────
                    0              208           AI (FLOP/Byte)
                              ↑
                        Critical Point
```

**临界点 (Critical AI)**:
```python
critical_ai = peak_flops / bandwidth
            = 312 TFLOP/s / 1.5 TB/s
            = 208 FLOP/Byte

判断规则:
- AI < 208: Memory-Bound (数据搬运是瓶颈)
- AI > 208: Compute-Bound (计算是瓶颈)
```

### 3. 常见操作的AI分析

```python
# 矩阵乘法 C = A @ B
# A: [M, K], B: [K, N], C: [M, N]

flops = 2 * M * K * N  # 乘加各算1次
bytes = (M*K + K*N + M*N) * sizeof(float)

# 方阵情况 M=K=N=n
AI = 2*n³ / (3*n² * 4)
   = n / 6

# 示例:
# n=1024: AI = 170 (Memory-Bound)
# n=2048: AI = 341 (Compute-Bound!)
```

**标准Attention的AI**:
```python
# Q, K, V: [B, H, S, D]
# Attention = softmax(Q @ K.T / sqrt(d)) @ V

# Step 1: Q @ K.T
flops_qk = 2 * B * H * S * S * D
bytes_qk = (2 * B * H * S * D + B * H * S * S) * 2  # FP16

# Step 2: Softmax (省略)

# Step 3: Scores @ V
flops_sv = 2 * B * H * S * S * D

# 总体AI (近似):
AI_attention ≈ 4 * S * D / (4 * S + 2 * D)
             ≈ D (当 S >> D)

# 典型配置 D=64:
AI_attention ≈ 64 << 208  # 严重 Memory-Bound!
```

### 4. FlashAttention原理

**问题: N² 存储瓶颈**
```
标准Attention流程:
1. S = Q @ K.T  → 写入HBM (N² 大小)
2. P = softmax(S)  → 读取S, 写入P到HBM
3. O = P @ V  → 读取P, 计算输出

HBM访问: ~4N² 次 (读写S和P各两次)

当 N=4096, FP16:
内存 = 4 × 4096² × 2 = 134MB (单个attention)
```

**FlashAttention解决方案: Tiling + Fusion**
```python
# 核心思想: 分块计算，数据留在SRAM

def flash_attention(Q, K, V, block_size=128):
    N, d = Q.shape
    O = zeros(N, d)
    L = zeros(N)  # log-sum-exp

    # 外循环: K, V 分块
    for j in range(0, N, block_size):
        Kj = K[j:j+block_size]  # 加载到SRAM
        Vj = V[j:j+block_size]  # 加载到SRAM

        # 内循环: Q 分块
        for i in range(0, N, block_size):
            Qi = Q[i:i+block_size]  # 加载到SRAM

            # 在SRAM中完成所有计算!
            Sij = Qi @ Kj.T / sqrt(d)  # 在SRAM

            # Online Softmax
            m_new = max(L[i:i+block_size], Sij.max(dim=-1))
            P_scaled = exp(Sij - m_new)

            # 更新输出
            O[i:i+block_size] = rescale(O, L, m_new) + P_scaled @ Vj
            L[i:i+block_size] = m_new + log(sum(P_scaled))

    return O
```

**HBM访问量对比**:
```
标准Attention:
- 读: Q(Nd) + K(Nd) + V(Nd) + S(N²) + P(N²) = 3Nd + 2N²
- 写: S(N²) + P(N²) + O(Nd) = Nd + 2N²
- 总计: 4N² + 4Nd

FlashAttention:
- 每个block读Q, K, V, O各一次
- 总计: O(N² d / M) 其中M是SRAM大小

典型配置 (N=4096, d=64, M=128KB):
- 标准: 4144 MB
- Flash: 1072 MB (3.86x 减少!)
```

**AI的提升**:
```
标准Attention AI: ~33 FLOP/Byte
FlashAttention AI: ~4096 FLOP/Byte (考虑SRAM数据复用)

从 Memory-Bound → Compute-Bound!
```

### 5. Tiling最优块大小

**约束条件**:
```python
# 块大小 B_r × B_c
# 需要在SRAM中存储: Qi, Kj, Vj, Sij, Oi

sram_usage = (B_r * d) + (B_c * d) + (B_c * d) + (B_r * B_c) + (B_r * d)
           = B_r * d * 2 + B_c * d * 2 + B_r * B_c
           ≤ SRAM_size

# A100: SRAM = 164KB = 164 * 1024 / 2 = 84K (FP16元素)
```

**最优块大小分析**:
```python
# 假设 d = 64, 正方形块 B_r = B_c = B

sram = 2 * B * 64 + 2 * B * 64 + B²
     = 256B + B²
     ≤ 84000

# 解: B ≤ 168
# 实践选择: B = 128 (2的幂次, 对齐友好)

# 验证:
sram = 256 * 128 + 128² = 32768 + 16384 = 49152 (58% 利用率)
```

### 6. Kernel Fusion (算子融合)

**问题: 多次HBM访问**
```python
# 未融合的操作
Y = dropout(softmax(Q @ K.T / sqrt(d))) @ V

# 展开:
S = Q @ K.T           # 写S到HBM
S = S / sqrt(d)       # 读S, 写S
P = softmax(S)        # 读S, 写P
P = dropout(P)        # 读P, 写P
O = P @ V             # 读P, 读V, 写O

# 总HBM访问: 5次读 + 5次写 = 10N² bytes
```

**融合后**:
```python
# 融合成一个kernel
@triton.jit
def fused_attention(Q, K, V, O):
    # 在SRAM中完成所有操作
    S = Q @ K.T
    S = S / sqrt(d)      # 在寄存器
    P = softmax(S)       # 在SRAM
    P = dropout(P)       # 在SRAM
    O = P @ V            # 在SRAM

# 总HBM访问: 3次读(Q,K,V) + 1次写(O) = 4Nd bytes
```

## 实践代码

### 简化的FlashAttention实现

```python
import torch
import torch.nn.functional as F
import math

def simple_flash_attention(Q, K, V, block_size=128):
    """
    简化的FlashAttention实现 (用于理解原理)
    实际应用请使用 flash_attn 库
    """
    batch, heads, seq_len, d = Q.shape
    O = torch.zeros_like(Q)
    L = torch.zeros(batch, heads, seq_len, 1, device=Q.device)  # log-sum-exp
    M = torch.full((batch, heads, seq_len, 1), float('-inf'), device=Q.device)  # max

    scale = 1.0 / math.sqrt(d)

    # 分块计算
    for j in range(0, seq_len, block_size):
        j_end = min(j + block_size, seq_len)
        Kj = K[:, :, j:j_end, :]  # [B, H, Bc, D]
        Vj = V[:, :, j:j_end, :]

        for i in range(0, seq_len, block_size):
            i_end = min(i + block_size, seq_len)
            Qi = Q[:, :, i:i_end, :]  # [B, H, Br, D]

            # 计算注意力分数
            Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) * scale  # [B, H, Br, Bc]

            # Online Softmax: 更新max
            M_old = M[:, :, i:i_end, :]
            M_new = torch.maximum(M_old, Sij.max(dim=-1, keepdim=True)[0])

            # 计算exp并更新
            exp_old = torch.exp(M_old - M_new)
            exp_new = torch.exp(Sij - M_new)

            # 更新L (用于归一化)
            L_old = L[:, :, i:i_end, :]
            L_new = exp_old * L_old + exp_new.sum(dim=-1, keepdim=True)

            # 更新输出
            O[:, :, i:i_end, :] = (
                exp_old * L_old / L_new * O[:, :, i:i_end, :] +
                torch.matmul(exp_new / L_new, Vj)
            )

            M[:, :, i:i_end, :] = M_new
            L[:, :, i:i_end, :] = L_new

    return O


# 验证正确性
def standard_attention(Q, K, V):
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = F.softmax(S, dim=-1)
    return torch.matmul(P, V)


# 测试
B, H, S, D = 2, 8, 256, 64
Q = torch.randn(B, H, S, D)
K = torch.randn(B, H, S, D)
V = torch.randn(B, H, S, D)

out_standard = standard_attention(Q, K, V)
out_flash = simple_flash_attention(Q, K, V, block_size=64)

print(f"Max diff: {(out_standard - out_flash).abs().max().item():.6f}")
# 应该非常小 (< 1e-5)
```

## 关键洞察

### 核心收获

1. **内存层次决定一切**: SRAM比HBM快20倍，这是所有优化的基础

2. **AI是关键指标**: 知道你的操作是Memory-Bound还是Compute-Bound

3. **FlashAttention = Tiling + Fusion**:
   - Tiling: 数据分块，留在SRAM
   - Fusion: 算子合并，减少HBM访问

4. **"重算比存储更快"**: FlashAttention的反直觉核心

5. **Roofline告诉你天花板在哪**: 优化前先分析，别盲目优化

### 常见误区

| 误区 | 正确理解 |
|------|----------|
| GPU利用率高=性能好 | 可能是Memory-Bound,计算在等数据 |
| FlashAttention是空间换时间 | 实际是时间换空间 (重算) |
| 更大的矩阵总是更慢 | 更大的矩阵可能AI更高,更高效 |
| Fusion只是减少kernel launch | 主要减少HBM访问 |

## 延伸阅读

### 推荐论文
- [FlashAttention](https://arxiv.org/abs/2205.14135) - IO感知的精确注意力
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) - 更好的工作分配

### 相关专题
- [Transformer架构精讲](../01-Transformer架构精讲/) - Attention的原理
- [大模型内存分析](../04-大模型内存分析/) - 内存计算实践

---

## 内容来源

本文档内容整理自以下来源：
- [来源: 深度讨论/Lecture05-Part3-计算与带宽分析完整总结(Q13-Q18).md]
- [来源: 深度讨论/Lecture05-Part2-GPU内存层次与优化完整总结(Q7-Q10).md]
- [来源: 深度讨论/内存层次结构深度讨论.md]

---

**作者**: peixingxin + Claude Code
**创建日期**: 2025-12-17
**最后更新**: 2025-12-17
