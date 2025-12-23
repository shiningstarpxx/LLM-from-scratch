# GPU架构与性能优化 - 60分钟深度版

---

## 封面

### GPU架构与性能优化深度解析
**副标题**: 从内存层次到FlashAttention，深入理解GPU性能优化

**分享人**: [你的名字]
**时长**: 60分钟 (含10min Q&A)

---

## 议程

```
Part 1: 背景与动机              (5 min)
Part 2: GPU内存层次深度解析     (15 min)
Part 3: Roofline模型与性能分析  (12 min)
Part 4: FlashAttention原理与实现 (15 min)
Part 5: 实战优化案例            (8 min)
Part 6: 总结与讨论              (5 min)
Q&A                             (10 min)
```

---

# Part 1: 背景与动机

---

## 1.1 GPU算力的"虚假繁荣"

```
A100 GPU 规格:
- 理论算力: 312 TFLOP/s (FP16)
- 显存带宽: 2 TB/s (HBM2e)
- 显存容量: 80GB

实际利用率:
- 典型深度学习workload: 10-30%
- 即使优化后: 40-60%

问题: 300+ TFLOP/s 的算力，大部分时间在"空转"
```

---

## 1.2 内存墙问题

```
算力增长 vs 带宽增长:

┌────────────────────────────────────────┐
│ Year    算力增长      带宽增长         │
├────────────────────────────────────────┤
│ 2016    Pascal → Volta   +50%   +25%   │
│ 2018    Volta → Turing   +40%   +20%   │
│ 2020    Turing → Ampere  +100%  +50%   │
│ 2022    Ampere → Hopper  +200%  +50%   │
└────────────────────────────────────────┘

结果: 算力/带宽 比值不断增大
→ 越来越多操作变成 Memory-Bound
```

---

## 1.3 今天的学习目标

```
1. 深入理解GPU内存层次结构
2. 掌握Roofline模型进行性能分析
3. 理解FlashAttention的设计原理
4. 学会分析和优化Memory-Bound操作
```

---

# Part 2: GPU内存层次深度解析

---

## 2.1 完整内存层次

```
                ┌─────────────┐
                │  Registers  │ ← 每个线程私有
                │   256KB/SM  │   ~20 TB/s, 1 cycle
                └──────┬──────┘
                       │
                ┌──────┴──────┐
                │   L1/SRAM   │ ← SM内共享
                │  192KB/SM   │   ~19 TB/s, ~20 cycles
                │(Shared Mem) │
                └──────┬──────┘
                       │
                ┌──────┴──────┐
                │   L2 Cache  │ ← 全局共享
                │    40MB     │   ~5 TB/s, ~200 cycles
                └──────┬──────┘
                       │
                ┌──────┴──────┐
                │    HBM      │ ← 全局显存
                │    80GB     │   2 TB/s, ~400 cycles
                └─────────────┘
```

---

## 2.2 各层级详细参数 (A100)

| 层级 | 容量 | 带宽 | 延迟 | 特点 |
|------|------|------|------|------|
| Register | 256KB/SM | ~20 TB/s | 1 cycle | 编译器自动管理 |
| Shared Memory | 164KB/SM | ~19 TB/s | ~20 cycles | 程序员可控 |
| L1 Cache | 与Shared共享 | ~19 TB/s | ~20 cycles | 硬件自动管理 |
| L2 Cache | 40MB | ~5 TB/s | ~200 cycles | 全局共享 |
| HBM2e | 80GB | 2 TB/s | ~400 cycles | 主存储 |

---

## 2.3 SRAM vs HBM: 关键对比

```python
# 速度差距
HBM_bandwidth = 2e12      # 2 TB/s
SRAM_bandwidth = 19e12    # 19 TB/s

speedup = SRAM_bandwidth / HBM_bandwidth
print(f"SRAM比HBM快: {speedup:.1f}x")  # 约9.5倍

# 延迟差距
HBM_latency = 400         # cycles
SRAM_latency = 20         # cycles

latency_ratio = HBM_latency / SRAM_latency
print(f"HBM延迟是SRAM的: {latency_ratio:.0f}x")  # 20倍
```

**关键洞察**: 同样的数据，从SRAM读取比HBM快一个数量级

---

## 2.4 Shared Memory编程模型

```cuda
// CUDA Shared Memory示例
__global__ void matmul_shared(float* A, float* B, float* C, int N) {
    // 声明共享内存 - 每个block共享
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // 分块计算
    for (int t = 0; t < N / TILE_SIZE; t++) {
        // 协作加载到shared memory (从HBM)
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];

        __syncthreads();  // 等待所有线程加载完成

        // 在shared memory中计算 (快!)
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}
```

---

## 2.5 Bank Conflict

```
Shared Memory组织: 32个bank，每个4字节

理想访问 (无冲突):
Thread 0 → Bank 0
Thread 1 → Bank 1
Thread 2 → Bank 2
...
所有线程同时访问，并行执行

Bank Conflict (冲突):
Thread 0 → Bank 0
Thread 1 → Bank 0  ← 冲突!
Thread 2 → Bank 0  ← 冲突!
...
串行化执行，性能下降

解决方法:
1. Padding: __shared__ float data[32][33];  // 加一列
2. 调整访问模式
```

---

## 2.6 数据复用的重要性

```
矩阵乘法 C = A @ B, 维度 [M, K] × [K, N]

无Tiling:
- 每个C[i,j]需要读取A的第i行(K元素)和B的第j列(K元素)
- 总HBM访问: M*N*2K 次
- 数据复用: 0 (每次都从HBM读)

有Tiling:
- 把A和B分成小块加载到Shared Memory
- 小块内多次复用
- HBM访问减少: sqrt(K)倍

这就是为什么Tiling是GPU优化的基础!
```

---

# Part 3: Roofline模型与性能分析

---

## 3.1 Arithmetic Intensity (AI)

```
定义:
AI = FLOPs / Bytes

单位: FLOP/Byte

含义: 每访问1字节内存，执行多少次浮点运算

高AI: 计算密集型
低AI: 内存密集型
```

---

## 3.2 计算不同操作的AI

```python
def compute_ai_matmul(M, N, K, dtype_bytes=2):
    """矩阵乘法的AI"""
    # FLOPs: 2*M*N*K (乘法+加法)
    flops = 2 * M * N * K

    # Bytes: 读A(M*K) + 读B(K*N) + 写C(M*N)
    bytes_accessed = (M*K + K*N + M*N) * dtype_bytes

    return flops / bytes_accessed

# 小矩阵
print(f"[128,128]×[128,128]: AI = {compute_ai_matmul(128,128,128):.1f}")  # ~42

# 大矩阵
print(f"[4096,4096]×[4096,4096]: AI = {compute_ai_matmul(4096,4096,4096):.1f}")  # ~1365
```

**发现**: 矩阵越大，AI越高!

---

## 3.3 标准Attention的AI分析

```python
def attention_ai(batch, heads, seq_len, head_dim, dtype_bytes=2):
    """
    Attention: softmax(Q @ K.T) @ V
    """
    n = seq_len
    d = head_dim
    B = batch * heads

    # Step 1: S = Q @ K.T
    # FLOPs: 2 * B * n * d * n = 2Bn²d
    # Bytes: 读Q(Bnd) + 读K(Bnd) + 写S(Bn²)

    # Step 2: P = softmax(S)
    # FLOPs: ~5Bn² (exp, sum, div)
    # Bytes: 读S(Bn²) + 写P(Bn²)

    # Step 3: O = P @ V
    # FLOPs: 2 * B * n * n * d = 2Bn²d
    # Bytes: 读P(Bn²) + 读V(Bnd) + 写O(Bnd)

    total_flops = 4 * B * n * n * d + 5 * B * n * n
    total_bytes = (4 * B * n * d + 4 * B * n * n) * dtype_bytes

    return total_flops / total_bytes

# seq_len=2048, head_dim=128
ai = attention_ai(1, 32, 2048, 128)
print(f"Standard Attention AI: {ai:.1f}")  # ~30 FLOP/Byte
```

**AI只有30**, 远低于临界点!

---

## 3.4 Roofline图详解

```
Performance (TFLOP/s)
        │
   312 ─┼─────────────────────────────── Compute Roof
        │                             /
        │                           /
        │                         /
        │                       /    Memory Roof
        │                     /      (slope = bandwidth)
        │                   /
        │                 /
        │               /
        │             /
        │           /
        │         / ← 标准Attention (AI=30)
        │       /
        │     /
        │   /
        │ /
        └──────────────────────────────────────────
        0            208                          AI
                      ↑
              Critical Point
              = 312 TFLOP/s / 1.5 TB/s
```

---

## 3.5 性能预测

```python
def predict_performance(ai, peak_compute, peak_bandwidth):
    """
    Roofline模型性能预测
    """
    critical_ai = peak_compute / peak_bandwidth

    if ai < critical_ai:
        # Memory-bound: 性能受带宽限制
        performance = ai * peak_bandwidth
        bottleneck = "Memory"
    else:
        # Compute-bound: 性能受算力限制
        performance = peak_compute
        bottleneck = "Compute"

    efficiency = performance / peak_compute * 100
    return performance, bottleneck, efficiency

# A100参数
peak_compute = 312e12  # FLOP/s
peak_bandwidth = 1.5e12  # Byte/s

# 标准Attention
perf, bottleneck, eff = predict_performance(30, peak_compute, peak_bandwidth)
print(f"标准Attention: {perf/1e12:.1f} TFLOP/s, {bottleneck}-bound, {eff:.1f}%效率")
# 输出: 45 TFLOP/s, Memory-bound, 14.4%效率

# 大矩阵乘法
perf, bottleneck, eff = predict_performance(1000, peak_compute, peak_bandwidth)
print(f"大矩阵乘法: {perf/1e12:.1f} TFLOP/s, {bottleneck}-bound, {eff:.1f}%效率")
# 输出: 312 TFLOP/s, Compute-bound, 100%效率
```

---

## 3.6 如何提高AI

```
三大策略:

1. Tiling (分块)
   - 数据加载到SRAM后多次复用
   - 减少HBM访问次数

2. Kernel Fusion (算子融合)
   - 多个操作合并为一个kernel
   - 中间结果不写回HBM

3. Recomputation (重计算)
   - 不存储中间结果，需要时重新计算
   - 用计算换存储

FlashAttention同时使用了这三种策略!
```

---

# Part 4: FlashAttention原理与实现

---

## 4.1 标准Attention回顾

```python
def standard_attention(Q, K, V):
    """
    Q, K, V: [batch, heads, seq_len, head_dim]
    """
    # Step 1: QK^T → S [batch, heads, seq_len, seq_len]
    S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # 写S到HBM: O(n²)

    # Step 2: softmax → P
    P = torch.softmax(S, dim=-1)
    # 读S写P到HBM: O(n²)

    # Step 3: PV → O
    O = torch.matmul(P, V)
    # 读P: O(n²)

    return O

# 问题: S和P的HBM访问是O(n²)
# n=4096时: 4096² × 2bytes = 32MB × 来回3次 ≈ 100MB HBM访问
```

---

## 4.2 FlashAttention核心思想

```
目标: 永远不在HBM中存储n×n矩阵

方法:
1. Tiling: 将Q, K, V分成小块
2. 小块在SRAM中完成所有计算
3. Online Softmax: 边算边更新

关键洞察:
- n×n矩阵放不进SRAM (192KB < 32MB)
- 但 block×block 可以! (64×64×2=8KB)
```

---

## 4.3 Online Softmax算法

```python
def online_softmax(x_blocks):
    """
    标准softmax需要全局信息:
    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    Online版本: 边算边更新
    """
    m = -float('inf')  # 当前最大值
    l = 0.0            # 当前exp和

    for block in x_blocks:
        # 新块的最大值
        m_new = max(m, block.max())

        # 更新累积和 (需要rescale旧的结果)
        l = l * math.exp(m - m_new) + sum(math.exp(block - m_new))

        # 更新最大值
        m = m_new

    # 最终softmax = exp(x - m) / l
    return m, l
```

---

## 4.4 FlashAttention伪代码

```python
def flash_attention(Q, K, V, block_size=64):
    """
    FlashAttention核心算法
    """
    n = Q.shape[0]
    O = torch.zeros_like(Q)

    # 外层循环: K, V分块
    for j in range(0, n, block_size):
        Kj = K[j:j+block_size]  # 加载K块到SRAM
        Vj = V[j:j+block_size]  # 加载V块到SRAM

        # 内层循环: Q分块
        for i in range(0, n, block_size):
            Qi = Q[i:i+block_size]  # 加载Q块到SRAM

            # 以下全在SRAM中完成!

            # 1. 计算注意力分数
            Sij = Qi @ Kj.T / sqrt(d)  # [block, block]

            # 2. Online softmax更新
            m_new = max(m[i], Sij.max(dim=-1))
            P_scale = exp(m[i] - m_new)
            Pij = exp(Sij - m_new)

            # 3. 更新输出 (带rescale)
            O[i] = O[i] * P_scale + Pij @ Vj

            # 4. 更新统计量
            l[i] = l[i] * P_scale + Pij.sum(dim=-1)
            m[i] = m_new

    # 最终归一化
    O = O / l
    return O
```

---

## 4.5 内存访问分析

```
标准Attention HBM访问:
- 读Q, K: O(nd)
- 写S: O(n²)
- 读S写P: O(n²)
- 读P: O(n²)
- 写O: O(nd)
总计: O(n² + nd)

FlashAttention HBM访问:
- 读Q: O(nd) × (n/block) 轮 = O(n²d/block)
- 读K, V: O(nd) × (n/block) 轮 = O(n²d/block)
- 写O: O(nd)
总计: O(n²d/block + nd)

当 block >> d 时:
FlashAttention ≈ O(nd) vs 标准 O(n²)

实际提升: 2-4倍速度，O(n)内存
```

---

## 4.6 FlashAttention-2改进

```
FlashAttention-1 → FlashAttention-2:

1. 更好的并行化
   - FA1: 外层循环在K,V上，内层在Q上
   - FA2: 外层在Q上，可以更好利用GPU并行

2. 减少非矩阵乘法操作
   - 这些操作在Tensor Core上效率低

3. 更好的work partitioning
   - 在warps之间更均匀分配工作

结果: FA2比FA1快约2倍
```

---

## 4.7 实际使用

```python
# PyTorch 2.0+ 内置FlashAttention
import torch
import torch.nn.functional as F

# 方式1: 使用scaled_dot_product_attention
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,      # 启用FlashAttention
    enable_math=False,      # 禁用标准实现
    enable_mem_efficient=False
):
    output = F.scaled_dot_product_attention(Q, K, V)

# 方式2: 使用flash_attn库
from flash_attn import flash_attn_func
output = flash_attn_func(Q, K, V, causal=True)

# 自动选择最佳实现
# PyTorch会根据输入形状自动选择Flash/Memory-efficient/Math实现
```

---

# Part 5: 实战优化案例

---

## 5.1 案例: 优化Transformer前向传播

```python
# 优化前: 标准实现
class TransformerBlock(nn.Module):
    def forward(self, x):
        # Attention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        attn = torch.matmul(q, k.T) / sqrt(d)  # 写HBM
        attn = F.softmax(attn, dim=-1)          # 读写HBM
        attn = self.dropout(attn)               # 读写HBM
        out = torch.matmul(attn, v)             # 读HBM

        # FFN
        x = self.ffn1(x)    # 写HBM
        x = F.gelu(x)       # 读写HBM
        x = self.ffn2(x)    # 读写HBM

        return x

# 问题: 每个操作都独立访问HBM
```

---

## 5.2 优化策略应用

```python
# 优化后
class OptimizedTransformerBlock(nn.Module):
    def forward(self, x):
        # 优化1: 使用FlashAttention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        out = F.scaled_dot_product_attention(q, k, v)  # 内部tiling

        # 优化2: 融合FFN操作
        # 使用fused kernel: linear + gelu + linear
        x = self.fused_ffn(x)  # 一次kernel完成

        return x

# 进一步优化: 使用torch.compile
model = OptimizedTransformerBlock()
model = torch.compile(model)  # 自动kernel fusion
```

---

## 5.3 使用Profiler分析

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # 运行模型
    output = model(input)

# 分析结果
print(prof.key_averages().table(sort_by="cuda_time_total"))

# 关键指标:
# - CUDA time: GPU执行时间
# - Self CUDA Mem: GPU内存使用
# - SM Efficiency: 流多处理器效率
# - Memory Bandwidth: 实际带宽利用率
```

---

## 5.4 性能对比

```
LLaMA-7B, seq=2048, batch=1:

┌────────────────────────────────────────────────┐
│ 配置                    延迟      显存          │
├────────────────────────────────────────────────┤
│ 标准Attention           100ms    8.2GB         │
│ + FlashAttention        45ms     2.1GB         │
│ + torch.compile         38ms     2.1GB         │
│ + 量化 (INT8)           25ms     1.2GB         │
└────────────────────────────────────────────────┘

关键发现:
- FlashAttention: 2.2x加速, 4x内存减少
- 编译优化: 额外15%加速
- 量化: 额外35%加速
```

---

## 5.5 常见性能陷阱

```python
# 陷阱1: 不必要的同步
x = x.cuda()
y = y.cuda()
torch.cuda.synchronize()  # 不要在中间同步!
z = x + y

# 陷阱2: 频繁CPU-GPU传输
for i in range(1000):
    x = x.cpu()   # 慢!
    x = process(x)
    x = x.cuda()  # 慢!

# 陷阱3: 小kernel堆叠
x = F.relu(x)      # kernel 1
x = F.dropout(x)   # kernel 2
x = F.layer_norm(x) # kernel 3
# 应该融合成一个kernel

# 陷阱4: 忽略内存对齐
# tensor大小应该是64/128的倍数
x = torch.randn(127, 127)  # 差
x = torch.randn(128, 128)  # 好
```

---

# Part 6: 总结与讨论

---

## 6.1 核心概念回顾

| 概念 | 定义 | 重要性 |
|------|------|--------|
| 内存墙 | 算力增长 > 带宽增长 | 性能瓶颈根源 |
| SRAM | 快速片上存储 | 优化的关键资源 |
| Arithmetic Intensity | FLOPs/Bytes | 判断瓶颈类型 |
| Roofline Model | 性能上界分析 | 指导优化方向 |
| Tiling | 分块到SRAM | 提高数据复用 |
| Kernel Fusion | 合并操作 | 减少HBM访问 |

---

## 6.2 关键数字速记

```
A100 GPU:
┌────────────────────────────────┐
│ 峰值算力      312 TFLOP/s      │
│ HBM带宽       2 TB/s          │
│ SRAM带宽      19 TB/s         │
│ 临界AI        ~200 FLOP/Byte  │
│ SRAM容量      192KB/SM        │
│ HBM容量       80GB            │
└────────────────────────────────┘

FlashAttention:
┌────────────────────────────────┐
│ 内存复杂度    O(n) vs O(n²)   │
│ 速度提升      2-4x            │
│ 支持seq长度   可达100K+       │
└────────────────────────────────┘
```

---

## 6.3 优化检查清单

```
□ 使用FlashAttention (几乎必选)
□ 启用torch.compile (PyTorch 2.0+)
□ 检查tensor维度是否对齐 (64/128倍数)
□ 避免不必要的CPU-GPU传输
□ 使用混合精度训练 (FP16/BF16)
□ 用profiler找到热点
□ 考虑算子融合
□ 检查batch size是否合理
```

---

## 6.4 常见误区

| 误区 | 正确理解 |
|------|----------|
| GPU利用率高=快 | 可能在等内存 |
| 大batch总是快 | 可能超出内存 |
| FlashAttention牺牲精度 | 数学等价，精度相同 |
| 优化=改代码 | 先profile，后优化 |
| 一种优化通吃 | 需要组合多种策略 |

---

## 6.5 进阶学习资源

```
论文:
- FlashAttention 1: "FlashAttention: Fast and Memory-Efficient..."
- FlashAttention 2: "FlashAttention-2: Faster Attention with..."
- Roofline: "Roofline: An Insightful Visual Performance Model"

工具:
- NVIDIA Nsight Compute: 详细kernel分析
- PyTorch Profiler: 端到端性能分析
- torch.compile: 自动优化

实践:
- 复现FlashAttention简化版
- 用profiler分析自己的模型
- 尝试不同block size的影响
```

---

## Q&A

### Q1: FlashAttention对所有序列长度都有效吗？
**A**: 短序列(< 512)可能效果不明显，因为overhead相对较大。中长序列(> 1024)效果显著。

### Q2: 为什么torch.compile能提速？
**A**: 它会自动进行kernel fusion、消除冗余计算、优化内存布局等，相当于自动应用多种优化策略。

### Q3: 如何判断我的模型是Memory-bound还是Compute-bound？
**A**: 使用profiler看SM utilization和Memory bandwidth utilization。如果内存带宽接近峰值而SM利用率低，就是Memory-bound。

### Q4: FlashAttention的block size如何选择？
**A**: 通常64或128效果最好。太小会增加循环次数，太大会超出SRAM容量。

---

**感谢聆听！**

---

## 附录: 代码实验

```python
# 验证FlashAttention效果
import torch
import time

def benchmark_attention(seq_len, batch=1, heads=32, dim=128, iterations=100):
    Q = torch.randn(batch, heads, seq_len, dim, device='cuda', dtype=torch.float16)
    K = torch.randn(batch, heads, seq_len, dim, device='cuda', dtype=torch.float16)
    V = torch.randn(batch, heads, seq_len, dim, device='cuda', dtype=torch.float16)

    # Warmup
    for _ in range(10):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    return (time.time() - start) / iterations * 1000  # ms

# 测试不同序列长度
for seq in [512, 1024, 2048, 4096, 8192]:
    ms = benchmark_attention(seq)
    print(f"seq={seq}: {ms:.2f}ms")
```
