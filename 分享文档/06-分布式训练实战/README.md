# 分布式训练实战

> **一句话摘要**: 理解数据并行、模型并行和ZeRO等核心技术，掌握大模型分布式训练的原理和实践。

## 核心概念

### 关键术语
| 术语 | 定义 | 重要性 |
|------|------|--------|
| Data Parallelism (DP) | 数据切分到多卡，模型复制 | 最基础的并行策略 |
| Tensor Parallelism (TP) | 单层内的张量切分 | 大模型单层太大时必需 |
| Pipeline Parallelism (PP) | 按层切分模型 | 深层模型并行 |
| ZeRO | Zero Redundancy Optimizer | 消除冗余内存 |
| All-Reduce | 聚合多卡梯度的通信操作 | DP的核心通信 |

### 概念图谱
```
分布式训练策略
├── 数据并行 (Data Parallelism)
│   ├── 简单DP (复制模型)
│   ├── DDP (分布式数据并行)
│   └── FSDP (全分片数据并行)
├── 模型并行
│   ├── 张量并行 (Tensor Parallelism)
│   └── 流水线并行 (Pipeline Parallelism)
├── 优化器并行
│   ├── ZeRO-1 (优化器状态分片)
│   ├── ZeRO-2 (+ 梯度分片)
│   └── ZeRO-3 (+ 参数分片)
└── 混合并行 (3D Parallelism)
    └── DP + TP + PP
```

## 技术深度

### 1. 数据并行 (Data Parallelism)

**基本原理**:
```python
# 数据并行: 每卡持有完整模型副本，数据不同
#
# GPU 0: Model + Data[0:B//N]
# GPU 1: Model + Data[B//N:2B//N]
# ...
# GPU N-1: Model + Data[(N-1)B//N:B]

def data_parallel_forward():
    # 1. 分发数据 (Scatter)
    local_batch = all_data[rank * batch_per_gpu : (rank+1) * batch_per_gpu]

    # 2. 各自前向传播
    local_output = model(local_batch)
    local_loss = criterion(local_output, local_labels)

    # 3. 各自反向传播
    local_loss.backward()

    # 4. 梯度聚合 (All-Reduce)
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= world_size

    # 5. 同步更新
    optimizer.step()
```

**通信分析**:
```
All-Reduce通信量 (Ring All-Reduce):
- 每卡发送: 2 × (N-1)/N × P ≈ 2P (P是参数量)
- 7B模型: 2 × 7B × 2 = 28GB (FP16)
- 8卡: 每卡发送28GB
```

**PyTorch DDP实现**:
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# 包装模型
model = YourModel().cuda()
model = DDP(model, device_ids=[local_rank])

# 训练循环 (与单卡几乎相同)
for batch in dataloader:
    loss = model(batch)
    loss.backward()  # DDP自动处理梯度同步
    optimizer.step()
```

### 2. ZeRO (Zero Redundancy Optimizer)

**核心思想**: 消除数据并行中的冗余存储

**传统DP的冗余**:
```
8卡训练7B模型，每卡存储:
- 模型参数: 14GB (FP16)
- 梯度: 14GB
- Adam状态: 56GB
- 总计: 84GB × 8 = 672GB

实际需要: 只有84GB，其他都是冗余!
```

**ZeRO三个阶段**:

```python
# ZeRO-1: 优化器状态分片
# 每卡只存 1/N 的Adam状态
memory_zero1 = {
    'params': 14,           # 仍然完整
    'gradients': 14,        # 仍然完整
    'adam_states': 56 / 8,  # 分片: 7GB
}
# 总计: 35GB (减少49GB)

# ZeRO-2: + 梯度分片
memory_zero2 = {
    'params': 14,           # 仍然完整
    'gradients': 14 / 8,    # 分片: 1.75GB
    'adam_states': 56 / 8,  # 分片: 7GB
}
# 总计: 22.75GB (减少61GB)

# ZeRO-3: + 参数分片
memory_zero3 = {
    'params': 14 / 8,       # 分片: 1.75GB
    'gradients': 14 / 8,    # 分片: 1.75GB
    'adam_states': 56 / 8,  # 分片: 7GB
}
# 总计: 10.5GB (减少73GB!)
```

**ZeRO-3的通信开销**:
```python
# 前向传播: All-Gather参数
# 反向传播: All-Gather参数 + Reduce-Scatter梯度

# 通信量 = 3 × P (比DP的2P多50%)
# 换来: 内存减少 ~N倍
```

**DeepSpeed使用**:
```python
import deepspeed

# 配置
ds_config = {
    "zero_optimization": {
        "stage": 2,  # ZeRO-2
        "offload_optimizer": {"device": "cpu"},  # 可选: offload到CPU
    },
    "fp16": {"enabled": True},
    "train_batch_size": 32,
}

# 初始化
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config,
)

# 训练 (API略有不同)
for batch in dataloader:
    loss = model(batch)
    model.backward(loss)
    model.step()
```

### 3. 张量并行 (Tensor Parallelism)

**问题**: 单层太大，单卡放不下

**解决**: 切分层内的矩阵

```python
# 线性层: Y = XW + b
# W: [d_in, d_out]

# 列切分 (Column Parallel):
# W = [W1, W2]  # 切成2块
# Y = X @ [W1, W2] = [X@W1, X@W2]
# 每卡计算一半，最后Concat

# 行切分 (Row Parallel):
# W = [W1; W2]  # 按行切
# Y = X @ [W1; W2] = X1@W1 + X2@W2
# 每卡计算一半，最后All-Reduce
```

**Transformer层的张量并行**:
```python
# MLP层: 列切分 → 行切分 (消除中间通信)
#
# FFN: X → Linear1 → GELU → Linear2 → Y
#
#      X        [Column]      [No Comm]     [Row]       Y
# ┌─────────┐  ┌───────────┐  ┌───────┐  ┌───────────┐
# │    X    │→│ W1_a W1_b │→│ GELU  │→│   W2_a   │→ All-Reduce
# └─────────┘  └───────────┘  └───────┘  │   W2_b   │
#                GPU0  GPU1              └───────────┘

# Attention: 每个head分配到不同GPU
# 12 heads, 4 GPUs → 每卡3个heads
```

**通信模式**:
```
一个Transformer层的TP通信:
- Attention: 2次All-Reduce (QKV和Output)
- FFN: 2次All-Reduce
- 总计: 4次All-Reduce/层

通信量: 4 × 2 × B × S × d = 8BSd bytes
比DP小很多 (DP是全部参数)
```

### 4. 流水线并行 (Pipeline Parallelism)

**原理**: 按层切分模型，像工厂流水线

```python
# 4 GPUs, 24层模型
# GPU 0: Layer 0-5
# GPU 1: Layer 6-11
# GPU 2: Layer 12-17
# GPU 3: Layer 18-23

def pipeline_forward(micro_batches):
    # Micro-batch切分减少bubble
    for mb in micro_batches:
        # GPU 0 处理
        x = gpu0_layers(mb)
        # 发送到 GPU 1
        send(x, dst=1)
        x = recv(src=0)
        # GPU 1 处理
        x = gpu1_layers(x)
        # ...以此类推
```

**Pipeline Bubble问题**:
```
朴素Pipeline (4 stages, 1 micro-batch):
时间: |F0|--|--|--|F1|--|--|--|F2|--|--|--|F3|B3|--|--|--|B2|--|--|--|B1|--|--|--|B0|
      GPU0       GPU1       GPU2       GPU3

Bubble时间 ≈ (P-1) × (F + B)  # P是stages数

解决: Micro-batch + 1F1B调度
时间: |F0|F1|F2|F3|B3|B2|B1|B0|  # 更紧凑
      重叠计算，减少bubble
```

**GPipe vs 1F1B**:
```
GPipe: 所有forward完成后再backward
- 简单
- 激活内存: O(micro_batches × layers)

1F1B: 一个forward接一个backward
- 复杂
- 激活内存: O(pipeline_stages) # 显著减少
```

### 5. 3D并行 (混合并行)

**大模型训练的标配**:
```python
# Megatron-LM的3D并行
# 假设64卡训练，175B模型

# 配置:
# - Tensor Parallel (TP) = 8  # 一个节点内
# - Pipeline Parallel (PP) = 4  # 跨节点
# - Data Parallel (DP) = 2  # 64 / 8 / 4 = 2

# 分组:
# TP groups: [0-7], [8-15], ..., [56-63]  # 8个组
# PP groups: [0,8,16,24], [1,9,17,25], ...  # 16个组
# DP groups: [0,32], [1,33], ...  # 32个组
```

**为什么这样配置?**:
```
1. TP在节点内 (NVLink高带宽):
   - 通信频繁 (每层4次)
   - 需要低延迟高带宽

2. PP跨节点 (InfiniBand):
   - 通信只在stage边界
   - 可以容忍较高延迟

3. DP最外层:
   - 全局batch size扩展
   - All-Reduce可以与计算重叠
```

### 6. 通信优化

**梯度桶化 (Gradient Bucketing)**:
```python
# 问题: 每个参数单独All-Reduce → 通信开销大

# 解决: 打包成桶，一起通信
bucket_size = 25 * 1024 * 1024  # 25MB

buckets = []
current_bucket = []
current_size = 0

for param in model.parameters():
    if current_size + param.numel() > bucket_size:
        buckets.append(current_bucket)
        current_bucket = [param]
        current_size = param.numel()
    else:
        current_bucket.append(param)
        current_size += param.numel()
```

**计算-通信重叠**:
```python
# 反向传播时，已计算的梯度可以先通信

async_handles = []

for layer in reversed(model.layers):
    # 计算当前层梯度
    layer.backward()

    # 异步发起通信
    handle = dist.all_reduce(layer.grad, async_op=True)
    async_handles.append(handle)

# 等待所有通信完成
for handle in async_handles:
    handle.wait()
```

## 实践代码

### PyTorch FSDP示例

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

def setup_fsdp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    model = YourLargeModel()

    # 包装成FSDP
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
        # 或 SHARD_GRAD_OP for ZeRO-2
        mixed_precision=MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        ),
        device_id=local_rank,
    )

    return model


def train_fsdp(model, dataloader, optimizer):
    for batch in dataloader:
        optimizer.zero_grad()

        # FSDP自动处理参数聚合和分片
        loss = model(batch)
        loss.backward()

        optimizer.step()
```

### 通信Benchmark

```python
import torch
import torch.distributed as dist
import time

def benchmark_all_reduce(size_mb=100, iterations=10):
    """测量All-Reduce带宽"""
    tensor = torch.randn(size_mb * 1024 * 256, device='cuda')  # size_mb MB

    # Warmup
    for _ in range(3):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    # 计算带宽
    world_size = dist.get_world_size()
    # Ring All-Reduce: 2 * (N-1)/N * size
    data_volume = 2 * (world_size - 1) / world_size * size_mb * iterations
    bandwidth = data_volume / elapsed

    if dist.get_rank() == 0:
        print(f"All-Reduce {size_mb}MB x {iterations}: {elapsed:.2f}s")
        print(f"Effective bandwidth: {bandwidth:.2f} MB/s")

    return bandwidth
```

## 关键洞察

### 核心收获

1. **数据并行是基础**: 简单有效，但有冗余

2. **ZeRO消除冗余**: Stage 3可以减少90%内存，代价是50%通信增加

3. **张量并行适合大层**: 单层放不下时必须，但通信密集

4. **流水线并行有bubble**: 需要micro-batch来减少

5. **3D并行是大模型标配**: 结合各策略优点

### 并行策略选择指南

| 场景 | 推荐策略 |
|------|----------|
| 模型放得下单卡 | 数据并行 (DDP) |
| 模型略超单卡内存 | ZeRO-2/3 |
| 单层超大 (>10GB) | 张量并行 |
| 模型极深 (>100层) | 流水线并行 |
| 超大模型 (>100B) | 3D并行 |

### 常见误区

| 误区 | 正确理解 |
|------|----------|
| 更多卡总是更快 | 通信开销可能超过收益 |
| ZeRO-3最好 | 通信增加50%，需要权衡 |
| 流水线并行高效 | Bubble开销显著 |
| 只用一种并行 | 大模型需要组合策略 |

## 延伸阅读

### 推荐论文
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)

### 推荐框架
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - 微软的分布式训练框架
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - NVIDIA的大模型训练框架
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html) - PyTorch原生支持

### 相关专题
- [大模型内存分析](../04-大模型内存分析/) - 理解内存需求
- [GPU架构与性能优化](../03-GPU架构与性能优化/) - 硬件基础

---

## 内容来源

本文档内容整理自以下来源：
- [来源: 学习笔记/02-硬件与系统/] (待学习完成后补充)
- [来源: 深度讨论/分布式训练相关讨论.md] (待补充)

**注**: 本专题基于课程大纲规划，详细内容将在完成Lecture 07-08学习后补充。

---

**作者**: peixingxin + Claude Code
**创建日期**: 2025-12-17
**最后更新**: 2025-12-17
