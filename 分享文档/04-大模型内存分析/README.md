# 大模型内存分析

> **一句话摘要**: 精确计算大模型训练和推理的内存需求，理解混合精度训练的内存布局，掌握KV Cache的内存优化。

## 核心概念

### 关键术语
| 术语 | 定义 | 重要性 |
|------|------|--------|
| FP32 Master Weights | 全精度主权重副本，用于数值稳定 | 混合精度训练必须 |
| Adam States | 优化器状态 (动量+方差) | 训练内存主要开销 |
| Activation Memory | 前向传播保存的激活值 | 反向传播需要 |
| KV Cache | 推理时缓存的Key/Value | 长序列推理瓶颈 |
| Gradient Checkpointing | 用重算换内存的技术 | 降低激活内存 |

### 概念图谱
```
大模型内存
├── 训练内存
│   ├── 权重相关
│   │   ├── FP32 Master Weights (28GB/7B)
│   │   ├── FP16 Working Weights (14GB/7B)
│   │   └── FP16 Gradients (14GB/7B)
│   ├── 优化器状态
│   │   ├── Adam Momentum (28GB/7B)
│   │   └── Adam Variance (28GB/7B)
│   └── 激活内存 (batch, seq依赖)
└── 推理内存
    ├── 模型权重 (14GB/7B in FP16)
    ├── KV Cache (序列长度依赖)
    └── 临时内存
```

## 技术深度

### 1. 7B模型训练内存精确计算

**基本公式**:
```python
# 7B 参数 = 7 × 10^9 个浮点数

# 混合精度训练内存布局:
memory = {
    # 权重相关 (必须)
    'fp32_master_weights': 7e9 * 4,      # 28GB - 主副本
    'fp16_working_weights': 7e9 * 2,     # 14GB - 计算用
    'fp16_gradients': 7e9 * 2,           # 14GB - 反向传播

    # Adam优化器状态 (必须)
    'adam_momentum': 7e9 * 4,            # 28GB - 一阶矩
    'adam_variance': 7e9 * 4,            # 28GB - 二阶矩
}

# 基础内存 (不含激活):
base_memory = sum(memory.values()) / (1024**3)
# = 112 GB
```

**为什么需要FP32 Master Weights?**
```python
# 数值稳定性问题演示
def fp16_precision_loss():
    # FP16精度: ~3.3位有效小数
    fp16_weight = 1.0
    fp16_update = 1e-6  # 小更新

    # 在FP16中，这个更新可能被丢弃!
    result = fp16_weight + fp16_update  # 还是 1.0

    # 在FP32中，精度得到保持
    fp32_weight = 1.0
    fp32_update = 1e-6
    result = fp32_weight + fp32_update  # 1.000001

# 这就是为什么需要FP32主副本存储累积更新
```

**激活内存估算**:
```python
def activation_memory(batch_size, seq_len, hidden_size, num_layers):
    """
    估算激活内存 (无checkpointing)
    """
    # 每层主要激活:
    # - Attention输入: B × S × H
    # - QKV投影: 3 × B × S × H
    # - Attention Scores: B × H × S × S (这是大头!)
    # - FFN激活: B × S × (4H)

    # 简化估算 (FP16):
    per_layer = batch_size * seq_len * hidden_size * 4  # 约4倍hidden
    per_layer_bytes = per_layer * 2  # FP16

    # Attention Scores (每个head)
    num_heads = hidden_size // 128  # 假设head_dim=128
    attn_scores = batch_size * num_heads * seq_len * seq_len * 2

    total = (per_layer_bytes + attn_scores) * num_layers
    return total / (1024**3)  # GB

# 示例: 7B模型
# batch=32, seq=2048, hidden=4096, layers=32
activation_gb = activation_memory(32, 2048, 4096, 32)
# ≈ 48GB (无checkpointing)
```

**完整训练内存**:
```
7B模型训练内存总结:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
组件                    大小
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FP32 主权重              28 GB
FP16 工作权重            14 GB
FP16 梯度                14 GB
Adam 动量                28 GB
Adam 方差                28 GB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
基础内存                112 GB
激活内存 (变化)         16-48 GB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总计                   128-160 GB

注意: 这远超简化计算的 56GB!
```

### 2. 不同精度策略对比

```python
def precision_strategies(num_params=7e9):
    """
    不同精度策略的内存对比
    """
    strategies = {
        'FP32 Full Precision': {
            'weights': num_params * 4,        # 28GB
            'gradients': num_params * 4,      # 28GB
            'adam_states': num_params * 8,    # 56GB
            'total_base': 112,  # GB
            'pros': '最高精度',
            'cons': '内存翻倍'
        },

        'FP16 Mixed Precision': {
            'fp32_master': num_params * 4,    # 28GB
            'fp16_weights': num_params * 2,   # 14GB
            'fp16_gradients': num_params * 2, # 14GB
            'adam_states': num_params * 8,    # 56GB (仍然FP32)
            'total_base': 112,  # GB
            'pros': '计算快2x, 精度保持',
            'cons': '需要Loss Scaling'
        },

        'BF16 Mixed Precision': {
            'bf32_master': num_params * 4,    # 28GB
            'bf16_weights': num_params * 2,   # 14GB
            'bf16_gradients': num_params * 2, # 14GB
            'adam_states': num_params * 8,    # 56GB
            'total_base': 112,  # GB
            'pros': '无需Loss Scaling, 更稳定',
            'cons': '需要Ampere+硬件'
        },

        'FP16 + 8bit Adam': {
            'fp32_master': num_params * 4,    # 28GB
            'fp16_weights': num_params * 2,   # 14GB
            'fp16_gradients': num_params * 2, # 14GB
            'adam_8bit': num_params * 2,      # 14GB (8bit × 2)
            'total_base': 70,   # GB
            'pros': '节省42GB (37%)',
            'cons': '可能影响收敛'
        }
    }
    return strategies
```

**对比表**:
| 策略 | 基础内存 | 相对FP32 | 适用场景 |
|------|----------|----------|----------|
| FP32 Full | 112 GB | 100% | 小模型/高精度需求 |
| FP16 Mixed | 112 GB | 100% | 主流方案 |
| BF16 Mixed | 112 GB | 100% | A100+推荐 |
| FP16 + 8bit Adam | 70 GB | 63% | 内存受限 |

### 3. KV Cache内存分析

**正确的KV Cache公式**:
```python
def kv_cache_memory(batch, seq_len, num_layers, num_heads, head_dim, dtype_bytes=2):
    """
    KV Cache内存计算

    注意: 使用 num_heads × head_dim 而不是 d_model
    虽然数值相等，但概念更清晰
    """
    # Key和Value各一份
    kv_per_layer = 2 * batch * seq_len * num_heads * head_dim * dtype_bytes

    total = kv_per_layer * num_layers
    return total / (1024**3)  # GB

# 7B模型配置 (LLaMA-7B like)
config_7b = {
    'num_layers': 32,
    'num_heads': 32,
    'head_dim': 128,  # d_model=4096, heads=32
}

# 不同序列长度的KV Cache
for seq_len in [512, 2048, 8192, 32768]:
    mem = kv_cache_memory(
        batch=1,
        seq_len=seq_len,
        **config_7b
    )
    print(f"seq={seq_len:5d}: {mem:.2f} GB")

# 输出:
# seq=  512: 0.25 GB
# seq= 2048: 1.00 GB
# seq= 8192: 4.00 GB
# seq=32768: 16.00 GB  ← 长序列主要开销!
```

**常见错误纠正**:
```python
# 错误公式 (早期版本):
kv_wrong = 2 * B * S * d_model * L * bytes  # 使用d_model

# 正确公式:
kv_correct = 2 * B * S * num_heads * head_dim * L * bytes

# 数值上相等 (因为 d_model = num_heads × head_dim)
# 但概念上，KV Cache是per-head的
```

**GQA (Grouped Query Attention) 优化**:
```python
def kv_cache_gqa(batch, seq_len, num_layers, num_kv_heads, head_dim):
    """
    GQA的KV Cache: 只有num_kv_heads个KV对

    LLaMA-2 70B: num_heads=64, num_kv_heads=8
    → KV Cache减少 8x!
    """
    kv_per_layer = 2 * batch * seq_len * num_kv_heads * head_dim * 2
    return kv_per_layer * num_layers / (1024**3)

# 对比 (70B模型, seq=4096)
# MHA: 8 × 4096 × 64 × 128 × 80 × 2 = 168 GB
# GQA: 8 × 4096 × 8 × 128 × 80 × 2 = 21 GB (8x减少!)
```

### 4. 内存优化技术

#### Gradient Checkpointing (梯度检查点)
```python
# 原理: 用重算换内存
# 前向传播时只保存部分激活，反向时重新计算

def forward_with_checkpointing(x, layers, checkpoint_every=2):
    """
    每隔checkpoint_every层保存一次激活
    """
    saved_activations = {}

    for i, layer in enumerate(layers):
        x = layer(x)

        if i % checkpoint_every == 0:
            saved_activations[i] = x.detach()  # 保存
        # 其他层的激活被丢弃

    return x, saved_activations

# 内存减少: 约 sqrt(num_layers) 倍
# 计算增加: 约 33% (重算)
```

**效果对比**:
| 策略 | 激活内存 | 计算开销 |
|------|----------|----------|
| 无checkpointing | 100% | 100% |
| 全部checkpoint | ~10% | ~133% |
| 每2层checkpoint | ~50% | ~117% |

#### Gradient Accumulation (梯度累积)
```python
# 问题: 想用大batch但内存不够
# 解决: 小batch前向+累积梯度，最后一起更新

def train_with_accumulation(model, dataloader, accumulation_steps=8):
    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        # 前向 + 反向 (小batch)
        loss = model(batch) / accumulation_steps
        loss.backward()  # 梯度累积

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

# 效果: 等效batch_size = 实际batch × accumulation_steps
# 内存: 只需实际batch的激活内存
```

### 5. 实际案例分析

**案例: 单卡80GB A100训练7B模型**
```python
available_memory = 80  # GB

# 基础开销 (必须)
fp32_master = 28
fp16_weights = 14
fp16_gradients = 14
adam_momentum = 28
adam_variance = 28
base_total = 112  # 超出80GB!

# 解决方案1: 8-bit Adam
adam_8bit = 14  # 从56GB降到14GB
new_total = 28 + 14 + 14 + 14 = 70  # GB

# 剩余给激活: 80 - 70 = 10GB
# 支持的batch_size (估算):
# activation ≈ batch × seq × hidden × layers × factor
# 10GB ≈ batch × 2048 × 4096 × 32 × 4 × 2 / 1e9
# batch ≈ 2

# 结论: 可以训练，但batch很小
```

**案例: 8×A100 分布式训练7B**
```python
# ZeRO Stage 2: 划分优化器状态和梯度
# 每卡:
per_gpu = {
    'weights': 14,                  # 14GB (广播)
    'gradients': 14 / 8,            # 1.75GB
    'adam_states': 56 / 8,          # 7GB
}
per_gpu_total = 14 + 1.75 + 7 = 22.75  # GB

# 剩余: 80 - 23 ≈ 57GB 给激活
# 可以用更大的batch!
```

## 实践代码

### 内存分析工具

```python
import torch

def model_memory_analysis(model, batch_size, seq_len, dtype=torch.float16):
    """
    分析模型的内存占用
    """
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 参数内存
    bytes_per_param = 2 if dtype == torch.float16 else 4
    param_memory_mb = total_params * bytes_per_param / (1024**2)

    # 混合精度训练估算
    training_memory = {
        'fp32_master': total_params * 4 / (1024**3),
        'fp16_weights': total_params * 2 / (1024**3),
        'fp16_gradients': total_params * 2 / (1024**3),
        'adam_momentum': total_params * 4 / (1024**3),
        'adam_variance': total_params * 4 / (1024**3),
    }
    training_total = sum(training_memory.values())

    print(f"模型参数量: {total_params/1e9:.2f}B")
    print(f"可训练参数: {trainable_params/1e9:.2f}B")
    print(f"\n训练内存估算 (不含激活):")
    for name, mem in training_memory.items():
        print(f"  {name}: {mem:.2f} GB")
    print(f"  总计: {training_total:.2f} GB")

    return training_memory


def measure_peak_memory():
    """
    测量实际峰值内存
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        # ... 运行你的代码 ...
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"峰值显存: {peak_memory:.2f} GB")


# 使用示例
# model = YourModel()
# model_memory_analysis(model, batch_size=8, seq_len=2048)
```

## 关键洞察

### 核心收获

1. **简化计算严重低估内存**: 7B模型实际需要112GB，不是56GB

2. **FP32主副本不能省**: 数值稳定性的关键，累积小更新需要高精度

3. **Adam是内存大户**: 占56GB (50%)，可以用8-bit Adam优化

4. **KV Cache随序列长度线性增长**: 长序列推理的主要瓶颈

5. **GQA是KV Cache的银弹**: LLaMA-2使用GQA减少8倍KV Cache

### 常见误区

| 误区 | 正确理解 |
|------|----------|
| 7B模型需要56GB | 实际需要112GB+ |
| FP16训练省一半内存 | Adam状态仍然FP32，只省计算 |
| KV Cache可以忽略 | 长序列时KV Cache>模型参数 |
| 激活内存是固定的 | 随batch和seq变化很大 |

## 延伸阅读

### 推荐资源
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) - 混合精度原始论文
- [ZeRO](https://arxiv.org/abs/1910.02054) - 分布式内存优化

### 相关专题
- [GPU架构与性能优化](../03-GPU架构与性能优化/) - 内存层次理解
- [分布式训练实战](../06-分布式训练实战/) - ZeRO等技术

---

## 内容来源

本文档内容整理自以下来源：
- [来源: 深度讨论/7B模型内存计算分析.md]
- [来源: 深度讨论/KV缓存内存计算精确分析.md]
- [来源: 学习笔记/01-基础建立/02-Lecture02-PyTorch基础/]

---

**作者**: peixingxin + Claude Code
**创建日期**: 2025-12-17
**最后更新**: 2025-12-17
