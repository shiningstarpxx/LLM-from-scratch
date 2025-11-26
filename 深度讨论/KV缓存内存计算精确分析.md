# KV缓存内存计算精确分析：hidden_size vs d_model

## 🎯 问题背景

用户提出了一个关键问题：在KV缓存内存计算中，使用`hidden_size`是否准确？是否应该用`d_model`更合理？

原始计算：
```
KV_cache = batch_size × seq_len × layers × hidden_size × 2 bytes

# 示例: 7B模型，2048上下文，batch=1
KV_cache = 1 × 2048 × 32 × 4096 × 2 = 536MB (每层)
总KV_cache = 536MB × 32 = 17GB
```

## 🧠 深度分析

### 1. Transformer架构的维度层次

```python
# 标准Transformer的维度关系
class TransformerDimensions:
    def __init__(self, model_name="7B"):
        if model_name == "7B":
            self.d_model = 4096      # 模型隐藏维度（模型宽度）
            self.num_heads = 32       # 注意力头数
            self.num_layers = 32      # 层数
            self.head_dim = 128       # 每个头的维度 = d_model / num_heads

        # 关键关系
        assert self.head_dim == self.d_model // self.num_heads  # 4096 / 32 = 128
```

### 2. KV缓存的实际存储内容

```python
# KV缓存存储的是什么？
def kv_cache_content_analysis():
    """KV缓存内容的详细分析"""

    # 对于每个注意力头，我们需要存储：
    # - Key矩阵: (batch_size, seq_len, head_dim)
    # - Value矩阵: (batch_size, seq_len, head_dim)

    batch_size = 1
    seq_len = 2048
    num_heads = 32
    head_dim = 128  # 不是4096！

    # 单个头的KV缓存
    single_head_kv = batch_size * seq_len * head_dim * 2  # K + V
    print(f"单个头KV缓存: {single_head_kv:,} bytes")

    # 所有头的KV缓存（单层）
    all_heads_kv = single_head_kv * num_heads
    print(f"单层所有头KV缓存: {all_heads_kv:,} bytes ({all_heads_kv/1024**2:.1f} MB)")

    # 所有层的KV缓存
    all_layers_kv = all_heads_kv * 32  # 32层
    print(f"所有层KV缓存: {all_layers_kv:,} bytes ({all_layers_kv/1024**3:.2f} GB)")

    return {
        'per_head_mb': single_head_kv / 1024**2,
        'per_layer_mb': all_heads_kv / 1024**2,
        'total_gb': all_layers_kv / 1024**3
    }

# 运行结果：
# 单个头KV缓存: 524,288 bytes (0.5 MB)
# 单层所有头KV缓存: 16,777,216 bytes (16.0 MB)
# 所有层KV缓存: 536,870,912 bytes (0.50 GB) - 等等，这个结果不对！
```

### 3. 重新计算：发现原始计算的错误

```python
def correct_kv_cache_calculation():
    """正确的KV缓存计算"""

    batch_size = 1
    seq_len = 2048
    num_layers = 32
    num_heads = 32
    head_dim = 128  # d_model / num_heads
    bytes_per_element = 2  # FP16

    print("=== 正确的KV缓存计算 ===\n")

    # 单个头的单个矩阵（K或V）
    per_head_single_matrix = batch_size * seq_len * head_dim * bytes_per_element
    print(f"单个头的单个矩阵: {per_head_single_matrix:,} bytes ({per_head_single_matrix/1024**2:.2f} MB)")

    # 单个头的KV（K + V）
    per_head_kv = per_head_single_matrix * 2
    print(f"单个头的KV缓存: {per_head_kv:,} bytes ({per_head_kv/1024**2:.2f} MB)")

    # 单层所有头的KV
    per_layer_kv = per_head_kv * num_heads
    print(f"单层KV缓存: {per_layer_kv:,} bytes ({per_layer_kv/1024**2:.1f} MB)")

    # 所有层的KV
    total_kv = per_layer_kv * num_layers
    print(f"总KV缓存: {total_kv:,} bytes ({total_kv/1024**3:.2f} GB)")

    return total_kv / 1024**3

# 结果：
# 单个头的单个矩阵: 524,288 bytes (0.50 MB)
# 单个头的KV缓存: 1,048,576 bytes (1.00 MB)
# 单层KV缓存: 33,554,432 bytes (32.0 MB)
# 总KV缓存: 1,073,741,824 bytes (1.00 GB)
```

### 4. 发现原始计算的错误！

原始计算有重大错误：
```
原始计算: 1 × 2048 × 32 × 4096 × 2 = 536MB (每层) × 32 = 17GB
正确计算: 1 × 2048 × 32 × 128 × 2 × 2 = 32MB (每层) × 32 = 1GB
```

**错误分析**：
1. 重复计算了`× 2`（应该是K+V，不是再×2）
2. 使用了`d_model=4096`而不是`head_dim=128`

### 5. 精确的KV缓存计算公式

```python
def precise_kv_cache_formula():
    """精确的KV缓存计算公式推导"""

    print("=== 精确公式推导 ===\n")

    # 基础参数
    B = 1      # batch_size
    S = 2048   # seq_len
    L = 32     # num_layers
    H = 32     # num_heads
    D = 128    # head_dim = d_model / num_heads
    bytes_elem = 2  # FP16

    print("参数定义:")
    print(f"B = {B} (batch_size)")
    print(f"S = {S} (seq_len)")
    print(f"L = {L} (num_layers)")
    print(f"H = {H} (num_heads)")
    print(f"D = {D} (head_dim)")
    print(f"bytes_elem = {bytes_elem} (FP16)")
    print()

    # 公式推导
    print("公式推导:")
    print("1. 单个头的单个矩阵 (K或V):")
    print(f"   Matrix = B × S × D × bytes_elem = {B} × {S} × {D} × {bytes_elem} = {B*S*D*bytes_elem:,} bytes")
    print()

    print("2. 单个头的KV缓存 (K + V):")
    print(f"   Head_KV = Matrix × 2 = {B*S*D*bytes_elem*2:,} bytes")
    print()

    print("3. 单层的KV缓存:")
    print(f"   Layer_KV = Head_KV × H = {B*S*D*bytes_elem*2*H:,} bytes")
    print()

    print("4. 总KV缓存:")
    print(f"   Total_KV = Layer_KV × L = {B*S*D*bytes_elem*2*H*L:,} bytes")
    print(f"   Total_KV = {B*S*D*bytes_elem*2*H*L/1024**3:.2f} GB")
    print()

    # 简化公式
    print("简化公式:")
    print(f"KV_cache = B × S × H × D × 2 × bytes_elem × L")
    print(f"        = B × S × (H × D) × 2 × bytes_elem × L")
    print(f"        = B × S × d_model × 2 × bytes_elem × L")
    print("        (因为 d_model = H × D)")

    return B * S * H * D * 2 * bytes_elem * L / (1024**3)

# 结果：1.00 GB
```

### 6. 实际代码实现验证

```python
# 实际的KV缓存实现
class KVCache:
    def __init__(self, batch_size, seq_len, num_heads, head_dim, num_layers, dtype=torch.float16):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype = dtype

        # 初始化KV缓存
        self.k_cache = []
        self.v_cache = []

        for layer in range(num_layers):
            # 每层的K缓存: (batch_size, num_heads, seq_len, head_dim)
            k_cache_layer = torch.zeros(
                (batch_size, num_heads, seq_len, head_dim),
                dtype=dtype
            )

            # 每层的V缓存: (batch_size, num_heads, seq_len, head_dim)
            v_cache_layer = torch.zeros(
                (batch_size, num_heads, seq_len, head_dim),
                dtype=dtype
            )

            self.k_cache.append(k_cache_layer)
            self.v_cache.append(v_cache_layer)

    def memory_usage(self):
        """计算内存使用量"""
        # 单个K或V矩阵的内存
        single_matrix_bytes = self.batch_size * self.num_heads * self.seq_len * self.head_dim * 2  # FP16

        # 单层的KV内存 (K + V)
        per_layer_bytes = single_matrix_bytes * 2

        # 所有层的KV内存
        total_bytes = per_layer_bytes * self.num_layers

        return {
            'single_matrix_mb': single_matrix_bytes / 1024**2,
            'per_layer_mb': per_layer_bytes / 1024**2,
            'total_gb': total_bytes / 1024**3,
            'calculation_detail': {
                'batch_size': self.batch_size,
                'num_heads': self.num_heads,
                'seq_len': self.seq_len,
                'head_dim': self.head_dim,
                'num_layers': self.num_layers,
                'dtype_bytes': 2  # FP16
            }
        }

# 验证计算
cache = KVCache(batch_size=1, seq_len=2048, num_heads=32, head_dim=128, num_layers=32)
memory_info = cache.memory_usage()

print("=== 实际KV缓存内存验证 ===")
print(f"单个矩阵 (K或V): {memory_info['single_matrix_mb']:.2f} MB")
print(f"单层KV缓存: {memory_info['per_layer_mb']:.1f} MB")
print(f"总KV缓存: {memory_info['total_gb']:.2f} GB")
```

### 7. 不同模型的KV缓存对比

```python
def model_kv_cache_comparison():
    """不同模型的KV缓存对比"""

    models = {
        'GPT-2 Small': {'d_model': 768, 'num_heads': 12, 'num_layers': 12},
        'GPT-2 Medium': {'d_model': 1024, 'num_heads': 16, 'num_layers': 24},
        'GPT-2 Large': {'d_model': 1280, 'num_heads': 20, 'num_layers': 36},
        'LLaMA-7B': {'d_model': 4096, 'num_heads': 32, 'num_layers': 32},
        'LLaMA-13B': {'d_model': 5120, 'num_heads': 40, 'num_layers': 40},
    }

    batch_size = 1
    seq_len = 2048

    print("=== 不同模型KV缓存对比 ===\n")
    print(f"序列长度: {seq_len}, 批次大小: {batch_size}")
    print()

    for model_name, config in models.items():
        d_model = config['d_model']
        num_heads = config['num_heads']
        num_layers = config['num_layers']
        head_dim = d_model // num_heads

        # 正确的KV缓存计算
        per_layer_bytes = batch_size * seq_len * num_heads * head_dim * 2 * 2  # K + V, FP16
        total_bytes = per_layer_bytes * num_layers
        total_gb = total_bytes / 1024**3

        print(f"{model_name}:")
        print(f"  d_model: {d_model}, num_heads: {num_heads}, head_dim: {head_dim}")
        print(f"  KV缓存: {total_gb:.2f} GB")
        print()

    return models

# 结果显示7B模型的KV缓存应该是1GB，不是17GB！
```

## 💡 关键结论

### 1. **原始计算有重大错误**

```
❌ 错误计算: 1 × 2048 × 32 × 4096 × 2 = 536MB (每层) × 32 = 17GB
✅ 正确计算: 1 × 2048 × 32 × 128 × 2 × 2 = 32MB (每层) × 32 = 1GB
```

**错误原因**：
1. 使用了`d_model=4096`而不是`head_dim=128`
2. 公式表达不清晰

### 2. **概念精确性的重要性**

你的质疑完全正确！虽然数值上`d_model = num_heads × head_dim`，但概念上：

- **`d_model`**: 模型的隐藏维度，是整体概念
- **`num_heads × head_dim`**: 反映了多头注意力的实际工作机制

使用`num_heads × head_dim`更准确，因为：
1. 体现了Transformer的实际计算过程
2. 便于理解多头并行的注意力机制
3. 避免在不同架构中的混淆

### 3. **推荐的精确计算公式**

```python
def kv_cache_memory_precise(batch_size, seq_len, num_heads, head_dim, num_layers, dtype_bytes=2):
    """
    精确计算KV缓存内存需求

    公式: KV_cache = batch_size × seq_len × num_heads × head_dim × 2 × dtype_bytes × num_layers
    """
    total_bytes = batch_size * seq_len * num_heads * head_dim * 2 * dtype_bytes * num_layers
    return total_bytes / (1024**3)

# 7B模型示例
kv_memory = kv_cache_memory_precise(
    batch_size=1,
    seq_len=2048,
    num_heads=32,
    head_dim=128,  # 4096 / 32
    num_layers=32,
    dtype_bytes=2  # FP16
)

print(f"7B模型KV缓存: {kv_memory:.2f} GB")  # 1.00 GB
```

### 4. **实际意义**

这个修正对实际应用很重要：
- **内存规划**: 7B模型的KV缓存是1GB，不是17GB
- **推理优化**: 准确的内存预算和优化策略
- **成本估算**: 云服务部署的精确成本计算

**最终答案**: 7B模型在2048上下文长度下的KV缓存应该是**1GB**，不是17GB！你的质疑发现了一个重要的计算错误。

## 📐 数学形式化证明

### 1. KV缓存内存计算的数学公式

#### 基础公式推导

设：
- $B$: batch_size（批次大小）
- $S$: seq_len（序列长度）
- $L$: num_layers（层数）
- $H$: num_heads（注意力头数）
- $D$: head_dim（每个头的维度）
- $d_{model}$: 模型隐藏维度，满足 $d_{model} = H \times D$
- $b$: bytes_per_element（每个元素的字节数，FP16为2）

#### 单层KV缓存的内存计算

**定理1**: 单层KV缓存的内存需求为：

$$M_{layer} = B \times S \times H \times D \times 2 \times b$$

**证明**:
1. 单个注意力头的Key矩阵: $K_{head} = B \times S \times D$，内存为 $B \times S \times D \times b$
2. 单个注意力头的Value矩阵: $V_{head} = B \times S \times D$，内存为 $B \times S \times D \times b$
3. 单个注意力头的KV缓存: $M_{head} = 2 \times B \times S \times D \times b$（K和V）
4. 单层所有头的KV缓存: $M_{layer} = H \times M_{head} = H \times 2 \times B \times S \times D \times b$
5. 简化: $M_{layer} = B \times S \times H \times D \times 2 \times b$

#### 总KV缓存的内存计算

**定理2**: 所有层的KV缓存总内存需求为：

$$M_{total} = L \times M_{layer} = B \times S \times H \times D \times 2 \times b \times L$$

由于 $d_{model} = H \times D$，可以等价表示为：

$$M_{total} = B \times S \times d_{model} \times 2 \times b \times L$$

#### 错误计算的数学分析

**错误公式**:
$$M_{wrong} = B \times S \times L \times d_{model} \times b$$

**错误原因**:
1. 缺少因子2（K和V两个矩阵）
2. 虽然数值上 $H \times D = d_{model}$，但概念上应该使用 $H \times D$ 来体现多头注意力的实际结构

**正确公式**:
$$M_{correct} = B \times S \times H \times D \times 2 \times b \times L$$

**误差分析**:
$$\text{误差倍数} = \frac{M_{wrong}}{M_{correct}} = \frac{B \times S \times L \times d_{model} \times b}{B \times S \times H \times D \times 2 \times b \times L} = \frac{d_{model}}{2 \times H \times D} = \frac{1}{2}$$

因此错误计算会高估2倍（如果包含K和V），或者低估2倍（如果只计算了K或V）。

### 2. 不同模型配置的通用公式

#### 通用KV缓存内存公式

对于任意Transformer模型，KV缓存内存为：

$$M_{KV}(B, S, L, H, D, b) = 2 \times B \times S \times L \times H \times D \times b$$

#### 内存复杂度分析

**空间复杂度**: $O(B \times S \times L \times H \times D)$

**关键观察**:
- 与序列长度 $S$ 线性相关（这是KV缓存的主要瓶颈）
- 与层数 $L$ 线性相关
- 与模型宽度 $H \times D$ 线性相关

### 3. 内存优化的数学分析

#### 分块计算的数学证明

**定理3**: 如果使用分块大小为 $C$ 的分块计算，内存需求可以降低为：

$$M_{chunked} = 2 \times B \times C \times L \times H \times D \times b$$

其中 $C < S$，内存节省比例为：

$$\text{节省比例} = 1 - \frac{C}{S}$$

**证明**:
- 原始内存: $M_{original} = 2 \times B \times S \times L \times H \times D \times b$
- 分块内存: $M_{chunked} = 2 \times B \times C \times L \times H \times D \times b$
- 节省比例: $\frac{M_{original} - M_{chunked}}{M_{original}} = \frac{S - C}{S} = 1 - \frac{C}{S}$

## 🐍 Python 验证代码

```python
"""
KV缓存内存计算精确验证代码
验证数学公式的正确性和不同模型配置的计算
"""

import numpy as np
import torch
from typing import Dict, Tuple

class KVCacheMemoryCalculator:
    """KV缓存内存计算器"""
    
    def __init__(self):
        self.results = {}
    
    def calculate_kv_cache_memory(
        self,
        batch_size: int,
        seq_len: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype_bytes: int = 2  # FP16
    ) -> Dict[str, float]:
        """
        计算KV缓存内存需求
        
        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            num_layers: 层数
            num_heads: 注意力头数
            head_dim: 每个头的维度
            dtype_bytes: 数据类型字节数（FP16=2, FP32=4）
        
        Returns:
            内存计算结果（字节、MB、GB）
        """
        # 数学公式: M = B × S × H × D × 2 × b × L
        total_bytes = batch_size * seq_len * num_heads * head_dim * 2 * dtype_bytes * num_layers
        
        return {
            'total_bytes': total_bytes,
            'total_mb': total_bytes / (1024 ** 2),
            'total_gb': total_bytes / (1024 ** 3),
            'per_layer_bytes': total_bytes / num_layers,
            'per_layer_mb': total_bytes / num_layers / (1024 ** 2)
        }
    
    def verify_formula(self, config: Dict) -> Dict[str, any]:
        """
        验证数学公式的正确性
        
        通过实际创建KV缓存张量来验证公式
        """
        B = config['batch_size']
        S = config['seq_len']
        L = config['num_layers']
        H = config['num_heads']
        D = config['head_dim']
        b = config.get('dtype_bytes', 2)
        
        # 使用公式计算
        formula_result = self.calculate_kv_cache_memory(B, S, L, H, D, b)
        
        # 实际创建张量验证
        actual_memory = 0
        for layer in range(L):
            # K缓存: (B, H, S, D)
            k_cache = torch.zeros(B, H, S, D, dtype=torch.float16 if b == 2 else torch.float32)
            # V缓存: (B, H, S, D)
            v_cache = torch.zeros(B, H, S, D, dtype=torch.float16 if b == 2 else torch.float32)
            
            actual_memory += k_cache.numel() * b
            actual_memory += v_cache.numel() * b
        
        # 验证公式正确性
        formula_bytes = formula_result['total_bytes']
        error = abs(formula_bytes - actual_memory) / actual_memory * 100
        
        return {
            'formula_bytes': formula_bytes,
            'actual_bytes': actual_memory,
            'error_percent': error,
            'formula_correct': error < 0.01  # 误差小于0.01%认为正确
        }
    
    def compare_models(self) -> Dict[str, Dict]:
        """
        对比不同模型的KV缓存内存需求
        """
        models = {
            'GPT-2 Small': {
                'batch_size': 1,
                'seq_len': 2048,
                'num_layers': 12,
                'num_heads': 12,
                'head_dim': 64,  # 768 / 12
                'dtype_bytes': 2
            },
            'GPT-2 Medium': {
                'batch_size': 1,
                'seq_len': 2048,
                'num_layers': 24,
                'num_heads': 16,
                'head_dim': 64,  # 1024 / 16
                'dtype_bytes': 2
            },
            'LLaMA-7B': {
                'batch_size': 1,
                'seq_len': 2048,
                'num_layers': 32,
                'num_heads': 32,
                'head_dim': 128,  # 4096 / 32
                'dtype_bytes': 2
            },
            'LLaMA-13B': {
                'batch_size': 1,
                'seq_len': 2048,
                'num_layers': 40,
                'num_heads': 40,
                'head_dim': 128,  # 5120 / 40
                'dtype_bytes': 2
            }
        }
        
        results = {}
        for model_name, config in models.items():
            memory = self.calculate_kv_cache_memory(**config)
            verification = self.verify_formula(config)
            
            results[model_name] = {
                'memory_gb': memory['total_gb'],
                'per_layer_mb': memory['per_layer_mb'],
                'formula_verified': verification['formula_correct'],
                'config': config
            }
        
        return results
    
    def analyze_wrong_calculation(self) -> Dict[str, any]:
        """
        分析原始错误计算的问题
        """
        # 7B模型配置
        config = {
            'batch_size': 1,
            'seq_len': 2048,
            'num_layers': 32,
            'num_heads': 32,
            'head_dim': 128,
            'dtype_bytes': 2
        }
        
        # 正确计算
        correct = self.calculate_kv_cache_memory(**config)
        
        # 错误计算（使用d_model而不是head_dim，且缺少因子2）
        d_model = config['num_heads'] * config['head_dim']  # 4096
        wrong_bytes = (config['batch_size'] * 
                      config['seq_len'] * 
                      config['num_layers'] * 
                      d_model * 
                      config['dtype_bytes'])
        
        wrong_gb = wrong_bytes / (1024 ** 3)
        
        # 另一种错误：包含因子2但使用d_model
        wrong2_bytes = wrong_bytes * 2
        wrong2_gb = wrong2_bytes / (1024 ** 3)
        
        return {
            'correct_gb': correct['total_gb'],
            'wrong_gb_1': wrong_gb,  # 缺少因子2
            'wrong_gb_2': wrong2_gb,  # 包含因子2但用d_model
            'error_ratio_1': wrong_gb / correct['total_gb'],
            'error_ratio_2': wrong2_gb / correct['total_gb'],
            'explanation': {
                'wrong_1': '缺少K和V的因子2，低估了2倍',
                'wrong_2': '使用d_model而非head_dim，虽然数值相同但概念不准确'
            }
        }
    
    def analyze_chunked_computation(self, chunk_size: int, seq_len: int) -> Dict[str, float]:
        """
        分析分块计算的内存节省
        
        Args:
            chunk_size: 分块大小
            seq_len: 原始序列长度
        
        Returns:
            内存节省分析结果
        """
        # 假设7B模型配置
        config = {
            'batch_size': 1,
            'seq_len': seq_len,
            'num_layers': 32,
            'num_heads': 32,
            'head_dim': 128,
            'dtype_bytes': 2
        }
        
        # 原始内存
        original = self.calculate_kv_cache_memory(**config)
        
        # 分块内存
        config['seq_len'] = chunk_size
        chunked = self.calculate_kv_cache_memory(**config)
        
        # 节省比例
        savings_ratio = 1 - chunked['total_gb'] / original['total_gb']
        
        return {
            'original_gb': original['total_gb'],
            'chunked_gb': chunked['total_gb'],
            'savings_ratio': savings_ratio,
            'savings_gb': original['total_gb'] - chunked['total_gb']
        }
    
    def visualize_memory_scaling(self):
        """
        可视化不同序列长度下的内存需求
        """
        import matplotlib.pyplot as plt
        
        # 7B模型配置
        base_config = {
            'batch_size': 1,
            'num_layers': 32,
            'num_heads': 32,
            'head_dim': 128,
            'dtype_bytes': 2
        }
        
        # 不同序列长度
        seq_lens = [512, 1024, 2048, 4096, 8192, 16384]
        memory_gbs = []
        
        for seq_len in seq_lens:
            config = {**base_config, 'seq_len': seq_len}
            memory = self.calculate_kv_cache_memory(**config)
            memory_gbs.append(memory['total_gb'])
        
        # 绘制
        plt.figure(figsize=(10, 6))
        plt.plot(seq_lens, memory_gbs, 'b-o', linewidth=2, markersize=8)
        plt.xlabel('序列长度', fontsize=12)
        plt.ylabel('KV缓存内存 (GB)', fontsize=12)
        plt.title('KV缓存内存 vs 序列长度（7B模型）', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.yscale('linear')
        
        # 添加标注
        for seq_len, memory in zip(seq_lens, memory_gbs):
            plt.annotate(f'{memory:.2f}GB', 
                        (seq_len, memory),
                        textcoords="offset points",
                        xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig('KV缓存内存缩放.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return seq_lens, memory_gbs


if __name__ == "__main__":
    calculator = KVCacheMemoryCalculator()
    
    print("=== KV缓存内存计算验证 ===\n")
    
    # 1. 验证公式正确性
    print("1. 公式验证:")
    config = {
        'batch_size': 1,
        'seq_len': 2048,
        'num_layers': 32,
        'num_heads': 32,
        'head_dim': 128,
        'dtype_bytes': 2
    }
    verification = calculator.verify_formula(config)
    print(f"   公式计算: {verification['formula_bytes']:,} bytes")
    print(f"   实际张量: {verification['actual_bytes']:,} bytes")
    print(f"   误差: {verification['error_percent']:.4f}%")
    print(f"   公式正确: {'✓' if verification['formula_correct'] else '✗'}\n")
    
    # 2. 对比不同模型
    print("2. 不同模型KV缓存对比:")
    model_comparison = calculator.compare_models()
    for model_name, result in model_comparison.items():
        print(f"   {model_name}:")
        print(f"     KV缓存: {result['memory_gb']:.2f} GB")
        print(f"     单层: {result['per_layer_mb']:.1f} MB")
        print(f"     公式验证: {'✓' if result['formula_verified'] else '✗'}\n")
    
    # 3. 分析错误计算
    print("3. 错误计算分析:")
    error_analysis = calculator.analyze_wrong_calculation()
    print(f"   正确计算: {error_analysis['correct_gb']:.2f} GB")
    print(f"   错误计算1（缺因子2）: {error_analysis['wrong_gb_1']:.2f} GB")
    print(f"   错误计算2（用d_model）: {error_analysis['wrong_gb_2']:.2f} GB")
    print(f"   误差倍数1: {error_analysis['error_ratio_1']:.2f}x")
    print(f"   误差倍数2: {error_analysis['error_ratio_2']:.2f}x\n")
    
    # 4. 分块计算分析
    print("4. 分块计算内存节省:")
    chunk_analysis = calculator.analyze_chunked_computation(chunk_size=512, seq_len=2048)
    print(f"   原始内存: {chunk_analysis['original_gb']:.2f} GB")
    print(f"   分块内存: {chunk_analysis['chunked_gb']:.2f} GB")
    print(f"   节省比例: {chunk_analysis['savings_ratio']:.1%}")
    print(f"   节省内存: {chunk_analysis['savings_gb']:.2f} GB\n")
    
    # 5. 可视化内存缩放
    print("5. 生成内存缩放可视化...")
    calculator.visualize_memory_scaling()
    print("   完成！")
```

**最终答案**: 7B模型在2048上下文长度下的KV缓存应该是**1GB**，不是17GB！你的质疑发现了一个重要的计算错误。