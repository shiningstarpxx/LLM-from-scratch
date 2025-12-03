# CS336 实验代码

本目录包含 Lecture 01-04 的所有实验代码，从零实现深度学习核心组件。

## 📁 目录结构

```
experiments/
├── README.md                    # 本文档
├── utils/                       # 通用工具
│   └── device_utils.py          # 设备检测（CUDA/MPS/CPU）
├── lecture01/                   # Tokenization
│   └── tokenization_demo.py     # BPE、词级、字符级分词
├── lecture02/                   # PyTorch基础
│   └── pytorch_basics.py        # Tensor、Autograd、FLOP、内存分析
├── lecture03/                   # Transformer
│   └── transformer_demo.py      # Attention、位置编码、KV Cache
└── lecture04/                   # MoE
    └── moe_demo.py              # Router、Expert、负载均衡
```

## 🖥️ 环境要求

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **支持平台**:
  - NVIDIA GPU (CUDA)
  - Apple Silicon (MPS)
  - CPU

## 📦 安装依赖

### MacBook (Apple Silicon / MPS)
```bash
# 使用pip安装PyTorch (MPS支持)
pip3 install torch torchvision torchaudio

# 验证MPS是否可用
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### NVIDIA GPU (CUDA)
```bash
# PyTorch + CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证CUDA是否可用
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### CPU Only
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 🚀 快速开始

### 1. 设备检测测试

```bash
cd experiments
python utils/device_utils.py
```

输出示例：
```
============================================================
🖥️  Device Configuration
============================================================
Selected Device: mps
Device Type: MPS
Platform: Apple Silicon (MPS Backend)
PyTorch MPS: Enabled
PyTorch Version: 2.1.0
Python Version: 3.10.12
============================================================
```

### 2. 运行各讲座实验

```bash
# Lecture 01: Tokenization
python lecture01/tokenization_demo.py

# Lecture 02: PyTorch Basics
python lecture02/pytorch_basics.py

# Lecture 03: Transformer
python lecture03/transformer_demo.py

# Lecture 04: MoE
python lecture04/moe_demo.py
```

## 📚 各讲座内容详解

### Lecture 01: Tokenization

**文件**: `lecture01/tokenization_demo.py`

**内容**:
- Character-level tokenization（字符级）
- Word-level tokenization（单词级）
- BPE (Byte Pair Encoding)（子词级）
- OOV处理对比
- 性能分析

**核心概念**:
```python
# BPE核心思想
1. 初始化为字符级
2. 统计最频繁的字符pair
3. 合并pair为新token
4. 重复直到达到目标词表大小

# 优势
- 平衡词表大小和序列长度
- 处理OOV能力强
- 学习morphology（词法）
```

---

### Lecture 02: PyTorch Basics

**文件**: `lecture02/pytorch_basics.py`

**内容**:
- Tensor创建与操作
- 自动微分（Autograd）
- FLOP计算器
- 内存分析器
- 简单训练循环

**核心概念**:
```python
# 自动微分
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
# x.grad = 4.0 (dy/dx = 2x)

# FLOP计算
MatMul [M, K] @ [K, N] = 2*M*K*N FLOPs
```

---

### Lecture 03: Transformer

**文件**: `lecture03/transformer_demo.py`

**内容**:
- Scaled Dot-Product Attention
- Multi-Head Attention
- Sinusoidal Position Encoding
- RoPE (Rotary Position Embedding)
- Feed-Forward Network
- 完整Transformer Block
- KV Cache

**核心概念**:
```python
# Attention公式
Attention(Q, K, V) = softmax(QK^T / √d_k) V

# Multi-Head
- 将Q, K, V投影到多个head
- 分别计算attention
- 合并输出

# KV Cache
- 缓存历史K和V
- 新token只需计算1次
- O(n²) → O(n) 计算复杂度
```

---

### Lecture 04: MoE (Mixture of Experts)

**文件**: `lecture04/moe_demo.py`

**内容**:
- Expert Network
- Top-K Router
- Load Balancing Loss
- Router Z-Loss
- Shared Expert (DeepSeek-V3 style)
- 完整MoE Layer
- MoE Transformer Block

**核心概念**:
```python
# MoE核心思想
- 多个Expert网络（FFN）
- Router选择top-k个expert
- 只激活选中的experts

# 效率优势
- 参数数量与计算解耦
- 64个experts，只激活2个
- 32x参数，2x计算

# Load Balancing Loss
L_balance = α * E * Σ(f_i * P_i)
# 确保experts被均匀使用
```

## 🔧 工具类说明

### DeviceManager

```python
from utils.device_utils import DeviceManager, get_device, to_device

# 方式1: 使用DeviceManager类
dm = DeviceManager()  # 自动检测最优设备
model = dm.to_device(model)
x = dm.to_device(x)

# 方式2: 使用快捷函数
device = get_device()
x = to_device(x)

# 指定偏好设备
dm = DeviceManager(prefer_device='cpu')  # 强制CPU
```

**功能**:
- 自动检测设备（CUDA > MPS > CPU）
- 统一的设备迁移接口
- 内存管理（cache清理）
- 同步操作（用于计时）

## 📊 预期输出示例

### Tokenization Demo
```
🔬 Tokenization 策略对比实验
...
📊 Performance Comparison
    Tokenizer            Vocab Size      Token Count     Compression
    -------------------------------------------------------------------
    Character-level      95              75              1.01x
    Word-level           125             11              6.82x
    BPE                  186             18              4.17x
```

### MoE Demo
```
🔀 Basic MoE Demo
...
4. Expert Usage (tokens routed to each expert):
   - Expert 0: 8 tokens (25.0%)
   - Expert 1: 9 tokens (28.1%)
   - Expert 2: 8 tokens (25.0%)
   - Expert 3: 7 tokens (21.9%)
```

## 💡 学习路径建议

1. **入门**: Lecture 01 (Tokenization) → 理解文本处理
2. **基础**: Lecture 02 (PyTorch) → 掌握深度学习工具
3. **核心**: Lecture 03 (Transformer) → 理解现代模型架构
4. **进阶**: Lecture 04 (MoE) → 学习扩展技术

## 🔗 相关资源

- [学习笔记](../学习笔记/)
- [深度讨论](../深度讨论/)
- [未来研究方向](../学习笔记/未来研究方向/)

## 📝 更新日志

| 日期 | 更新内容 |
|------|---------|
| 2025-12-02 | 创建实验代码目录，完成Lecture01-04基础实验 |

---

**Happy Learning!** 🎓
