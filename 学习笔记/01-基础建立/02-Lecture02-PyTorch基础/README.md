# Lecture 02: PyTorch Building Blocks & Resource Accounting

## 📚 学习指南

本讲座是深度学习系统优化的基础，专注于理解PyTorch的核心构建块和资源计算方法。这是后续学习GPU架构、分布式训练等高级主题的必备基础。

---

## 🎯 学习目标

### ✅ 核心概念掌握
- **内存层次结构**: 理解CPU/GPU内存架构对深度学习的影响
- **FLOP计算**: 精确分析模型的计算复杂度
- **资源账务**: 预估训练和推理的资源需求
- **训练循环**: 构建高效、稳健的训练流程
- **性能优化**: 掌握常用的优化技巧和权衡

### 🔧 实践技能
- 使用工具计算模型FLOP和内存需求
- 实现完整的训练循环（混合精度、梯度累积等）
- 诊断和解决性能瓶颈
- 进行系统级性能优化

---

## 📁 文件结构

```
02-Lecture02-PyTorch基础/
├── 01-理论概念.md          # 核心理论和概念详解
├── 02-实践代码.md          # 完整的代码实现和演示
├── 03-深度问答.md          # 苏格拉底式问答引导
├── README.md               # 本文件
└── (待创建) 实践项目/       # 动手实践项目
```

---

## 🚀 学习路径

### 📖 第一阶段：理论基础 (1-2天)

1. **阅读理论概念**
   - 重点理解内存层次和FLOP计算
   - 做笔记，画图理解关键概念

2. **思考深度问答**
   - 逐个回答问题，不要急于看答案
   - 写下你的思考过程

### 🔧 第二阶段：实践编码 (2-3天)

1. **运行实践代码**
   ```bash
   # 运行完整演示
   python 02-实践代码.py
   ```

2. **修改实验**
   - 改变模型大小，观察FLOP变化
   - 测试不同优化技术的效果
   - 分析性能瓶颈

### 🎯 第三阶段：项目实践 (3-4天)

1. **选择一个实际项目**
   - 分析现有模型的资源需求
   - 优化训练流程
   - 实现性能监控

2. **深入某个方向**
   - 混合精度训练优化
   - 内存使用分析
   - 分布式训练准备

---

## 🧠 核心概念速览

### 💾 内存计算公式

```python
# 模型参数内存
param_memory = num_parameters × bytes_per_parameter

# 梯度内存 (训练)
grad_memory = param_memory

# 优化器内存 (Adam)
optimizer_memory = param_memory × 2

# 激活内存 (估算)
activation_memory = batch_size × sequence_length × hidden_dim × layers × bytes_per_parameter

# KV缓存 (推理)
kv_cache = batch_size × sequence_length × layers × hidden_dim × 2 × bytes_per_parameter
```

### 📊 FLOP计算公式

```python
# 矩阵乘法: (m,k) × (k,n)
matmul_flops = 2 × m × k × n

# 注意力机制
attention_flops = 4 × batch_size × sequence_length² × hidden_dim

# Transformer层 (每token)
transformer_flops_per_token = 2 × num_layers × hidden_dim × (hidden_dim + 4×hidden_dim)
```

### ⚡ 优化技巧效果

| 技术 | 内存节省 | 速度提升 | 适用场景 |
|------|---------|---------|---------|
| 混合精度 | ~50% | ~2-3x | GPU训练 |
| 梯度累积 | 虚拟大批次 | 轻微开销 | 显存不足 |
| 梯度检查点 | ~50%激活内存 | ~20%重计算 | 内存瓶颈 |
| 算子融合 | 轻微 | ~10-30% | 推理优化 |

---

## 🎭 学习方法建议

### 🤔 苏格拉底式学习

本课程特别强调**苏格拉底式问答方法**：

1. **不直接给答案**: 通过引导性问题激发思考
2. **建立思维框架**: 培养系统性思考能力
3. **联系实际应用**: 理论与实践相结合
4. **鼓励质疑精神**: 挑战现有假设

### 📝 学习策略

1. **主动学习**: 不要passive reading，要active thinking
2. **实践验证**: 理论要过手，代码要跑通
3. **定期反思**: 每周总结学习心得
4. **讨论交流**: 与同学、同事讨论加深理解

### 🎯 学习检查

每学习完一个部分，问自己：
- 我能用自己的话解释这个概念吗？
- 我能在实际项目中应用吗？
- 我理解这个技术的局限性和适用场景吗？

---

## 🔗 相关资源

### 📚 推荐阅读

1. **PyTorch官方文档**
   - [Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
   - [CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)

2. **深度学习系统**
   - 《Deep Learning Systems》教材
   - Stanford CS231n, CS224n课程材料

3. **性能分析工具**
   - [PyTorch Profiler](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)
   - [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)

### 🛠️ 实用工具

```bash
# FLOP计算库
pip install fvcore ptflops thop

# 性能分析
pip install torch-tb-profiler
pip install nvidia-ml-py3

# 实验跟踪
pip install wandb tensorboard
```

### 📊 基准测试

- [MLPerf Training Benchmark](https://mlcommons.org/benchmarks/training/)
- [HuggingFace模型FLOP计算器](https://huggingface.co/spaces/hf-audio/openai-whisper-large-v2-flops-calculator)

---

## ⚠️ 常见陷阱

### 🚫 学习误区

1. **重理论轻实践**: 只看不懂，不会应用
2. **盲目追求优化**: 过早优化，没有瓶颈分析
3. **忽视硬件特性**: 不理解底层，优化效果差
4. **缺乏系统思维**: 只看局部，不看整体

### 💡 避免方法

1. **理论实践结合**: 每个概念都要代码验证
2. **先测量后优化**: 用数据说话，不要猜测
3. **理解硬件原理**: 知其然，更要知其所以然
4. **培养系统观**: 从端到端角度思考问题

---

## 🎯 学习成果

完成本讲座学习后，你应该能够：

### ✅ 技术能力
- [ ] 精确计算任意模型的FLOP和内存需求
- [ ] 实现高效的生产级训练循环
- [ ] 诊断和解决常见的性能问题
- [ ] 选择合适的优化技术

### 🧠 思维能力
- [ ] 具备系统级优化思维
- [ ] 理解硬件与算法的相互作用
- [ ] 能够权衡不同技术方案的利弊
- [ ] 培养工程化的思考方式

### 🔬 实践能力
- [ ] 独立完成模型性能分析
- [ ] 设计高效的训练流程
- [ ] 解决实际项目中的性能问题
- [ ] 为后续高级主题打下基础

---

## 🎉 下一步预告

完成Lecture 02后，你将具备：

1. **坚实基础**: 理解深度学习系统的核心原理
2. **实用技能**: 掌握性能优化和资源管理
3. **系统思维**: 培养工程化的思考模式

这为学习以下高级主题做好了准备：
- **Lecture 03**: Transformer Architecture
- **Lecture 05**: GPU Architecture
- **Lecture 07**: Parallelism Basics
- **Lecture 09**: Scaling Laws

---

**💡 提示**: 这是深度学习系统课程的转折点，从理论概念转向工程实践。打好这个基础，后续学习会更加顺畅！