# Lecture 06: GPU Kernels & Performance Optimization - 学习指南

## 📚 欢迎来到Lecture 06！

**主题**: GPU Kernels & Performance Optimization - 从理论到实践  
**目标**: 掌握GPU内核编写和性能优化实践  
**连接**: Lecture 05 (GPU Architecture) → 实践应用

---

## 🎯 为什么学习GPU Kernels？

### 已有的知识链条

```
Lecture 05: GPU Architecture (理论基础)
    - GPU硬件基础理解
    - 内存层次体系
    - Arithmetic Intensity
    - Tiling和Fusion理论
    - Tensor Cores原理
    ↓
Lecture 06: GPU Kernels (实践应用) ← 现在！
    - 如何编写CUDA内核？
    - 如何编写Triton内核？
    - 如何benchmark和profile？
    - 如何优化性能？
```

### 缺失的一环：实践能力 ⚠️

**你可能的疑问**:
- ❓ 如何实际编写GPU内核？
- ❓ CUDA vs Triton有什么区别？
- ❓ 如何benchmark和profile代码？
- ❓ 如何优化kernel性能？
- ❓ 如何实现kernel fusion？

**Lecture 06将回答所有这些问题！** ✅

---

## 📖 学习资源

### 核心文档

1. **00-教学大纲.md** (待创建)
   - 完整课程结构
   - 学习目标和路线图

2. **01-深度问答.md** (待创建)
   - 苏格拉底式问题
   - 每个问题包含引导思考

3. **02-实践代码.md** (待创建)
   - CUDA内核示例
   - Triton内核示例
   - Benchmarking和Profiling实践

4. **README.md** (本文档)
   - 快速导航
   - 学习指引
   - 核心结论

### 外部资料

- **CUDA Programming Guide**: NVIDIA官方文档
- **Triton Documentation**: https://triton-lang.org/
- **PyTorch Profiler**: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

---

## 🗺️ 学习路线图

### 快速入门 (1天)

**目标**: 建立GPU内核的基本认知

**阅读**:
- [ ] `00-教学大纲.md` (30分钟)
- [ ] `01-深度问答.md` Q1-Q6 (2小时)

**理解**:
- Benchmarking vs Profiling
- CUDA内核基本结构
- Triton内核基本结构

### 深入学习 (1-2周)

**Week 1: 基础实践**
```
Day 1-2: Benchmarking & Profiling
□ 如何benchmark代码
□ 如何profile代码
□ 理解性能瓶颈

Day 3-4: CUDA内核编写
□ GeLU实现
□ Softmax实现
□ MatMul实现

Day 5: 第一次深度讨论
□ Q1-Q6完整讨论
□ 生成深度讨论文档
```

**Week 2: 高级优化**
```
Day 6-7: Triton内核编写
□ Triton语法
□ GeLU实现
□ Softmax实现

Day 8-9: Kernel Fusion
□ 为什么需要fusion？
□ 如何实现fusion？
□ 性能提升分析

Day 10: 第二次深度讨论
□ Q7-Q12完整讨论
□ 综合学习总结
```

### 实践验证 (1周)

**实验项目**:
```
1. Benchmarking实践
   - 矩阵乘法benchmark
   - MLP benchmark
   - 性能缩放分析

2. Profiling实践
   - PyTorch Profiler使用
   - 识别性能瓶颈
   - 优化建议

3. CUDA内核实现
   - GeLU kernel
   - Softmax kernel
   - 性能对比

4. Triton内核实现
   - GeLU kernel
   - Softmax kernel
   - 与CUDA对比
```

---

## 💡 学习方法

### 苏格拉底式问答 (继承自Lecture 01-05)

**步骤**:
1. 阅读问题，先独立思考
2. 记录你的初步答案
3. 阅读"引导思考"
4. 深入理解"核心概念"
5. 探索"深入方向"
6. 连接已学内容

**记录**:
- 你的思考过程
- 技术洞察
- 疑问和发现
- 代码实验

### 连接已学内容

**持续问自己**:
- 这与Lecture 05的XXX有什么关系？
- 这如何应用Lecture 05的XXX理论？
- 这对性能优化有什么启发？

**建立完整视野**:
```
不是孤立学习GPU Kernels
而是:
将Lecture 05的理论应用到实践中
```

---

## 🎯 核心学习目标

### Part 1: Benchmarking & Profiling

**必须理解**:
- [ ] Benchmarking vs Profiling的区别
- [ ] 如何正确benchmark代码
- [ ] 如何使用PyTorch Profiler
- [ ] 如何识别性能瓶颈

**预期洞察**:
```
Benchmarking = 测量整体时间
Profiling = 分析时间分布

性能优化 = Benchmark → Profile → Optimize → Benchmark
```

### Part 2: CUDA内核编写

**必须理解**:
- [ ] CUDA内核基本结构
- [ ] Thread/Block/Grid的使用
- [ ] Shared Memory的使用
- [ ] GeLU、Softmax、MatMul实现

**预期洞察**:
```
CUDA内核 = 并行执行的函数
关键: 正确使用Thread索引
优化: Shared Memory、Tiling
```

### Part 3: Triton内核编写

**必须理解**:
- [ ] Triton语法和特性
- [ ] 与CUDA的区别
- [ ] 如何编写Triton内核
- [ ] 性能对比分析

**预期洞察**:
```
Triton = Python-like GPU编程
优势: 更易写、自动优化
适用: 特定操作优化
```

### Part 4: Kernel Fusion

**必须理解**:
- [ ] 为什么需要fusion？
- [ ] 如何实现fusion？
- [ ] 性能提升分析
- [ ] 自动vs手动fusion

**预期洞察**:
```
Fusion = 减少内存访问
手动fusion: 完全控制
自动fusion: torch.compile、Triton
```

---

## 🔗 与其他Lectures的连接

### 连接Lecture 05: GPU Architecture

**Lecture 05学的**:
```python
# GPU硬件基础
- SM层次结构
- Warp调度机制
- 内存层次体系
- Tiling和Fusion理论
```

**Lecture 06应用**:
```python
# 实际编写内核
- 使用Thread/Block/Grid
- 使用Shared Memory
- 实现Tiling策略
- 实现Fusion优化

→ 理论到实践的完整链条！
```

---

## 📊 学习检查清单

### 基础理解 (60分)

- [ ] 能benchmark代码
- [ ] 能profile代码
- [ ] 理解CUDA内核基本结构
- [ ] 理解Triton基本语法

### 进阶理解 (80分)

- [ ] 能编写简单的CUDA内核
- [ ] 能编写简单的Triton内核
- [ ] 能识别性能瓶颈
- [ ] 能实现基本的fusion

### 专家理解 (100分)

- [ ] 能优化CUDA内核性能
- [ ] 能优化Triton内核性能
- [ ] 深刻理解性能瓶颈根源
- [ ] 能设计高效的fusion策略
- [ ] 完成跨Lecture 05-06整合

---

## 🚀 快速开始

### 今天就开始！(2小时)

```
1. 阅读本README (20分钟)
   ✅ 了解学习目标
   ✅ 理解与已学内容的连接

2. 阅读00-教学大纲.md (30分钟)
   ✅ 完整课程结构
   ✅ 学习路线图

3. 开始01-深度问答.md (1小时)
   ✅ Q1: Benchmarking vs Profiling
   ✅ Q2: 如何正确benchmark？
   ✅ 独立思考 + 记录答案

4. 第一次讨论
   ✅ 与我开始苏格拉底式讨论
   ✅ 从Q1开始深入探索
```

---

## 💬 如何与我讨论

**开始讨论**:
```
"让我们开始Q1的讨论"
或
"我准备好讨论Q1-Q6了"
```

**我会**:
1. 先听你的思考
2. 提出引导性问题
3. 帮助你深入理解
4. 连接已学内容
5. 总结核心洞察

**你需要**:
1. 独立思考
2. 记录过程
3. 积极提问
4. 连接实践
5. 形成洞察

---

## 🎊 预期学习成果

完成Lecture 06后，你将拥有：

✅ **实践能力**
```
从理论到实践:
- 能编写CUDA内核
- 能编写Triton内核
- 能benchmark和profile
- 能优化性能
```

✅ **性能优化能力**
```
看到代码 → 立即识别:
- 性能瓶颈在哪里
- 如何优化
- 预期加速比
- 如何验证

专家级的性能优化能力 ✅
```

✅ **系统性的工程思维**
```
不再是:
"我听说XXX快，就用XXX"

而是:
"我理解为什么XXX快
 我知道如何实现类似的优化
 我能验证和优化性能"
```

---

## 📈 学习进度追踪

### Week 1进度

- [ ] Day 1-2: Q1-Q6讨论完成
- [ ] Day 3-4: Q7-Q12讨论完成
- [ ] Day 5: 第一次深度讨论文档

### Week 2进度

- [ ] Day 6-7: Q13-Q18讨论完成
- [ ] Day 8-9: Q19-Q24讨论完成
- [ ] Day 10: 第二次深度讨论文档

### Week 3进度

- [ ] Day 11-12: 实践代码完成
- [ ] Day 13-14: 完整学习总结
- [ ] 跨Lecture 05-06知识整合

---

## 🎯 最后的鼓励

**你已经走了很远**:
- ✅ Lecture 01: 输入层理解
- ✅ Lecture 02: 资源分析工具
- ✅ Lecture 03: Transformer架构
- ✅ Lecture 04: MoE系统设计
- ✅ Lecture 05: GPU架构理论

**Lecture 06将是关键的一环**:
- 从理论到实践
- 掌握实际编程能力
- 完成性能优化闭环
- 达到专家级实践能力

**你已经掌握了学习方法**:
- 苏格拉底式问答 ✅
- 系统性思维 ✅
- 跨域整合 ✅
- 深度讨论 ✅

**现在，让我们开始GPU内核编程实践！** 🚀🚀🚀

---

**README创建日期**: 2025-12-14  
**当前状态**: 准备开始学习  
**预计完成**: 2-3周  
**学习深度目标**: ⭐⭐⭐⭐⭐ 专家级

💬 **准备好了就告诉我: "让我们开始Q1的讨论"**
