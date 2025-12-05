# Lecture 05: GPU Architecture - 学习指南

## 📚 欢迎来到Lecture 05！

**主题**: GPU Architecture - 深度学习的硬件基础  
**目标**: 从硬件视角理解性能优化  
**连接**: Lecture 02 (Resource) + Lecture 03 (FlashAttention) + Lecture 04 (MoE)

---

## 🎯 为什么学习GPU架构？

### 已有的知识链条

```
Lecture 01: Tokenization (输入层)
    ↓
Lecture 02: Resource Accounting (工具层)
    - 内存计算: 7B模型 = 132GB
    - FLOP分析: 矩阵乘法主导
    - 内存墙概念 ⚠️
    ↓
Lecture 03: Transformer (架构层)
    - FlashAttention: 为什么快？
    - KV Cache: 内存优化
    - O(n²)复杂度挑战
    ↓
Lecture 04: MoE (扩展层)
    - All-to-All通信瓶颈
    - Expert Offloading
    - 量化策略
```

### 缺失的一环：硬件视角 ⚠️

**你可能的疑问**:
- ❓ 为什么FlashAttention能加速2-4倍？
- ❓ HBM vs SRAM到底差多少？
- ❓ 混合精度训练的硬件支持是什么？
- ❓ Tensor Cores如何工作？
- ❓ 为什么Tiling能优化内存？

**Lecture 05将回答所有这些问题！** ✅

---

## 📖 学习资源

### 核心文档

1. **00-教学大纲.md**
   - 完整课程结构
   - 学习目标和路线图
   - 4个部分24个问题

2. **01-深度问答.md** ⭐ 开始这里
   - 24个苏格拉底式问题
   - 每个问题包含:
     - 引导思考
     - 核心概念
     - 深入方向
     - 连接已学内容

3. **02-深度讨论记录.md** (待生成)
   - 完整讨论过程
   - 技术洞察沉淀
   - 思维演进追踪

4. **03-实验代码.py** (待编写)
   - CUDA基础实验
   - 性能profiling
   - 可视化分析

5. **README.md** (本文档)
   - 快速导航
   - 学习指引
   - 核心结论

### 外部资料

- **PDF**: `nonexecutable/2025 Lecture 5 - GPUs.pdf`
- **CUDA Programming Guide**: NVIDIA官方文档
- **FlashAttention论文**: 硬件视角重读

---

## 🗺️ 学习路线图

### 快速入门 (1天)

**目标**: 建立GPU的基本认知

**阅读**:
- [ ] `00-教学大纲.md` (30分钟)
- [ ] `01-深度问答.md` Q1-Q6 (2小时)

**理解**:
- GPU vs CPU的根本差异
- Warp调度机制
- Thread层次结构

### 深入学习 (1-2周)

**Week 1: 基础架构**
```
Day 1-2: Part 1 (Q1-Q6) - GPU基础
□ 并行机制
□ SM组织
□ Warp概念

Day 3-4: Part 2 (Q7-Q12) - 内存层次
□ HBM vs SRAM
□ Shared Memory
□ 访问优化

Day 5: 第一次深度讨论
□ Q1-Q12完整讨论
□ 生成深度讨论文档
□ 连接Lecture 02
```

**Week 2: 性能优化**
```
Day 6-7: Part 3 (Q13-Q18) - 计算与带宽
□ Roofline Model
□ FlashAttention硬件解析
□ Tiling和Fusion

Day 8-9: Part 4 (Q19-Q24) - 高级优化
□ Occupancy优化
□ Tensor Cores
□ 未来趋势

Day 10: 第二次深度讨论
□ Q13-Q24完整讨论
□ 综合学习总结
□ 跨Lecture整合
```

### 实践验证 (1周)

**实验项目**:
```
1. CUDA Hello World
   - 理解kernel launch
   - Grid/Block/Thread

2. 向量加法优化
   - Coalesced access
   - Shared Memory使用

3. 矩阵乘法
   - Tiling实现
   - 性能profiling

4. FlashAttention原型
   - 理解Tiling设计
   - 性能对比
```

---

## 💡 学习方法

### 苏格拉底式问答 (继承自Lecture 01-04)

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
- 这与Lecture 02的XXX有什么关系？
- 这如何解释Lecture 03的XXX？
- 这对Lecture 04的XXX有什么启发？

**建立完整视野**:
```
不是孤立学习GPU
而是:
GPU如何支持我已经学过的所有技术
```

---

## 🎯 核心学习目标

### Part 1: GPU基础架构 (Q1-Q6)

**必须理解**:
- [ ] GPU为什么适合深度学习？
- [ ] Warp是什么？为什么重要？
- [ ] Thread/Block/Grid的层次
- [ ] SIMT vs SIMD的区别
- [ ] 延迟隐藏机制

**预期洞察**:
```
GPU = 大规模并行的专用处理器
核心: 数千个简单核心
vs CPU: 少数复杂核心

深度学习 = 矩阵乘法
→ 天然并行
→ GPU完美匹配 ✅
```

### Part 2: GPU内存层次 (Q7-Q12)

**必须理解**:
- [ ] GPU内存层次结构
- [ ] HBM为什么重要？
- [ ] Shared Memory vs L1 Cache
- [ ] Bank Conflict
- [ ] Coalesced Access
- [ ] Register分配

**预期洞察**:
```
内存层次 = 速度vs容量的权衡
HBM = 瓶颈！
优化目标: 减少HBM访问

FlashAttention的秘密:
用Shared Memory代替HBM ✅
```

### Part 3: 计算与带宽 (Q13-Q18)

**必须理解**:
- [ ] Arithmetic Intensity概念
- [ ] Compute-bound vs Memory-bound判断
- [ ] Roofline Model使用
- [ ] FlashAttention硬件视角
- [ ] Tiling的本质
- [ ] Fusion的重要性

**预期洞察**:
```
性能瓶颈 = 内存带宽（大多数情况）
优化策略:
1. 提高AI (Tiling)
2. 减少访问 (Fusion)
3. 数据复用 (Shared Memory)

FlashAttention = 这些技术的完美应用 ✅
```

### Part 4: 优化技术 (Q19-Q24)

**必须理解**:
- [ ] Occupancy如何影响性能
- [ ] Warp Divergence的代价
- [ ] Tensor Cores工作原理
- [ ] 混合精度硬件支持
- [ ] Async Copy机制
- [ ] 未来GPU架构趋势

**预期洞察**:
```
现代GPU优化 = 多维度权衡
- Occupancy vs 资源使用
- 分支 vs 计算
- 精度 vs 性能

Tensor Cores = 深度学习专用硬件
→ 必须使用！ ✅

未来 = 硬件-软件协同设计
```

---

## 🔗 与其他Lectures的连接

### 连接Lecture 02: Resource Accounting

**Lecture 02学的**:
```python
# 7B模型内存需求
params = 7B × 2 bytes = 14 GB
grads = 7B × 2 bytes = 14 GB
adam = 7B × 8 bytes = 56 GB
activations = 48 GB
───────────────────────────
Total = 132 GB

问题: 为什么activations这么大？
```

**Lecture 05解释**:
```
Activations主要在HBM中
每层: Attention矩阵 + FFN中间值

Attention: [batch, heads, seq, seq]
32 × 8 × 2048 × 2048 × 2 = 8GB per layer

24层 → 192 GB (理论)
→ 梯度检查点 → 48 GB ✅

现在完全理解了！
```

### 连接Lecture 03: Transformer

**Lecture 03学的**:
```python
# FlashAttention为什么快？
# 算法层面: Tiling
# 但为什么Tiling能加速？
```

**Lecture 05解释**:
```
硬件视角:
HBM: 1.5 TB/s, 400 cycles延迟
SRAM: 100+ TB/s, 20 cycles延迟

Tiling策略:
- 数据从HBM加载到SRAM一次
- 在SRAM中完成所有计算
- 只写最终结果回HBM

减少HBM访问1000倍！
→ 2-4倍实际加速 ✅

数学:
AI提升: 50 → 5000 FLOP/Byte
从memory-bound → compute-bound
```

### 连接Lecture 04: MoE

**Lecture 04学的**:
```python
# All-to-All通信是MoE瓶颈
# Expert Offloading策略
# 量化: FP16 → INT4
```

**Lecture 05解释**:
```
All-to-All:
- GPU间通信: NVLink vs PCIe
- 带宽: 600 GB/s vs 64 GB/s
- 延迟: 1-2μs vs 10μs

Offloading:
- GPU HBM ↔ CPU DRAM ↔ SSD
- 每层都有硬件支持

量化:
- Tensor Cores支持INT8/INT4
- 硬件加速: 2-4倍
- 混合精度训练: TF32/FP16

完整的硬件支持链条 ✅
```

---

## 📊 学习检查清单

### 基础理解 (60分)

- [ ] 能解释GPU vs CPU的区别
- [ ] 理解Warp的概念
- [ ] 知道GPU内存层次
- [ ] 了解基本优化技术

### 进阶理解 (80分)

- [ ] 能计算Arithmetic Intensity
- [ ] 会使用Roofline Model
- [ ] 理解FlashAttention的硬件本质
- [ ] 掌握Tiling和Fusion原理

### 专家理解 (100分)

- [ ] 能设计硬件感知的算法
- [ ] 深刻理解性能瓶颈根源
- [ ] 建立硬件-软件协同思维
- [ ] 能优化实际CUDA代码
- [ ] 完成跨Lecture 02-05整合

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
   ✅ Q1: 为什么GPU适合深度学习？
   ✅ Q2: GPU如何实现大规模并行？
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

完成Lecture 05后，你将拥有：

✅ **完整的知识链条**
```
输入 (Lecture 01)
  ↓
资源分析 (Lecture 02)
  ↓
算法设计 (Lecture 03)
  ↓
系统扩展 (Lecture 04)
  ↓
硬件基础 (Lecture 05) ← 闭环！

从软件到硬件的端到端理解 ✅
```

✅ **硬件感知的优化能力**
```
看到算法 → 立即识别:
- 是compute-bound还是memory-bound
- 瓶颈在哪里
- 如何优化
- 预期加速比

专家级的性能分析能力 ✅
```

✅ **系统性的工程思维**
```
不再是:
"我听说XXX快，就用XXX"

而是:
"基于硬件特性，我理解为什么XXX快
 我知道如何设计类似的优化
 我能预判在我的场景下的效果"

真正的工程专家 ✅
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

- [ ] Day 11-12: 实验代码完成
- [ ] Day 13-14: 完整学习总结
- [ ] 跨Lecture 02-05知识整合

---

## 🎯 最后的鼓励

**你已经走了很远**:
- ✅ Lecture 01: 输入层理解
- ✅ Lecture 02: 资源分析工具
- ✅ Lecture 03: Transformer架构
- ✅ Lecture 04: MoE系统设计

**Lecture 05将是关键的一环**:
- 解答所有"为什么"
- 建立硬件视角
- 完成知识闭环
- 达到专家级理解

**你已经掌握了学习方法**:
- 苏格拉底式问答 ✅
- 系统性思维 ✅
- 跨域整合 ✅
- 深度讨论 ✅

**现在，让我们探索GPU的硬件世界！** 🚀🚀🚀

---

**README创建日期**: 2025-11-30  
**当前状态**: 准备开始学习  
**预计完成**: 2-3周  
**学习深度目标**: ⭐⭐⭐⭐⭐ 专家级

💬 **准备好了就告诉我: "让我们开始Q1的讨论"**
