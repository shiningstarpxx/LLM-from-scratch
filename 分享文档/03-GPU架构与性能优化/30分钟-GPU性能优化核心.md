# GPU性能优化核心 - 30分钟精华版

---

## 封面

### GPU架构与性能优化
**副标题**: 理解内存墙，掌握FlashAttention原理

**分享人**: [你的名字]
**时长**: 30分钟

---

## 议程

```
1. 为什么GPU优化重要？        (3 min)
2. 内存层次与瓶颈             (8 min)
3. Roofline模型               (8 min)
4. FlashAttention原理         (8 min)
5. 总结                       (3 min)
```

---

# Part 1: 为什么GPU优化重要？

---

## 残酷现实

```
A100理论算力: 312 TFLOP/s
实际利用率: 通常只有 10-30%

为什么？
```

---

## 答案: 内存墙

```
GPU计算能力增长: ~2x/2年
GPU内存带宽增长: ~1.3x/2年

差距越来越大!

结果: 计算单元在"等数据"
```

---

## 核心问题

> 不是算得不够快，
> 而是数据搬运不够快

**今天的目标**: 理解这个瓶颈，学会优化

---

# Part 2: 内存层次与瓶颈

---

## 2.1 GPU内存金字塔

```
         ┌─────────┐
         │Register │ ← 最快，最小
         │  ~256KB │   1 cycle
         └────┬────┘
              │
         ┌────┴────┐
         │ Shared  │ ← 程序员可控
         │ Memory  │   20 cycles
         │ ~100KB  │
         └────┬────┘
              │
         ┌────┴────┐
         │   HBM   │ ← 主显存
         │ 40-80GB │   400 cycles
         └─────────┘

速度差: SRAM比HBM快 20倍!
```

---

## 2.2 关键数字 (A100)

| 层级 | 容量 | 带宽 | 延迟 |
|------|------|------|------|
| Register | 256KB/SM | ~20 TB/s | 1 cycle |
| Shared Memory | 164KB/SM | ~19 TB/s | 20 cycles |
| L2 Cache | 40MB | ~5 TB/s | 200 cycles |
| HBM | 80GB | 2 TB/s | 400 cycles |

---

## 2.3 为什么这很重要？

```python
# 标准Attention操作
S = Q @ K.T        # 计算
save_to_HBM(S)     # 写HBM (慢!)
S = read_from_HBM(S)  # 读HBM (慢!)
P = softmax(S)     # 计算
save_to_HBM(P)     # 写HBM (慢!)
P = read_from_HBM(P)  # 读HBM (慢!)
O = P @ V          # 计算

# 大部分时间在等HBM读写!
```

---

# Part 3: Roofline模型

---

## 3.1 核心概念

### Arithmetic Intensity (AI)

$$
AI = \frac{\text{FLOPs}}{\text{Bytes访问}}
$$

**单位**: FLOP/Byte

**含义**: 每访问1字节数据，做多少次计算

---

## 3.2 Roofline图

```
                    │
    312 TFLOP/s ────┼─────────────────  Compute Roof
                    │               /
    Performance     │             /
                    │           /   Memory Roof
                    │         /
                    │       /
                    │     /
                    │   /
                    │ /
                    └────────────────────
                    0          208      AI
                           ↑
                      临界点
```

---

## 3.3 判断瓶颈

```
A100临界AI = 312 TFLOP/s / 1.5 TB/s = 208 FLOP/Byte

AI < 208: Memory-Bound (带宽瓶颈)
AI > 208: Compute-Bound (算力瓶颈)
```

---

## 3.4 常见操作的AI

| 操作 | AI (FLOP/Byte) | 瓶颈 |
|------|----------------|------|
| 矩阵乘法 (小) | ~50 | Memory |
| 矩阵乘法 (大) | ~500 | Compute |
| 标准Attention | ~30 | Memory |
| FlashAttention | ~1000+ | Compute |

**发现**: Attention天然是Memory-Bound!

---

## 3.5 优化目标

```
提高AI → 从Memory-Bound变成Compute-Bound

方法:
1. 减少内存访问 (数据复用)
2. 增加计算量 (算子融合)
3. 数据留在快速内存 (Tiling)
```

---

# Part 4: FlashAttention原理

---

## 4.1 标准Attention的问题

```python
# 标准流程
S = Q @ K.T         # [n, n] 写入HBM
P = softmax(S)      # 读S, 写P到HBM
O = P @ V           # 读P

# n=4096时:
# S和P各 4096² × 2 = 32MB
# 来回读写: 128MB HBM访问!
```

---

## 4.2 FlashAttention的解决方案

### 核心思想: Tiling + Online Softmax

```
不存储完整的n×n矩阵！

分块计算:
┌──────────────────┐
│  Q分块  │        │
├────────┼────────┤
│ K分块  │ 计算   │← 小块在SRAM中完成
├────────┼────────┤
│ V分块  │        │
└──────────────────┘
```

---

## 4.3 分块计算流程

```python
for j in range(0, n, block_size):
    Kj = K[j:j+block]  # 加载小块K到SRAM
    Vj = V[j:j+block]  # 加载小块V到SRAM

    for i in range(0, n, block_size):
        Qi = Q[i:i+block]  # 加载小块Q到SRAM

        # 在SRAM中完成所有计算!
        Sij = Qi @ Kj.T
        Pij = softmax(Sij)  # Online更新
        Oij = Pij @ Vj

# 从不存储完整n×n矩阵
```

---

## 4.4 Online Softmax

**问题**: 标准softmax需要全局信息

```python
softmax(x) = exp(x) / sum(exp(x))
# 需要知道所有x才能算sum
```

**解决**: 边算边更新

```python
# 维护最大值m和累积和l
m_new = max(m_old, block_max)
l_new = l_old * exp(m_old - m_new) + block_sum
```

---

## 4.5 效果对比

| 指标 | 标准Attention | FlashAttention |
|------|---------------|----------------|
| HBM访问 | O(n²) | O(n) |
| 内存占用 | O(n²) | O(n) |
| 速度 | 1x | 2-4x |

**关键纠正**: FlashAttention是"重算换存储"

---

## 4.6 为什么重算更快？

```
直觉上: 重算比存储慢?

实际上:
- HBM访问: 400 cycles
- SRAM计算: 20 cycles

重算20次 < 访问HBM 1次!

这就是FlashAttention的反直觉之处
```

---

# Part 5: 总结

---

## 核心要点

| 概念 | 一句话 |
|------|--------|
| 内存墙 | 计算快，搬数据慢 |
| SRAM vs HBM | 快20倍，但容量小 |
| AI (Arithmetic Intensity) | 计算量/内存访问 |
| Roofline | 判断瓶颈在计算还是内存 |
| FlashAttention | Tiling + Online Softmax |

---

## 关键数字

```
A100:
- 峰值算力: 312 TFLOP/s
- HBM带宽: 1.5 TB/s
- 临界AI: 208 FLOP/Byte
- SRAM比HBM快: 20倍

FlashAttention:
- 内存减少: O(n²) → O(n)
- 速度提升: 2-4倍
```

---

## 常见误区

| 误区 | 正确理解 |
|------|----------|
| GPU利用率高=性能好 | 可能在等内存 |
| FlashAttention空间换时间 | 实际是时间换空间 |
| 大矩阵总是慢 | 大矩阵AI更高，可能更快 |

---

## 下一步

- **深入版**: 60分钟完整解析
- **实践**: 用torch.profiler分析你的模型
- **论文**: FlashAttention 1 & 2

---

## Q&A

### Q: 什么时候需要FlashAttention？
**A**: 几乎总是需要。只要用Attention，就应该用FlashAttention。

### Q: FlashAttention有缺点吗？
**A**: 实现复杂，需要CUDA编程。好消息是PyTorch 2.0+已内置。

---

**感谢聆听！**
