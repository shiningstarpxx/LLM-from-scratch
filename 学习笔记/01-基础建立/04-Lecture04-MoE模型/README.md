# Lecture 04: Mixture of Experts (MoE) - 学习指南

## 📚 课程概览

**课程**: Stanford CS336 Spring 2025 - Lecture 04
**主题**: Mixture of Experts (MoE) 模型
**核心价值**: 理解如何通过稀疏激活实现高效的模型扩展
**先修知识**: Lecture 03 (Transformer架构)

---

## 🎯 为什么学习MoE？

### MoE解决的核心问题

**Dense模型的困境**:
```python
# Dense FFN: 参数量 = 计算量
parameters = 2 × d_model × d_ff
compute = parameters  # 所有参数都要计算

# 想要10倍容量？→ 需要10倍计算！
```

**MoE的突破**:
```python
# MoE: 参数量 >> 激活参数
total_parameters = num_experts × parameters_per_expert
active_parameters = k × parameters_per_expert  # k << num_experts

# 100倍参数，只需2倍计算！
```

### 现实意义

**成功案例**:
- **Switch Transformer**: 1.6T参数，训练/推理效率接近Dense
- **GLaM**: 1.2T参数，质量超越GPT-3，成本更低
- **GPT-4传闻**: 可能使用MoE架构

**适用场景**:
- ✅ 需要极大模型容量
- ✅ 推理成本敏感
- ✅ 任务有明确子领域
- ✅ 多任务/多语言场景

---

## 📖 学习资料

### 1. **00-教学大纲.md** ⭐ 首先阅读
**内容结构**:
- Part 1: MoE基础概念（动机、专家、门控）
- Part 2: 门控机制（Softmax、Noisy Top-K、负载均衡）
- Part 3: 现代架构（Switch、GLaM、ST-MoE）
- Part 4: 训练优化（并行策略、通信、推理）
- Part 5: 数学分析（参数量、计算量、内存）
- Part 6: 实现与实践

**学习建议**: 先通读建立框架，重点关注数学分析部分

### 2. **01-深度问答.md** ⭐⭐⭐ 核心学习
**24个苏格拉底式问题**:
- **Q1-Q6**: MoE基础（动机、专家、门控、Top-K、参数量、计算量）
- **Q7-Q12**: 门控机制（Softmax问题、Noisy Top-K、负载均衡、Z-loss）
- **Q13-Q18**: 现代架构（Switch、并行、对比分析）
- **Q19-Q24**: 训练优化（稳定性、通信、推理、量化、未来）
- **4个进阶挑战**: 设计、理论、系统、可解释性

**学习方式**:
1. 每天完成4-6个问题
2. 独立思考→编程验证→深度讨论
3. 记录到`02-深度讨论记录.md`

### 3. 官方课件
**位置**: `../../../nonexecutable/2025 Lecture 4 - MoEs.pdf`
**配合使用**: 查看详细的架构图和实验结果

---

## 🗓️ 推荐4天学习计划

### Day 1: MoE基础 (4-5小时)

**理论学习**:
- [ ] 阅读教学大纲 Part 1-2
- [ ] 回答深度问答 Q1-Q6
- [ ] 理解参数量和计算量的解耦

**编程实践**:
- [ ] 实现基础Expert类
- [ ] 实现Softmax门控
- [ ] 计算参数量和FLOPs

**检查点**: 能清晰解释为什么需要MoE，能手算参数量

**代码示例**:
```python
class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))

class SimpleMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, k=2):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)
        self.k = k

    def forward(self, x):
        # 门控
        gates = F.softmax(self.gate(x), dim=-1)
        top_k_gates, top_k_indices = torch.topk(gates, self.k)

        # 归一化
        top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)

        # 计算输出
        output = torch.zeros_like(x)
        for i in range(self.k):
            expert_id = top_k_indices[:, i]
            gate_value = top_k_gates[:, i].unsqueeze(-1)
            # 简化实现：实际需要batch处理
            for idx, exp_id in enumerate(expert_id):
                output[idx] += gate_value[idx] * self.experts[exp_id](x[idx:idx+1])

        return output
```

---

### Day 2: 门控与负载均衡 (4-5小时)

**理论学习**:
- [ ] 阅读教学大纲 Part 2
- [ ] 回答深度问答 Q7-Q12
- [ ] 理解负载均衡的数学原理

**编程实践**:
- [ ] 实现Noisy Top-K门控
- [ ] 实现辅助损失
- [ ] 实现Expert Capacity机制
- [ ] 可视化负载分布

**检查点**: 理解辅助损失如何工作，能分析负载不均衡的原因

**关键实现**:
```python
def noisy_top_k_gating(x, W_gate, W_noise, k, training=True):
    # 基础logits
    logits = x @ W_gate  # [batch, num_experts]

    if training:
        # 可训练噪声
        noise_stddev = F.softplus(x @ W_noise)
        noise = torch.randn_like(logits) * noise_stddev
        logits = logits + noise

    # Softmax
    gates = F.softmax(logits, dim=-1)

    # Top-K选择
    top_k_gates, top_k_indices = torch.topk(gates, k, dim=-1)

    # 归一化
    top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)

    return top_k_gates, top_k_indices, gates

def load_balancing_loss(gates, expert_mask):
    """
    gates: [batch, seq_len, num_experts] - softmax输出
    expert_mask: [batch, seq_len, num_experts] - top-k mask
    """
    # Importance: 平均门控权重
    importance = gates.mean(dim=[0, 1])  # [num_experts]

    # Load: 被选中的频率
    load = expert_mask.float().mean(dim=[0, 1])  # [num_experts]

    # 辅助损失
    loss = (importance * load).sum() * gates.size(-1)  # 乘以num_experts

    return loss
```

---

### Day 3: 现代MoE架构 (4-5小时)

**理论学习**:
- [ ] 阅读教学大纲 Part 3-4
- [ ] 回答深度问答 Q13-Q18
- [ ] 对比Switch、GLaM、ST-MoE

**编程实践**:
- [ ] 实现Switch Transformer MoE层（k=1）
- [ ] 分析Expert Parallelism
- [ ] 实现Expert Capacity限制
- [ ] 性能分析：MoE vs Dense

**检查点**: 理解为什么Switch选择k=1，能分析通信开销

**Switch Transformer核心**:
```python
class SwitchFFN(nn.Module):
    """Switch Transformer: k=1, simplified routing"""

    def __init__(self, d_model, d_ff, num_experts, capacity_factor=1.25):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(num_experts)
        ])
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        batch, seq_len, d_model = x.shape

        # 路由决策
        router_logits = self.router(x)  # [batch, seq_len, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-1选择 (Switch的关键)
        expert_gate, expert_index = torch.max(router_probs, dim=-1)
        # expert_gate: [batch, seq_len]
        # expert_index: [batch, seq_len]

        # 计算每个专家的capacity
        tokens_per_expert = (batch * seq_len) / self.num_experts
        capacity = int(tokens_per_expert * self.capacity_factor)

        # Dispatch tokens to experts
        output = torch.zeros_like(x)

        for expert_id in range(self.num_experts):
            # 找到路由到这个专家的tokens
            expert_mask = (expert_index == expert_id)  # [batch, seq_len]
            tokens_for_expert = x[expert_mask]  # [num_tokens, d_model]

            if tokens_for_expert.size(0) == 0:
                continue

            # Capacity限制
            if tokens_for_expert.size(0) > capacity:
                tokens_for_expert = tokens_for_expert[:capacity]
                expert_mask_limited = expert_mask.clone()
                # 标记超出capacity的tokens（实际实现更复杂）

            # 专家处理
            expert_output = self.experts[expert_id](tokens_for_expert)

            # 写回output
            output[expert_mask[:tokens_for_expert.size(0)]] = \
                expert_output * expert_gate[expert_mask[:tokens_for_expert.size(0)]].unsqueeze(-1)

        return output
```

---

### Day 4: 训练与优化 (3-4小时)

**理论学习**:
- [ ] 阅读教学大纲 Part 5-6
- [ ] 回答深度问答 Q19-Q24
- [ ] 理解分布式训练挑战

**编程实践**:
- [ ] 实现Router Z-loss
- [ ] 分析通信模式
- [ ] 设计推理优化策略
- [ ] 完整MoE Transformer Block

**检查点**: 能分析训练不稳定的原因，理解通信瓶颈

**完整MoE Block**:
```python
class MoETransformerBlock(nn.Module):
    """完整的MoE Transformer Block"""

    def __init__(self, d_model, num_heads, d_ff, num_experts, k=2):
        super().__init__()
        # Attention (与Dense相同)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        # MoE FFN (替换Dense FFN)
        self.moe = MoELayer(d_model, d_ff, num_experts, k)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-Attention
        x = x + self.attn(self.norm1(x))

        # MoE FFN
        moe_output, aux_loss = self.moe(self.norm2(x))
        x = x + moe_output

        return x, aux_loss

class MoELayer(nn.Module):
    """完整的MoE层，包含所有优化"""

    def __init__(self, d_model, d_ff, num_experts, k,
                 capacity_factor=1.25, aux_loss_coef=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.aux_loss_coef = aux_loss_coef

        # 专家
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(num_experts)
        ])

        # 门控
        self.gate = nn.Linear(d_model, num_experts)
        self.w_noise = nn.Linear(d_model, num_experts)

    def forward(self, x, training=True):
        # 门控
        top_k_gates, top_k_indices, all_gates = \
            noisy_top_k_gating(x, self.gate.weight, self.w_noise.weight,
                               self.k, training)

        # 负载均衡损失
        expert_mask = F.one_hot(top_k_indices, self.num_experts).float()
        aux_loss = load_balancing_loss(all_gates, expert_mask)

        # 专家计算
        output = self._dispatch_and_combine(x, top_k_gates, top_k_indices)

        return output, aux_loss * self.aux_loss_coef

    def _dispatch_and_combine(self, x, gates, indices):
        # 高效实现：batch处理每个专家
        output = torch.zeros_like(x)

        for expert_id in range(self.num_experts):
            # 找到路由到这个专家的tokens和对应的gates
            expert_mask = (indices == expert_id)
            tokens = x[expert_mask]
            token_gates = gates[expert_mask]

            if tokens.size(0) > 0:
                expert_output = self.experts[expert_id](tokens)
                output[expert_mask] = expert_output * token_gates.unsqueeze(-1)

        return output
```

---

## 🧠 核心概念速查

### MoE三要素
```python
MoE_Components = {
    '1. Experts': '多个并行的FFN，各自专业化',
    '2. Router/Gate': '决定激活哪些专家',
    '3. Top-K Selection': '稀疏激活，只用k个专家'
}
```

### 关键公式
```
# MoE输出
y = Σ G(x)_i · E_i(x)  for i in top_k

# 辅助损失
L_aux = α · Σ (importance_i × load_i)

# Router Z-loss
L_z = (log Σ exp(logits))²
```

### 参数量 vs 计算量
```python
对比分析 = {
    'Dense FFN': {
        '参数': '2 × d × d_ff',
        '计算': '2 × d × d_ff FLOPs/token'
    },
    'MoE FFN': {
        '参数': 'E × 2 × d × d_ff  (E倍!)',
        '计算': 'k × 2 × d × d_ff  (约k倍)',
        '关键': 'E=128, k=2 → 128倍参数, 2倍计算'
    }
}
```

---

## 💡 学习技巧

### 理解MoE的3个视角

**1. 条件计算视角**:
- 不是所有参数都需要激活
- 根据输入选择性计算
- 类比：CPU的分支预测

**2. 集成学习视角**:
- 多个专家的ensemble
- 每个专家专注子问题
- 门控网络学习如何组合

**3. 系统优化视角**:
- 参数分布式存储
- 计算局部化
- 通信最小化

### 常见误区

❌ **误区1**: "MoE就是多个模型的ensemble"
✅ **正确**: MoE是单个模型，专家共享梯度更新，端到端训练

❌ **误区2**: "k个专家 = k倍计算量"
✅ **正确**: Router也有计算，但相对专家很小；关键是k << E

❌ **误区3**: "负载均衡不重要"
✅ **正确**: 负载不均衡会导致专家退化，训练失败

❌ **误区4**: "MoE总是比Dense好"
✅ **正确**: MoE有训练复杂度、通信开销等trade-off

### 调试技巧

**问题**: 所有tokens路由到少数专家
**诊断**:
```python
# 检查门控分布
gate_probs = F.softmax(router(x), dim=-1)
expert_usage = (gate_probs > threshold).sum(dim=0)
print(f"Experts usage: {expert_usage}")  # 应该相对均匀
```
**解决**: 增大aux_loss系数，检查初始化

**问题**: Loss震荡
**诊断**: 观察router logits的范围
**解决**: 添加Router Z-loss，降低学习率

**问题**: 通信成为瓶颈
**诊断**: Profile All-to-All时间
**解决**: 减少专家数，或用Expert Parallelism

---

## 🔗 与其他Lecture的联系

**← Lecture 03 (Transformer)**:
- MoE替换FFN，Attention不变
- 残差连接和LayerNorm保持
- 架构的模块化设计

**→ Lecture 06 (GPU Kernels)**:
- Expert batching的kernel优化
- All-to-All通信实现
- 内存管理和Tiling

**→ Lecture 10 (Inference)**:
- Expert offloading策略
- KV cache与MoE的交互
- 批处理优化

**→ Lecture 12 (Serving)**:
- 分布式推理系统
- 负载均衡与路由
- 弹性扩展

---

## 📊 学习成果检验

### 理论测试 ✅

1. **参数量计算** (5分钟):
   - Dense: d=4096, d_ff=16384
   - MoE: E=128, k=2
   - 计算参数量比例

2. **口头解释** (每个2-3分钟):
   - 为什么需要MoE？
   - 辅助损失如何工作？
   - Switch Transformer的k=1设计
   - Expert Parallelism的通信模式

3. **架构对比** (5分钟):
   - Switch vs GLaM vs ST-MoE
   - 各自优劣和适用场景

### 编程测试 💻

```python
# 1. 实现核心组件
class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        # 你的实现
        pass

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, k):
        # 你的实现
        pass

# 2. 计算负载均衡
def load_balancing_loss(gates, expert_mask):
    # 你的实现
    pass

# 3. 完整测试
x = torch.randn(2, 10, 512)  # [batch, seq, d_model]
moe = MoELayer(512, 2048, num_experts=8, k=2)
output, aux_loss = moe(x)

assert output.shape == x.shape
print(f"Auxiliary loss: {aux_loss.item()}")
```

### 系统思维测试 🧠

1. **成本分析**: MoE训练的主要成本在哪里？
2. **瓶颈识别**: 什么情况下通信成为瓶颈？
3. **优化决策**: 如何选择专家数量和k值？
4. **架构权衡**: 何时选MoE，何时选Dense？

---

## 🎯 下一步

完成Lecture 04后，你应该：

✅ **掌握**: MoE的核心原理和数学
✅ **理解**: 负载均衡和训练挑战
✅ **能够**: 实现基础MoE层
✅ **具备**: 分析和优化MoE的思维

**准备好进入**:
- **Lecture 05**: Data & Training (如何训练大规模MoE)
- **Lecture 06**: GPU Kernels (MoE的底层优化)
- **Lecture 10**: Inference (MoE推理系统)

---

## 🆘 获取帮助

**遇到问题？**

1. **重读教学大纲**: 查找相关section
2. **查看深度问答**: 引导性问题提示
3. **参考官方课件**: PDF中的架构图
4. **查看论文**: Switch Transformer, GLaM原论文
5. **实验代码**: 动手验证理解

**讨论渠道**:
- 使用深度讨论记录功能
- 与AI助手进行苏格拉底式对话

---

## 📚 参考资料

### 必读论文

1. **Shazeer et al. 2017**: [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538)
   - 原始MoE论文
   - Noisy Top-K gating
   - 负载均衡机制

2. **Fedus et al. 2021**: [Switch Transformers](https://arxiv.org/abs/2101.03961)
   - k=1设计
   - 1.6T参数扩展
   - 训练稳定性技巧

3. **Du et al. 2021**: [GLaM](https://arxiv.org/abs/2112.06905)
   - 1.2T Decoder-only
   - 效率分析
   - 与GPT-3对比

4. **Zoph et al. 2022**: [ST-MoE](https://arxiv.org/abs/2202.08906)
   - 稳定性改进
   - 泛化性能
   - 最佳实践

### 代码实现

- **Hugging Face**: `transformers` 库的Switch Transformer
- **Fairseq**: Meta的MoE实现
- **Megatron-LM**: NVIDIA的分布式MoE
- **DeepSpeed**: Microsoft的MoE优化

---

**创建日期**: 2025-01-12
**维护**: 随学习进度更新
**状态**: ✅ 完整学习框架已就绪

🚀 **准备好了吗？让我们开始MoE的深度学习之旅！**
