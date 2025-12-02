# Lecture 03: Attention可视化项目指南

## 🎨 项目概述

**项目目标**: 可视化Transformer中的Attention模式，理解不同layers和heads学到的关系类型  
**学习价值**: 从抽象数学到直观理解  
**完成时间**: 1-2小时  
**状态**: 理论设计完成，待实践

---

## 🎯 可视化内容

### 1. Attention Weights Heatmap

**目标**: 可视化attention权重矩阵

**数据**:
```python
attention_weights: [batch, heads, seq_len, seq_len]
# 每个元素表示query_i对key_j的注意力权重
```

**可视化示例** (理论设计):

```
示例句子: "The cat sat on the mat"
Tokens:    [The, cat, sat, on, the, mat]

Head 1 - 语法关系:
       The  cat  sat  on  the  mat
The   [0.1  0.2  0.1  0.1  0.4  0.1]  注意"the"(共指)
cat   [0.1  0.3  0.4  0.1  0.0  0.1]  注意"sat"(主谓)
sat   [0.1  0.3  0.2  0.2  0.1  0.1]  注意"cat"和"on"
on    [0.0  0.1  0.2  0.2  0.1  0.4]  注意"mat"(介宾)
the   [0.1  0.1  0.1  0.1  0.5  0.1]  注意前面的"The"
mat   [0.0  0.1  0.1  0.5  0.1  0.2]  注意"on"(被修饰)

观察: 主谓宾、介宾关系 (语法) ✅

Head 2 - 位置关系:
       The  cat  sat  on  the  mat
The   [0.8  0.1  0.0  0.0  0.0  0.1]  主要注意自己
cat   [0.4  0.4  0.2  0.0  0.0  0.0]  注意附近
sat   [0.1  0.3  0.4  0.2  0.0  0.0]  注意附近
on    [0.0  0.1  0.2  0.5  0.2  0.0]  注意附近
the   [0.0  0.0  0.1  0.2  0.6  0.1]  注意附近
mat   [0.0  0.0  0.0  0.1  0.3  0.6]  注意附近+自己

观察: 局部attention (位置) ✅

Head 3 - 语义关系:
       The  cat  sat  on  the  mat
The   [0.1  0.1  0.1  0.1  0.5  0.1]  "the"关联
cat   [0.0  0.2  0.1  0.0  0.1  0.6]  "cat"-"mat"(都是名词)
sat   [0.1  0.4  0.3  0.2  0.0  0.0]  动词关注主语
on    [0.0  0.0  0.1  0.2  0.0  0.7]  介词关注宾语
the   [0.4  0.1  0.0  0.0  0.4  0.1]  冠词关联
mat   [0.0  0.6  0.0  0.0  0.1  0.3]  名词关联

观察: 词性、语义类别 (语义) ✅

不同heads学习不同类型的关系！ ✅✅✅
```

**可视化效果** (Heatmap):
```
每个head一个热力图
颜色: 白色(0) → 红色(1)
对角线: 自注意力
非对角线: 跨token注意力

┌───────────────────┐
│██▒░░░│ Head 1    │  深色=高权重
│▒██▒░░│ (语法)    │  浅色=低权重
│░▒██▒░│           │
│░░▒██▒│           │
│░░░▒██│           │
│░░░░▒█│           │
└───────────────────┘
```

---

### 2. Layer-wise Attention Evolution

**目标**: 观察不同层的attention模式演变

**预期模式**:

**Layer 1 (浅层)**: 主要关注局部
```
- 对角线附近权重高
- 局部语法关系
- 词性标注相关
```

**Layer 12 (中层)**: 语法结构
```
- 主谓宾关系清晰
- 依存关系显现
- 长距离依赖开始
```

**Layer 24 (深层)**: 语义关系
```
- 主题相关token关联
- 共指消解
- 抽象语义关系
```

**可视化**: 多层并排对比
```
Layer 1    Layer 12   Layer 24
[热力图]   [热力图]   [热力图]
 局部      →  语法   →   语义
```

---

### 3. Head Importance Analysis

**目标**: 哪些heads更重要？

**分析方法**:
```python
# 方法1: Attention Entropy
H(head) = -Σ p_ij × log(p_ij)

高熵: attention分散 (关注多个tokens)
低熵: attention集中 (关注少数tokens)

# 方法2: Head Pruning
逐个删除head，测量性能下降

重要head: 性能下降大 ⚠️
不重要head: 性能下降小 ✅

# 方法3: Gradient-based
测量head输出对loss的梯度
梯度大 → 重要 ⚠️
```

**预期发现**:
```
8个heads中:
- 2-3个head非常重要 (语法核心) ⚠️
- 3-4个head中等重要 (语义辅助) ⚠️
- 1-2个head可以删除 (冗余) ✅

这解释了为什么Head Pruning有效！
```

---

### 4. Causal Mask可视化

**目标**: 理解Decoder的Causal Masking

**可视化**:
```python
# Causal Mask (Decoder)
[1 0 0 0]  Token 0 只能看自己
[1 1 0 0]  Token 1 能看0,1
[1 1 1 0]  Token 2 能看0,1,2
[1 1 1 1]  Token 3 能看0,1,2,3

# 应用后的Attention
       t0  t1  t2  t3
t0    [██  --  --  --]  -- = masked (权重=0)
t1    [▒█  ██  --  --]
t2    [░▒  ▒█  ██  --]
t3    [░░  ░▒  ▒█  ██]

下三角: 可见
上三角: masked (防止看到未来)

这是自回归生成的关键！✅
```

---

## 🛠️ 实现方案

### 方案A: 使用预训练模型 (推荐✅)

**优点**:
- 真实的attention模式
- 可以用有意义的句子
- 结果更有洞察力

**步骤**:
```python
# 1. 加载预训练模型
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained('bert-base-uncased', 
                                  output_attentions=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 2. 准备输入
text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors='pt')

# 3. 前向传播，提取attention
outputs = model(**inputs)
attentions = outputs.attentions  # Tuple of [batch, heads, seq, seq]

# 4. 可视化
import matplotlib.pyplot as plt
import seaborn as sns

for layer_idx, layer_attn in enumerate(attentions):
    # layer_attn: [batch, heads, seq, seq]
    for head_idx in range(layer_attn.size(1)):
        attn = layer_attn[0, head_idx].detach().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn, 
                    xticklabels=tokens,
                    yticklabels=tokens,
                    cmap='Reds',
                    cbar_kws={'label': 'Attention Weight'})
        plt.title(f'Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key')
        plt.ylabel('Query')
        plt.savefig(f'attention_L{layer_idx}_H{head_idx}.png')
        plt.close()
```

### 方案B: 自己训练小模型

**优点**:
- 完全控制
- 可以观察训练过程中attention的演化

**步骤**:
```python
# 1. 训练简单任务
#    例如: 复制序列, 排序, 简单QA

# 2. 每N个epoch保存attention
#    观察学习过程

# 3. 可视化attention演化
#    Epoch 0 → Epoch 100
#    从随机 → 有意义的模式
```

---

## 📊 预期观察与洞察

### 观察1: Head的专业化 ✅

**预期**:
```
不同heads学习不同模式:
- Head 1: 主要对角线 (位置bias)
- Head 2: 语法关系 (动词→主语)
- Head 3: 指代消解 (代词→名词)
- Head 4: 长距离依赖
- ...

这验证了Multi-Head的价值:
不是参数量，是多样性！✅
```

### 观察2: 层的层次性 ✅

**预期**:
```
Layer 1: 局部模式 (相邻tokens)
Layer 12: 语法结构 (句法树)
Layer 24: 语义关系 (主题相关)

从浅到深的抽象层次！✅
```

### 观察3: 注意力的稀疏性 ✅

**预期**:
```
大部分权重很小 (<0.1)
只有少数token获得高权重 (>0.3)

例如:
"The cat sat on the mat"中
"cat"主要关注: "sat" (0.4), "The" (0.3)
其他都<0.1

自然语言的稀疏性体现在attention中！✅

这也解释了:
- 为什么Sparse Attention可行
- 为什么MoE有效 (稀疏激活)
- 稀疏性是language的本质特征
```

### 观察4: Position Encoding的作用 ✅

**实验**: 对比有无Position Encoding

**无Position Encoding**:
```
Attention完全基于内容相似度
"cat"在哪里都一样
顺序信息丢失 ❌
```

**有Position Encoding**:
```
Attention考虑位置信息
"The cat" vs "cat The"
结果不同 ✅

Position Encoding确实被学习和利用！
```

---

## 🎯 深入分析问题

### 问题1: 不同heads真的学到不同模式吗？

**分析方法**:
```python
# 计算head之间的相似度
for i in range(num_heads):
    for j in range(i+1, num_heads):
        similarity = cosine_similarity(
            attn_i.flatten(),
            attn_j.flatten()
        )
        
预期:
相似度 < 0.5 → heads确实不同 ✅
相似度 > 0.8 → heads冗余 ⚠️
```

### 问题2: 能否解释每个head的"语义"？

**分析方法**:
```python
# 对于每个head，统计:
# 1. 最高权重的token pair
# 2. 依存关系类型 (如果有parse tree)
# 3. 词性模式

例如发现:
Head 3: 80%的最高权重都是 (verb → subject)
       → 这是"主谓关系head" ✅
       
Head 5: 70%的最高权重都是相邻tokens
       → 这是"局部上下文head" ✅
```

### 问题3: Attention真的在"关注"语言学特征吗？

**验证方法**:
```python
# 与语言学分析对比

1. 依存分析 (Dependency Parsing)
   Attention权重 vs 依存关系
   
2. 共指消解 (Coreference)
   代词的attention vs 共指链
   
3. 语义角色标注 (SRL)
   动词的attention vs 论元

如果correlation > 0.6:
→ Attention确实学到了语言学知识！✅
```

---

## 📚 学习价值

### 价值1: 从抽象到具象 ✅

**理论**: 
```
Attention = Q @ K.T / sqrt(d_k)
然后softmax，然后乘V
```

**可视化后**:
```
"哦！这个head在找主谓关系！"
"这个head在做指代消解！"
数学 → 直观理解 ✅
```

### 价值2: 验证理论假设 ✅

**假设1**: Multi-Head学习不同模式
**验证**: 可视化确实显示head专业化 ✅

**假设2**: 深层学习抽象特征
**验证**: Layer 24的attention确实更语义化 ✅

**假设3**: Position Encoding被利用
**验证**: 对比实验显示位置信息被学习 ✅

### 价值3: 指导模型优化 ✅

**发现1**: 某些heads冗余
**应用**: Head Pruning，减少参数 ✅

**发现2**: Attention很稀疏
**应用**: Sparse Attention，降低复杂度 ✅

**发现3**: 不同层关注不同特征
**应用**: Layer-wise学习率，Adapter等 ✅

---

## 🚀 扩展项目

### 扩展1: 动态Attention可视化

**创建GIF/视频**:
```
生成过程中的attention变化

Token 1: [attention图]
Token 2: [attention图]
...
Token 50: [attention图]

观察: 随着生成，attention模式如何演化
```

### 扩展2: 对比不同模型

**GPT vs BERT**:
```
GPT (Causal):  下三角attention
BERT (Bidirectional): 全矩阵attention

可视化差异 → 理解架构选择
```

### 扩展3: 错误案例分析

**找模型犯错的例子**:
```
例如: "The bank by the river"
模型理解为"银行" (错误)

可视化attention:
发现模型没有正确关注"by the river"
→ 理解错误原因 ✅
```

---

## 📝 项目总结模板

```markdown
# Attention可视化项目报告

## 实验设置
- 模型: BERT-base-uncased
- 层数: 12
- Heads: 12 per layer
- 测试句子: [列出]

## 主要发现

### 发现1: Head专业化
[插入热力图]
观察: ...

### 发现2: 层次性
[插入对比图]
观察: ...

### 发现3: 稀疏性
统计: 平均90%的权重 < 0.1
观察: ...

## 洞察

### 洞察1: ...
### 洞察2: ...

## 连接到Lecture 03

这些可视化验证了:
- Q8: Multi-Head确实学习多样性 ✅
- Q15: Position Encoding被利用 ✅
- Q21: Attention的稀疏性 → Sparse Attention可行 ✅

## 未来工作
- [ ] 对比更多模型
- [ ] 分析错误案例
- [ ] 与语言学分析对比
```

---

## 🎓 学习成果

完成这个项目后，你将：

✅ **直观理解Attention** (⭐⭐⭐⭐⭐)
- 从数学公式到实际模式
- "看到"模型在关注什么

✅ **验证理论知识** (⭐⭐⭐⭐⭐)
- Multi-Head的价值
- 层的层次性
- 稀疏性

✅ **建立分析能力** (⭐⭐⭐⭐⭐)
- 可视化工具
- 模式识别
- 错误诊断

✅ **指导实践优化** (⭐⭐⭐⭐⭐)
- Head Pruning
- Sparse Attention
- Layer-wise优化

---

**项目状态**: 理论设计完成 ✅  
**预计完成时间**: 1-2小时实践  
**学习价值**: ⭐⭐⭐⭐⭐ 极高

🎨 **准备好看到Transformer的"内心世界"了吗？**
