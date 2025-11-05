# Lecture 01: Tokenization 原理与实践

## 📚 Tokenization 概述

### 🎯 核心概念

**什么是Tokenization？**
- Tokenizer是字符串和token序列之间的转换器
- **Encode**: 字符串 → 整数序列（tokens）
- **Decode**: 整数序列 → 字符串
- **词汇表大小**: 可能的token数量（整数范围）

**为什么需要Tokenization？**
- 语言模型需要对token序列（通常是整数索引）建模概率分布
- 原始文本需要转换为模型可以处理的数字格式
- 这是文本预处理的第一步，也是最基础的一步

### 🔍 交互式体验

推荐工具：[https://tiktokenizer.vercel.app/?encoder=gpt2](https://tiktokenizer.vercel.app/?encoder=gpt2)

**观察现象**：
- 单词和前面的空格通常是同一个token（如 " world"）
- 单词在开头和中间的表示不同（如 "hello hello"）
- 数字被分割成每几位一组
- 特殊字符和emoji有独特的tokenization方式

---

## 🔄 Tokenization方法的演进

### 1️⃣ Character-based Tokenization（字符级）

#### 原理
- 将Unicode字符串转换为Unicode码点序列
- 每个字符通过`ord()`转换为整数
- 通过`chr()`将整数转换回字符

#### 实现示例
```python
class CharacterTokenizer(Tokenizer):
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))

    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))
```

#### 测试案例
```python
tokenizer = CharacterTokenizer()
string = "Hello, 🌍! 你好!"
indices = tokenizer.encode(string)  # [72, 101, 108, 108, 111, 44, 32, 127757, 33, 32, 20320, 22909, 33]
```

#### 优缺点分析

**优点**：
- ✅ 实现简单，概念清晰
- ✅ 完全覆盖所有Unicode字符
- ✅ 无需UNK（未知）token

**缺点**：
- ❌ **词汇表过大**：约15万Unicode字符
- ❌ **效率低下**：很多字符很少使用（如🌍）
- ❌ **压缩比差**：序列长度与字符数相同

**压缩比**：`len(bytes(string)) / len(tokens) ≈ 1.0`

---

### 2️⃣ Byte-based Tokenization（字节级）

#### 原理
- 将Unicode字符串编码为UTF-8字节序列
- 每个字节转换为0-255的整数
- 词汇表固定为256个可能值

#### UTF-8编码特点
- **ASCII字符**：1字节表示（如 'a' → b"a"）
- **多字节字符**：多个字节表示（如 '🌍' → b"\xf0\x9f\x8c\x8d"）

#### 实现示例
```python
class ByteTokenizer(Tokenizer):
    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8")
        return list(map(int, string_bytes))

    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)
        return string_bytes.decode("utf-8")
```

#### 测试案例
```python
tokenizer = ByteTokenizer()
string = "Hello, 🌍! 你好!"
indices = tokenizer.encode(string)  # 每个字符转换为1-4个字节
```

#### 优缺点分析

**优点**：
- ✅ **词汇表固定且小**：只有256个可能值
- ✅ **完全覆盖**：所有Unicode字符都能表示
- ✅ **无UNK问题**：任何字符都能编码

**缺点**：
- ❌ **压缩比极差**：`compression_ratio = 1.0`
- ❌ **序列过长**：注意力计算复杂度O(n²)下不实用
- ❌ **语义信息丢失**：字符被分割成字节片段

**压缩比**：`len(bytes(string)) / len(tokens) = 1.0`（最差）

---

### 3️⃣ Word-based Tokenization（词级）

#### 原理
- 使用正则表达式将文本分割为单词
- 建立单词到整数的映射词典
- 处理未知词使用UNK token

#### 分词策略
```python
# 简单版本：保留字母数字组合
segments = regex.findall(r"\w+|.", string)

# GPT-2风格：更复杂的分词规则
pattern = GPT2_TOKENIZER_REGEX
segments = regex.findall(pattern, string)
```

#### 实现挑战
- **词汇表大小**：取决于训练数据中的不同词数
- **罕见词问题**：很多词出现频率低，模型学习效果差
- **UNK处理**：未见过的词需要特殊处理，影响计算
- **固定词汇表**：难以处理新词和OOV（词汇表外）问题

#### 优缺点分析

**优点**：
- ✅ **语义单元**：单词是有意义的语言单元
- ✅ **压缩比较好**：比字符级更紧凑

**缺点**：
- ❌ **词汇表巨大**：数万到数十万词汇
- ❌ **UNK问题**：未见词汇需要特殊处理
- ❌ **形态变化**：相同词的不同形式被视为不同词
- ❌ **语言依赖**：不同语言需要不同的分词策略

---

### 4️⃣ Byte Pair Encoding (BPE) Tokenization

#### 🎯 历史背景
- **1994年**：Philip Gage为数据压缩提出
- **2016年**：Sennrich等人为神经机器翻译适配到NLP
- **2019年**：GPT-2采用并推广
- **现状**：现代LLM的标准选择

#### 🔧 核心思想
**基本理念**：
- 从原始文本自动学习词汇表
- 常见字符序列用单个token表示
- 罕见序列用多个token表示

**算法流程**：
1. **初始化**：每个字节作为一个token
2. **迭代合并**：找到最常相邻的token对进行合并
3. **构建词汇表**：每次合并产生新的token

#### 📊 算法详解

##### 训练过程示例
```python
def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:
    # 1. 初始化：字节级token
    indices = list(map(int, string.encode("utf-8")))
    vocab = {x: bytes([x]) for x in range(256)}  # 0-255 → 单字节
    merges = {}  # (token1, token2) → new_token

    for i in range(num_merges):
        # 2. 统计相邻token对频率
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):
            counts[(index1, index2)] += 1

        # 3. 找到最频繁的token对
        pair = max(counts, key=counts.get)
        new_index = 256 + i  # 新token索引

        # 4. 记录合并规则
        merges[pair] = new_index
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]

        # 5. 应用合并
        indices = merge(indices, pair, new_index)

    return BPETokenizerParams(vocab=vocab, merges=merges)
```

##### 合并操作实现
```python
def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    """将indices中所有pair实例替换为new_index"""
    new_indices = []
    i = 0
    while i < len(indices):
        if (i + 1 < len(indices) and
            indices[i] == pair[0] and
            indices[i + 1] == pair[1]):
            new_indices.append(new_index)
            i += 2  # 跳过已合并的两个token
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices
```

##### 编码和解码
```python
class BPETokenizer(Tokenizer):
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        # 1. 转换为字节序列
        indices = list(map(int, string.encode("utf-8")))

        # 2. 应用所有合并规则（简单但低效的实现）
        for pair, new_index in self.params.merges.items():
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices: list[int]) -> str:
        # 1. 将token索引转换为字节
        bytes_list = [self.params.vocab[idx] for idx in indices]
        # 2. 解码为字符串
        return b"".join(bytes_list).decode("utf-8")
```

#### 🎯 GPT-2的改进

##### 预分词（Pre-tokenization）
- 使用正则表达式先分割文本
- 在每个片段内独立应用BPE
- 保持数字、特殊字符的完整性

```python
GPT2_TOKENIZER_REGEX = r"""'s|'t|'re|'ve|'m|'ll|'d|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+"""
```

##### 特殊token处理
- 保留特殊token（如`<|endoftext|>`）
- 防止特殊token被BPE合并

#### 📈 性能特点

**压缩比**：
- 远好于字符级和字节级
- 常见模式得到有效压缩
- 罕见模式合理展开

**词汇表大小**：
- 可控：通过合并次数控制
- 典型范围：10K-100K tokens
- 平衡效率和表达能力

#### 🔧 优化方向

根据Assignment 1的要求，BPE实现还需要优化：

1. **编码效率**：
   - 当前：遍历所有合并规则 → O(M×L)
   - 优化：只应用相关合并 → 更高效的算法

2. **特殊token处理**：
   - 检测和保护特殊token
   - 防止被意外合并

3. **预分词集成**：
   - 集成GPT-2风格的正则表达式
   - 提高分词质量

4. **性能优化**：
   - 使用更高效的数据结构
   - 减少不必要的内存分配

---

## 🔄 Tokenization-Free 方法

### 🎯 前沿探索

虽然BPE是当前主流，但研究者也在探索无tokenization的方法：

**代表性工作**：
- **ByT5** (Google): 直接处理字节序列
- **Megabyte** (Meta): 分层字节处理
- **BLT** (Byte-Level Transformer): 字节级transformer
- **TFree** (Token-Free): 无token方法

### 🚀 潜力与挑战

**优势**：
- ✅ 真正的通用性：任何语言、任何符号
- ✅ 无分词歧义：确定性的字节处理
- ✅ 理论优雅：避免人工设计的词汇表

**挑战**：
- ❌ 计算效率：序列过长
- ❌ 训练难度：需要更长的上下文理解
- ❌ 未经验证：尚未在大规模上证明可行

**现状**： promising but not yet scaled up to the frontier

---

## 💡 核心洞察与思考

### 🎯 Tokenization的本质

**为什么需要Tokenization？**
1. **离散化需求**：神经网络需要离散输入
2. **计算效率**：合理的序列长度
3. **语义单元**：有意义的文本片段
4. **可学习性**：固定大小的词汇表

**Tokenization的权衡**：
- **压缩比 vs 语义完整性**
- **词汇表大小 vs 计算效率**
- **通用性 vs 领域特化**

### 🤔 设计哲学

**BPE的成功因素**：
1. **数据驱动**：从语料自动学习
2. **渐进式**：从简单到复杂的层次结构
3. **可扩展**：控制词汇表大小
4. **实用主义**：在实际约束下的最优解

**"必要的恶"**：
- Tokenization是当前架构下的必要妥协
- 理想情况下可能直接处理字节
- 但在当前transformer架构下是必需的

### 🔮 未来展望

**可能的演进方向**：
1. **更高效的BPE变体**
2. **无tokenization方法的突破**
3. **自适应tokenization**
4. **多模态统一的tokenization**

---

## 🛠️ 实践练习

### 📝 Assignment 1预告

**BPE Tokenizer实现要求**：
- [ ] 实现高效的BPE训练算法
- [ ] 实现快速的编码/解码
- [ ] 处理特殊token
- [ ] 集成预分词（GPT-2风格）
- [ ] 性能优化和基准测试

**评估指标**：
- **压缩比**：token序列vs字节序列
- **编码速度**：每秒处理的字符数
- **解码速度**：每秒生成的字符数
- **内存使用**：词汇表和合并规则的内存占用

### 🎯 学习目标

通过tokenization学习，你应该掌握：

1. **理论理解**：
   - 不同tokenization方法的原理和权衡
   - BPE算法的工作机制
   - 在现代LLM中的作用

2. **实践能力**：
   - 实现完整的BPE tokenizer
   - 优化算法性能
   - 处理边界情况和特殊需求

3. **工程思维**：
   - 在约束条件下寻找最优解
   - 理解为什么当前生态选择BPE
   - 批判性思考现有方法的局限性

---

**📝 备注**: Tokenization虽然看似简单，但涉及深刻的语言学、信息论和工程权衡。理解它有助于更好地理解整个语言模型的工作原理。