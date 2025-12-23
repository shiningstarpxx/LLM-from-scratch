# Tokenization深度解析 - 60分钟深度版

---

## 封面

### Tokenization原理深度解析
**副标题**: 从BPE到多语言，掌握LLM的文本预处理

**分享人**: [你的名字]
**时长**: 60分钟 (含10min Q&A)

---

## 议程

```
Part 1: 背景与动机              (5 min)
Part 2: Tokenization方法演进    (10 min)
Part 3: BPE算法深度解析         (15 min)
Part 4: 实现细节与优化          (12 min)
Part 5: 多语言与评估            (8 min)
Part 6: 总结与讨论              (5 min)
Q&A                             (10 min)
```

---

# Part 1: 背景与动机

---

## 1.1 什么是Tokenization?

```
Tokenization: 文本 → 整数序列

"Hello, world!" → [15496, 11, 995, 0]

为什么需要?
1. 神经网络只能处理数字
2. 需要有限的词汇表 (embedding层)
3. 需要合理的序列长度 (attention O(n²))
```

---

## 1.2 核心权衡

```
词汇表大小 ←→ 序列长度

┌───────────────────────────────────────────┐
│ 词汇表小        词汇表大                  │
├───────────────────────────────────────────┤
│ 序列更长        序列更短                  │
│ embedding小     embedding大               │
│ 无OOV问题       可能有稀疏token           │
│ 计算更多        计算更少                  │
└───────────────────────────────────────────┘

目标: 找到最佳平衡点
```

---

## 1.3 压缩比定义

```python
def compression_ratio(text, tokens):
    """
    压缩比 = 原始字节数 / token数

    含义: 每个token承载多少字节信息
    越高越好 = 更高效的编码
    """
    bytes_count = len(text.encode('utf-8'))
    return bytes_count / len(tokens)

# 示例
text = "Hello, world!"
tokens = tokenizer.encode(text)  # [15496, 11, 995, 0]
ratio = compression_ratio(text, tokens)
print(f"压缩比: {ratio:.2f}")  # 约3.25
```

---

# Part 2: Tokenization方法演进

---

## 2.1 方法概览

```
演进历程:

字符级 → 字节级 → 词级 → 子词级 (BPE)
  │         │        │         │
简单      通用    语义好     平衡
但长      但长    有OOV      最佳
```

---

## 2.2 字符级Tokenization

```python
def char_tokenize(text):
    """字符级: 每个Unicode字符一个token"""
    # 建立词汇表: 所有可能的Unicode字符
    # 中文: ~20,000+ 常用字
    # 日文: ~10,000+ (汉字+假名)
    # 韩文: ~11,000+ 音节
    # 表情: ~3,000+
    # 总计: 可能超过150,000

    return [ord(c) for c in text]

# 问题:
# 1. 词汇表太大
# 2. 序列太长 ("Hello" = 5个token)
# 3. 无法学习词级语义
```

---

## 2.3 字节级Tokenization

```python
def byte_tokenize(text):
    """字节级: UTF-8编码的每个字节一个token"""
    return list(text.encode('utf-8'))

# 优点: 词汇表固定为256
# 缺点: 非ASCII字符变长

# 示例:
print(byte_tokenize("Hello"))  # [72,101,108,108,111] - 5字节
print(byte_tokenize("你好"))   # [228,189,160,229,165,189] - 6字节!

# 中文每字3字节，序列长度3倍于英文
# 这对非英语语言很不公平
```

---

## 2.4 词级Tokenization

```python
def word_tokenize(text, vocab):
    """词级: 每个词一个token"""
    words = text.split()
    UNK_ID = vocab['<unk>']
    return [vocab.get(w, UNK_ID) for w in words]

# 优点:
# - 语义单元清晰
# - 序列最短

# 缺点:
# - OOV (Out-of-Vocabulary) 问题严重
# - 词汇表巨大 (英语50万+词)
# - 变体问题: run, runs, running, ran...

# 实际上已被淘汰
```

---

## 2.5 子词级: BPE

```
BPE (Byte Pair Encoding) 思想:

1. 从最小单元(字节/字符)开始
2. 统计相邻单元对的频率
3. 合并最频繁的对为新单元
4. 重复直到达到目标词汇表大小

结果:
- 常见词 → 单个token
- 罕见词 → 多个token
- 永远不会OOV!
```

---

## 2.6 方法对比总结

| 方法 | 词汇表 | 序列长度 | OOV | 语义 | 现状 |
|------|--------|----------|-----|------|------|
| 字符 | ~150K | 最长 | 无 | 差 | 少用 |
| 字节 | 256 | 最长 | 无 | 差 | 做基础 |
| 词 | ~500K | 最短 | 严重 | 好 | 淘汰 |
| BPE | 30K-100K | 适中 | 无 | 较好 | 主流 |

---

# Part 3: BPE算法深度解析

---

## 3.1 训练流程详解

```python
def train_bpe(corpus: str, num_merges: int):
    """BPE训练完整流程"""

    # Step 1: 初始化为字节
    tokens = list(corpus.encode('utf-8'))
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}

    for i in range(num_merges):
        # Step 2: 统计所有相邻对的频率
        pair_counts = {}
        for j in range(len(tokens) - 1):
            pair = (tokens[j], tokens[j+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        if not pair_counts:
            break

        # Step 3: 找最频繁的对
        best_pair = max(pair_counts, key=pair_counts.get)

        # Step 4: 创建新token
        new_id = 256 + i
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges[best_pair] = new_id

        # Step 5: 在序列中应用合并
        tokens = apply_merge(tokens, best_pair, new_id)

        print(f"Merge {i}: {best_pair} -> {new_id} ({vocab[new_id]})")

    return vocab, merges
```

---

## 3.2 合并操作实现

```python
def apply_merge(tokens: list, pair: tuple, new_id: int) -> list:
    """
    将所有出现的pair替换为new_id

    tokens = [1, 2, 3, 2, 3, 4]
    pair = (2, 3)
    new_id = 100
    result = [1, 100, 100, 4]
    """
    result = []
    i = 0

    while i < len(tokens):
        # 检查是否匹配pair
        if (i < len(tokens) - 1 and
            tokens[i] == pair[0] and
            tokens[i+1] == pair[1]):
            result.append(new_id)
            i += 2  # 跳过两个token
        else:
            result.append(tokens[i])
            i += 1

    return result
```

---

## 3.3 BPE训练示例

```
输入语料: "aaabdaaabac"
目标: 学习3次合并

初始:
tokens = [97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99]
         ['a','a','a','b','d','a','a','a','b','a','c']

迭代1:
- 统计: (97,97)=4, (97,98)=2, (98,100)=1, ...
- 最频繁: (97,97) = "aa"
- 合并为256: "aa"
- tokens = [256, 97, 98, 100, 256, 97, 98, 97, 99]
           ['aa','a','b','d','aa','a','b','a','c']

迭代2:
- 统计: (256,97)=2, (97,98)=2, ...
- 最频繁: (256,97) = "aaa"
- 合并为257: "aaa"
- tokens = [257, 98, 100, 257, 98, 97, 99]
           ['aaa','b','d','aaa','b','a','c']

迭代3:
- 统计: (257,98)=2, ...
- 合并为258: "aaab"
```

---

## 3.4 编码 (Encode)

```python
def encode(text: str, merges: dict) -> list:
    """
    使用学习的merges规则编码文本

    关键: 必须按训练时的顺序应用合并!
    """
    # 初始化为字节
    tokens = list(text.encode('utf-8'))

    # 按顺序应用每个合并规则
    for pair, new_id in merges.items():
        tokens = apply_merge(tokens, pair, new_id)

    return tokens


# 为什么顺序重要?
# 训练时: 先合并 "aa" -> 256, 再合并 "aaa" (256+97) -> 257
# 如果顺序错了, "aaa" 可能被错误编码
```

---

## 3.5 解码 (Decode)

```python
def decode(tokens: list, vocab: dict) -> str:
    """
    将token序列解码回文本

    简单: 查表 → 连接 → UTF-8解码
    """
    byte_list = []
    for token_id in tokens:
        byte_list.append(vocab[token_id])

    # 连接所有字节
    all_bytes = b''.join(byte_list)

    # UTF-8解码
    return all_bytes.decode('utf-8', errors='replace')


# 解码永远不会失败 (最差情况用replacement字符)
```

---

## 3.6 高效编码: 优先队列

```python
import heapq

def encode_efficient(text: str, merges: dict) -> list:
    """
    更高效的编码: 使用优先队列

    原始方法: O(n × m), n=文本长度, m=合并规则数
    优化方法: O(n log n)
    """
    tokens = list(text.encode('utf-8'))

    # 建立反向索引: pair -> (优先级, new_id)
    merge_priority = {pair: (i, new_id)
                      for i, (pair, new_id) in enumerate(merges.items())}

    while True:
        # 找当前序列中优先级最高的可合并对
        best_pair = None
        best_priority = float('inf')

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            if pair in merge_priority:
                priority, _ = merge_priority[pair]
                if priority < best_priority:
                    best_priority = priority
                    best_pair = pair

        if best_pair is None:
            break

        # 应用合并
        _, new_id = merge_priority[best_pair]
        tokens = apply_merge(tokens, best_pair, new_id)

    return tokens
```

---

# Part 4: 实现细节与优化

---

## 4.1 Pre-tokenization

```python
# 问题: 纯BPE可能产生不自然的合并
# "I like eating" -> "I like eat" + "ing" ?
# 应该是: "I" + " like" + " eating"

# 解决: Pre-tokenization (预分词)

import regex

# GPT-2的pre-tokenization模式
GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

def pre_tokenize(text):
    """先按正则分割，再在每段内做BPE"""
    segments = regex.findall(GPT2_PATTERN, text)
    return segments

# 示例
text = "Hello, I've got 123 apples!"
segments = pre_tokenize(text)
# ['Hello', ',', ' I', "'ve", ' got', ' 123', ' apples', '!']

# 然后对每个segment独立做BPE
# 这样 "Hello" 永远不会和 "," 合并
```

---

## 4.2 特殊Token处理

```python
class Tokenizer:
    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.merges = merges

        # 特殊token
        self.special_tokens = {
            '<|endoftext|>': 50256,
            '<|pad|>': 50257,
            '<|startoftext|>': 50258,
        }

    def encode(self, text, allowed_special=set()):
        """
        编码时保护特殊token
        """
        # 1. 找到所有特殊token的位置
        special_pattern = '|'.join(
            re.escape(k) for k in self.special_tokens
            if k in allowed_special
        )

        # 2. 按特殊token分割
        if special_pattern:
            parts = re.split(f'({special_pattern})', text)
        else:
            parts = [text]

        # 3. 分别处理每部分
        tokens = []
        for part in parts:
            if part in self.special_tokens:
                tokens.append(self.special_tokens[part])
            else:
                tokens.extend(self._encode_ordinary(part))

        return tokens
```

---

## 4.3 处理未知字节

```python
def handle_invalid_utf8(byte_sequence):
    """
    处理无效UTF-8序列

    BPE基于字节，所以任何输入都能编码
    但解码时可能遇到无效UTF-8
    """
    try:
        return byte_sequence.decode('utf-8')
    except UnicodeDecodeError:
        # 选项1: 使用replacement字符
        return byte_sequence.decode('utf-8', errors='replace')

        # 选项2: 忽略无效字节
        # return byte_sequence.decode('utf-8', errors='ignore')

        # 选项3: 用\x转义
        # return byte_sequence.decode('latin-1')
```

---

## 4.4 Tiktoken实现

```python
# OpenAI的高性能Tokenizer

import tiktoken

# 加载GPT-4 tokenizer
enc = tiktoken.get_encoding("cl100k_base")

# 编码
tokens = enc.encode("Hello, world!")
print(tokens)  # [9906, 11, 1917, 0]

# 解码
text = enc.decode(tokens)
print(text)  # "Hello, world!"

# 查看单个token
print(enc.decode([9906]))  # "Hello"

# 特殊token
tokens = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
print(tokens)  # [100257]
```

---

## 4.5 SentencePiece

```python
# Google的Tokenizer库，支持多种算法

import sentencepiece as spm

# 训练
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='my_tokenizer',
    vocab_size=32000,
    model_type='bpe',  # 或 'unigram'
    character_coverage=0.9995,  # 重要: 覆盖99.95%字符
)

# 加载
sp = spm.SentencePieceProcessor()
sp.load('my_tokenizer.model')

# 使用
tokens = sp.encode("Hello, world!")
text = sp.decode(tokens)

# 特点:
# 1. 直接在Unicode上工作
# 2. 支持Unigram算法 (概率模型)
# 3. 更好的多语言支持
```

---

## 4.6 Unigram vs BPE

```
BPE (Byte Pair Encoding):
- 自底向上合并
- 确定性: 同样的输入总是同样的输出
- 贪婪算法

Unigram:
- 自顶向下剪枝
- 概率模型: P(text) = Π P(token_i)
- 可以输出多种分词方案

# Unigram训练
1. 从大词汇表开始
2. 用EM算法估计每个token的概率
3. 移除降低likelihood最少的token
4. 重复直到达到目标大小

# 实际效果差异不大，BPE更普遍
```

---

# Part 5: 多语言与评估

---

## 5.1 多语言压缩比差异

```python
def compare_languages(tokenizer):
    """比较不同语言的压缩比"""

    texts = {
        "English": "The quick brown fox jumps over the lazy dog.",
        "Chinese": "敏捷的棕色狐狸跳过了懒狗。",
        "Japanese": "素早い茶色の狐が怠け者の犬を飛び越える。",
        "Korean": "빠른 갈색 여우가 게으른 개를 뛰어넘는다.",
        "Arabic": "الثعلب البني السريع يقفز فوق الكلب الكسول.",
    }

    for lang, text in texts.items():
        tokens = tokenizer.encode(text)
        ratio = len(text.encode('utf-8')) / len(tokens)
        print(f"{lang}: {len(tokens)} tokens, 压缩比={ratio:.2f}")

# GPT-2 tokenizer 结果:
# English: 11 tokens, 压缩比=4.18
# Chinese: 24 tokens, 压缩比=1.50  ← 差!
# Japanese: 23 tokens, 压缩比=1.74
# Korean: 21 tokens, 压缩比=1.43
# Arabic: 37 tokens, 压缩比=1.24
```

---

## 5.2 为什么非英语吃亏?

```
根本原因: 训练语料以英文为主

1. BPE学习频繁的模式
2. 英文模式被学成高级token
3. 其他语言模式频率低，保持低级

影响:
- 中文: 1token ≈ 1字节 (vs 英文 4字节)
- 同样语义，中文序列长4倍
- 更少有效上下文
- 更高推理成本

解决:
- 更大词汇表 (GPT-4: 100K)
- 更平衡的训练语料
- 多语言预训练
```

---

## 5.3 词汇表大小选择

```
权衡因素:

词汇表小 (10K-30K):
+ Embedding矩阵小: 10K × 4096 × 2 = 80MB
+ 每个token更常见，学得更好
- 压缩比低，序列长

词汇表大 (100K+):
+ 压缩比高，序列短
+ 更好的多语言支持
- Embedding矩阵大: 100K × 4096 × 2 = 800MB
- 稀有token学不好

实践:
- GPT-2: 50,257
- GPT-3: 50,257
- GPT-4: ~100,000
- LLaMA: 32,000
- Claude: ~100,000

趋势: 向100K+发展
```

---

## 5.4 数字处理问题

```python
# 问题: BPE对数字处理不一致

tokenizer = tiktoken.get_encoding("cl100k_base")

# 不同数字的token数
for num in ["123", "1234", "12345", "123456"]:
    tokens = tokenizer.encode(num)
    print(f"{num}: {len(tokens)} tokens - {tokens}")

# 可能输出:
# 123: 1 token - [4513]
# 1234: 1 token - [10234]
# 12345: 2 tokens - [4513, 1234]
# 123456: 2 tokens - [4513, 12345]

# 问题:
# 1. 相邻数字token化不一致
# 2. 模型难以学习数字规律
# 3. 算术能力受限

# 解决: pre-tokenization分割数字
# "12345" -> "12" + "345" 或 "1" + "2" + "3" + "4" + "5"
```

---

## 5.5 评估指标

```python
def evaluate_tokenizer(tokenizer, test_corpus):
    """评估tokenizer的各项指标"""

    results = {}

    # 1. 压缩比 (越高越好)
    total_bytes = sum(len(t.encode('utf-8')) for t in test_corpus)
    total_tokens = sum(len(tokenizer.encode(t)) for t in test_corpus)
    results['compression_ratio'] = total_bytes / total_tokens

    # 2. 词汇表覆盖率
    all_tokens = set()
    for text in test_corpus:
        all_tokens.update(tokenizer.encode(text))
    results['vocab_coverage'] = len(all_tokens) / len(tokenizer.vocab)

    # 3. 平均token长度 (bytes)
    results['avg_token_length'] = total_bytes / total_tokens

    # 4. 稀有token比例 (出现少于N次的token)
    from collections import Counter
    token_counts = Counter()
    for text in test_corpus:
        token_counts.update(tokenizer.encode(text))
    rare_tokens = sum(1 for c in token_counts.values() if c < 10)
    results['rare_token_ratio'] = rare_tokens / len(token_counts)

    return results
```

---

## 5.6 Tokenization-free方向

```
研究方向: 完全跳过tokenization

方法1: 字节级模型
- 直接在UTF-8字节上训练
- 问题: 序列太长

方法2: 字符级 + 层次结构
- 底层处理字符
- 高层学习词级表示

方法3: 可学习tokenization
- Tokenization作为模型的一部分
- 端到端学习

现状:
- 研究活跃但不成熟
- BPE仍是工业标准
- 未来可能有突破
```

---

# Part 6: 总结与讨论

---

## 6.1 核心概念回顾

| 概念 | 定义 |
|------|------|
| Tokenization | 文本→整数序列 |
| BPE | 从数据学习合并规则 |
| 压缩比 | bytes/tokens，越高越好 |
| Pre-tokenization | BPE前的预分割 |
| 特殊Token | 控制符如EOS, PAD |

---

## 6.2 关键数字

```
常见词汇表大小:
- GPT-2/3: 50,257
- GPT-4: ~100,000
- LLaMA: 32,000

压缩比 (GPT-2):
- 英文: ~4.0 bytes/token
- 中文: ~1.5 bytes/token
- 代码: ~3.5 bytes/token

合并规则数:
- 词汇表大小 - 256 (基础字节)
- 50K词汇: ~50,000次合并
```

---

## 6.3 最佳实践

```
1. 选择tokenizer:
   - 使用成熟实现 (tiktoken, sentencepiece)
   - 考虑目标语言分布

2. 词汇表大小:
   - 单语言: 30K-50K
   - 多语言: 100K+

3. Pre-tokenization:
   - 分割数字 (提高算术能力)
   - 保护特殊token
   - 考虑语言特性

4. 训练:
   - 语料要有代表性
   - 考虑下游任务
```

---

## 6.4 常见误区

| 误区 | 正确理解 |
|------|----------|
| BPE是压缩算法 | 是分词算法 |
| 词汇表越大越好 | 有最优点 |
| 所有语言平等 | 英文优势明显 |
| Tokenization不影响性能 | 影响很大 |
| 一个tokenizer通吃 | 不同任务可能需要不同 |

---

## 6.5 前沿话题

```
1. Tokenization-free LLM
   - 直接处理字节
   - 更统一的多语言处理

2. 动态词汇表
   - 根据输入调整
   - 更高效的编码

3. 多模态Tokenization
   - 图像、音频、视频
   - 统一表示空间

4. 可微分Tokenization
   - 端到端学习
   - 任务特定优化
```

---

## Q&A

### Q1: 为什么不用词级tokenization?
**A**: OOV问题太严重。任何新词、变体、拼写错误都会变成UNK。BPE保证任何输入都能编码。

### Q2: 压缩比能做到多高?
**A**: 理论上限取决于文本的熵。英文BPE约4，接近英文自然语言熵的估计值。

### Q3: 不同tokenizer能混用吗?
**A**: 不能。每个模型绑定特定tokenizer。使用错误的tokenizer会导致无意义输出。

### Q4: 中文模型应该用什么tokenizer?
**A**: 要么用大词汇表(100K+)的多语言tokenizer，要么训练中文专用tokenizer。

---

**感谢聆听！**

---

## 附录: 完整BPE实现

```python
"""
完整的BPE Tokenizer实现
"""
import regex
from collections import Counter

class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}
        self.special_tokens = {}
        self.pattern = regex.compile(
            r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
        )

    def train(self, text: str, vocab_size: int):
        """训练BPE"""
        # 初始化词汇表为所有字节
        self.vocab = {i: bytes([i]) for i in range(256)}

        # Pre-tokenize
        segments = self.pattern.findall(text)

        # 将所有segments转为字节
        tokens = []
        for seg in segments:
            tokens.extend(list(seg.encode('utf-8')))

        # 迭代合并
        num_merges = vocab_size - 256
        for i in range(num_merges):
            # 统计pair频率
            pairs = Counter()
            for j in range(len(tokens) - 1):
                pairs[(tokens[j], tokens[j+1])] += 1

            if not pairs:
                break

            # 找最频繁的
            best = max(pairs, key=pairs.get)

            # 创建新token
            new_id = 256 + i
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            self.merges[best] = new_id

            # 应用合并
            tokens = self._merge(tokens, best, new_id)

    def _merge(self, tokens, pair, new_id):
        result = []
        i = 0
        while i < len(tokens):
            if (i < len(tokens) - 1 and
                tokens[i] == pair[0] and
                tokens[i+1] == pair[1]):
                result.append(new_id)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result

    def encode(self, text: str) -> list:
        """编码"""
        segments = self.pattern.findall(text)
        all_tokens = []

        for seg in segments:
            tokens = list(seg.encode('utf-8'))
            for pair, new_id in self.merges.items():
                tokens = self._merge(tokens, pair, new_id)
            all_tokens.extend(tokens)

        return all_tokens

    def decode(self, tokens: list) -> str:
        """解码"""
        byte_list = [self.vocab[t] for t in tokens]
        return b''.join(byte_list).decode('utf-8', errors='replace')

    def add_special_token(self, token: str, id: int):
        """添加特殊token"""
        self.special_tokens[token] = id
        self.vocab[id] = token.encode('utf-8')


# 使用示例
if __name__ == "__main__":
    tokenizer = BPETokenizer()

    # 训练
    corpus = open("corpus.txt").read()
    tokenizer.train(corpus, vocab_size=1000)

    # 测试
    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
```
