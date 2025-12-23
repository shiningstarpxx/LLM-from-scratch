# Tokenization原理

> **一句话摘要**: 从字符级到BPE，理解语言模型的第一道处理步骤，掌握现代Tokenizer的设计原理和实现技巧。

## 核心概念

### 关键术语
| 术语 | 定义 | 重要性 |
|------|------|--------|
| Tokenization | 将文本转换为模型可处理的整数序列 | LLM的第一步 |
| BPE (Byte Pair Encoding) | 从数据自动学习词汇表的算法 | 现代LLM标准 |
| Vocabulary | 词汇表，所有可能token的集合 | 决定模型能表示什么 |
| Compression Ratio | 字节数/token数，衡量编码效率 | 越高越好 |
| OOV (Out of Vocabulary) | 词汇表外的词 | BPE无此问题 |

### 概念图谱
```
Tokenization演进
├── 字符级 (Character)
│   ├── 优点: 完整覆盖
│   └── 缺点: 词表大, 序列长
├── 字节级 (Byte)
│   ├── 优点: 词表固定256
│   └── 缺点: 压缩比最差
├── 词级 (Word)
│   ├── 优点: 语义单元
│   └── 缺点: OOV问题严重
├── 子词级 (Subword) ← 主流
│   ├── BPE (GPT系列)
│   ├── WordPiece (BERT)
│   └── Unigram (T5)
└── 无Tokenization (实验阶段)
    ├── ByT5
    └── MegaByte
```

## 技术深度

### 1. 为什么需要Tokenization?

**根本原因**:
```python
# 语言模型的数学本质
# P(token_t | token_1, ..., token_{t-1})

# 模型需要:
# 1. 有限的词汇表 → 可以学习embedding
# 2. 合理的序列长度 → Attention是O(n²)
# 3. 语义单元 → 便于学习规律
```

**不同方法的权衡**:
```
方法         词汇表大小    序列长度    OOV问题
──────────────────────────────────────────
字符级       ~150K        很长       无
字节级       256          最长       无
词级         10万+        最短       严重
BPE          10K-100K     适中       无 ← 最优平衡!
```

### 2. 字符级 vs 字节级

**字符级**:
```python
class CharacterTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return ''.join(chr(t) for t in tokens)

# 示例
text = "Hello, 🌍!"
tokens = [72, 101, 108, 108, 111, 44, 32, 127757, 33]

# 问题:
# - 词汇表约15万 (所有Unicode字符)
# - 🌍 (emoji) 占一个token但很少用
# - 压缩比 ≈ 1.0 (每字符一个token)
```

**字节级**:
```python
class ByteTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode('utf-8'))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode('utf-8')

# 示例
text = "Hello, 🌍!"
tokens = [72, 101, 108, 108, 111, 44, 32, 240, 159, 140, 141, 33]
#                                      └── 🌍 变成4个字节 ──┘

# 优点: 词汇表固定为256
# 缺点:
# - 压缩比最差 (UTF-8非ASCII字符膨胀)
# - 中文每字3字节 → 序列过长
```

### 3. BPE算法详解

**核心思想**: 从数据自动学习合并规则

```python
def train_bpe(text: str, num_merges: int):
    """
    BPE训练过程
    """
    # 1. 初始化: 字节级token
    tokens = list(text.encode('utf-8'))
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}  # 合并规则

    for i in range(num_merges):
        # 2. 统计相邻pair频率
        pair_counts = {}
        for j in range(len(tokens) - 1):
            pair = (tokens[j], tokens[j+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        # 3. 找最频繁的pair
        best_pair = max(pair_counts, key=pair_counts.get)

        # 4. 创建新token
        new_token = 256 + i
        vocab[new_token] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges[best_pair] = new_token

        # 5. 应用合并
        tokens = merge(tokens, best_pair, new_token)

    return vocab, merges
```

**合并操作**:
```python
def merge(tokens: list[int], pair: tuple, new_token: int) -> list[int]:
    """
    将所有pair实例替换为new_token
    """
    result = []
    i = 0
    while i < len(tokens):
        if (i < len(tokens) - 1 and
            tokens[i] == pair[0] and
            tokens[i+1] == pair[1]):
            result.append(new_token)
            i += 2  # 跳过两个
        else:
            result.append(tokens[i])
            i += 1
    return result
```

**编码过程**:
```python
class BPETokenizer:
    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.merges = merges

    def encode(self, text: str) -> list[int]:
        # 1. 转字节
        tokens = list(text.encode('utf-8'))

        # 2. 按顺序应用合并规则
        for pair, new_token in self.merges.items():
            tokens = merge(tokens, pair, new_token)

        return tokens

    def decode(self, tokens: list[int]) -> str:
        # 查表 + 连接 + 解码UTF-8
        byte_list = [self.vocab[t] for t in tokens]
        return b''.join(byte_list).decode('utf-8')
```

### 4. GPT-2的改进

**Pre-tokenization (预分词)**:
```python
import regex

# GPT-2的正则表达式
GPT2_PATTERN = r"""
    's|'t|'re|'ve|'m|'ll|'d|  # 英语缩写
    [^\r\n\p{L}\p{N}]?\p{L}+|  # 单词(可能前有标点)
    \p{N}{1,3}|                 # 数字(最多3位一组)
    \ ?[^\s\p{L}\p{N}]+[\r\n]*| # 标点符号
    \s*[\r\n]+|                 # 换行
    \s+                         # 空白
"""

def pretokenize(text: str) -> list[str]:
    """先分割成片段，再在每个片段内做BPE"""
    return regex.findall(GPT2_PATTERN, text)

# 示例
text = "Hello world! 12345"
segments = ["Hello", " world", "!", " 12", "345"]
# 在每个segment内独立做BPE
```

**为什么预分词?**:
```
Without pretokenization:
"Hello world" 可能合并成 "Hello world" (一个token)
→ 对罕见组合效率低

With pretokenization:
"Hello" 和 " world" 分开处理
→ 每个常见词独立学习
→ 更好的压缩和泛化
```

### 5. 压缩比分析

**定义**:
```python
compression_ratio = len(text.encode('utf-8')) / len(tokens)
# 字节数 / token数
# 越高越好 (每个token承载更多信息)
```

**不同方法对比**:
```python
text = "The quick brown fox jumps over the lazy dog."

# 字节级
byte_tokens = list(text.encode('utf-8'))
# 44 tokens, ratio = 1.0

# BPE (GPT-2, vocab=50257)
gpt2_tokens = gpt2_encode(text)
# ~10 tokens, ratio ≈ 4.4

# 压缩比提升: 4.4x!
```

**压缩比的实际意义**:
```
更高的压缩比 →
1. 更短的序列 → 更少的Attention计算
2. 更大的有效上下文 → 更多信息
3. 更快的推理 → 更低的成本

GPT-4 context: 128K tokens
如果压缩比4.0 → 等效512K字节文本
如果压缩比1.0 → 只有128K字节文本
```

### 6. 特殊Token处理

```python
class TokenizerWithSpecialTokens:
    def __init__(self, base_tokenizer):
        self.base = base_tokenizer
        self.special_tokens = {
            '<|endoftext|>': 50256,
            '<|pad|>': 50257,
            '<|unk|>': 50258,
        }

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        # 1. 检测并保护特殊token
        for special, token_id in self.special_tokens.items():
            if special in text:
                # 分割，单独处理
                parts = text.split(special)
                result = []
                for i, part in enumerate(parts):
                    result.extend(self.base.encode(part))
                    if i < len(parts) - 1:
                        result.append(token_id)
                return result

        return self.base.encode(text)
```

### 7. 多语言Tokenization

**挑战**:
```python
# 同样的语义，不同语言token数差异巨大
text_en = "Hello, how are you?"  # 6 tokens
text_zh = "你好，你怎么样？"        # 9 tokens (每字~2-3 tokens)
text_ja = "こんにちは"             # 3-5 tokens

# 问题: 非英语语言"吃亏"
# - 更长的序列
# - 更少的有效上下文
# - 更高的推理成本
```

**解决方案**:
```
1. 多语言训练数据: 词汇表包含各语言常见词
2. SentencePiece: 语言无关的tokenization
3. 增大词汇表: 从50K到100K+
4. 专门的多语言模型: XLM, mBERT等
```

## 实践代码

### 简化BPE实现

```python
from collections import Counter
from typing import Dict, List, Tuple

class SimpleBPE:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab: Dict[int, bytes] = {}
        self.merges: Dict[Tuple[int, int], int] = {}

    def train(self, text: str):
        """训练BPE tokenizer"""
        # 初始化: 256个字节token
        tokens = list(text.encode('utf-8'))
        self.vocab = {i: bytes([i]) for i in range(256)}

        num_merges = self.vocab_size - 256

        for i in range(num_merges):
            if len(tokens) < 2:
                break

            # 统计pair频率
            pairs = Counter(zip(tokens, tokens[1:]))
            if not pairs:
                break

            # 最频繁的pair
            best_pair = pairs.most_common(1)[0][0]

            # 新token
            new_token = 256 + i
            self.vocab[new_token] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges[best_pair] = new_token

            # 合并
            tokens = self._merge(tokens, best_pair, new_token)

            if (i + 1) % 100 == 0:
                print(f"Merge {i+1}: {best_pair} -> {new_token}, vocab: {self.vocab[new_token]}")

    def _merge(self, tokens: List[int], pair: Tuple[int, int], new_token: int) -> List[int]:
        result = []
        i = 0
        while i < len(tokens):
            if (i < len(tokens) - 1 and
                tokens[i] == pair[0] and
                tokens[i+1] == pair[1]):
                result.append(new_token)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result

    def encode(self, text: str) -> List[int]:
        tokens = list(text.encode('utf-8'))
        for pair, new_token in self.merges.items():
            tokens = self._merge(tokens, pair, new_token)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        byte_list = [self.vocab[t] for t in tokens]
        return b''.join(byte_list).decode('utf-8', errors='replace')


# 使用示例
corpus = """
The quick brown fox jumps over the lazy dog.
The dog barks at the fox.
Quick foxes are brown.
""" * 100  # 重复以获得更好的统计

tokenizer = SimpleBPE(vocab_size=500)
tokenizer.train(corpus)

# 测试
test_text = "The quick brown fox"
tokens = tokenizer.encode(test_text)
decoded = tokenizer.decode(tokens)

print(f"Original: {test_text}")
print(f"Tokens: {tokens}")
print(f"Decoded: {decoded}")
print(f"Compression ratio: {len(test_text.encode('utf-8')) / len(tokens):.2f}")
```

### 压缩比对比实验

```python
import tiktoken

def compare_tokenizers(text: str):
    """对比不同tokenizer的效率"""

    # 字节级
    byte_tokens = len(text.encode('utf-8'))

    # GPT-2 (tiktoken)
    enc_gpt2 = tiktoken.get_encoding("gpt2")
    gpt2_tokens = len(enc_gpt2.encode(text))

    # cl100k (GPT-4)
    enc_cl100k = tiktoken.get_encoding("cl100k_base")
    cl100k_tokens = len(enc_cl100k.encode(text))

    results = {
        'Byte-level': (byte_tokens, 1.0),
        'GPT-2 (50K vocab)': (gpt2_tokens, byte_tokens/gpt2_tokens),
        'GPT-4 (100K vocab)': (cl100k_tokens, byte_tokens/cl100k_tokens),
    }

    print(f"Text: {text[:50]}...")
    print(f"Bytes: {byte_tokens}")
    print("-" * 40)
    for name, (tokens, ratio) in results.items():
        print(f"{name}: {tokens} tokens, ratio: {ratio:.2f}")


# 测试不同语言
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "人工智能正在改变世界的方方面面。",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
]

for text in texts:
    compare_tokenizers(text)
    print()
```

## 关键洞察

### 核心收获

1. **BPE是"数据驱动"的哲学**: 让数据告诉我们什么应该是一个token

2. **压缩比决定模型效率**: 4x压缩 = 4x有效上下文

3. **预分词不可忽视**: GPT-2的regex是精心设计的

4. **词汇表大小是权衡**:
   - 太小: 压缩不够，序列太长
   - 太大: embedding矩阵巨大，罕见token学不好

5. **多语言是开放问题**: 非英语语言仍然"吃亏"

### 常见误区

| 误区 | 正确理解 |
|------|----------|
| BPE是压缩算法 | BPE是用于NLP的分词算法 |
| 词汇表越大越好 | 有最优点，太大反而有害 |
| BPE等同于子词 | BPE是子词方法之一，还有WordPiece等 |
| Tokenization已解决 | 多语言、代码等仍有挑战 |

## 延伸阅读

### 推荐资源
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) - BPE用于NLP的原始论文
- [tiktoken](https://github.com/openai/tiktoken) - OpenAI的高效tokenizer库
- [SentencePiece](https://github.com/google/sentencepiece) - Google的多语言tokenizer

### 在线工具
- [Tiktokenizer](https://tiktokenizer.vercel.app/) - 交互式tokenization可视化

### 相关专题
- [Transformer架构精讲](../01-Transformer架构精讲/) - 为什么序列长度重要

---

## 内容来源

本文档内容整理自以下来源：
- [来源: 学习笔记/01-基础建立/01-Lecture01-Introduction/03-Tokenization原理与实践.md]
- [来源: 学习笔记/01-基础建立/01-Lecture01-Introduction/04-Tokenization深度问答.md]
- [来源: 学习笔记/01-基础建立/01-Lecture01-Introduction/bpe_core.py]

---

**作者**: peixingxin + Claude Code
**创建日期**: 2025-12-17
**最后更新**: 2025-12-17
