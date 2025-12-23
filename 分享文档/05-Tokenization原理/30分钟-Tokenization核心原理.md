# Tokenization核心原理 - 30分钟精华版

---

## 封面

### Tokenization原理
**副标题**: 理解BPE，掌握LLM的第一步处理

**分享人**: [你的名字]
**时长**: 30分钟

---

## 议程

```
1. 为什么需要Tokenization？    (3 min)
2. 方法演进                   (8 min)
3. BPE算法详解                (12 min)
4. 实践考量                   (5 min)
5. 总结                       (2 min)
```

---

# Part 1: 为什么需要Tokenization？

---

## 语言模型的需求

```python
# 语言模型本质:
P(next_token | previous_tokens)

# 需要:
# 1. 有限的词汇表 → 可学习embedding
# 2. 合理的序列长度 → Attention是O(n²)
# 3. 语义单元 → 便于学习规律
```

---

## 核心权衡

```
词汇表大小 ↔ 序列长度

词汇表小 → 序列长 → 计算多
词汇表大 → 序列短 → 但稀疏词学不好

需要找到平衡点!
```

---

# Part 2: 方法演进

---

## 2.1 字符级

```python
def char_tokenize(text):
    return [ord(c) for c in text]

# "Hello" → [72, 101, 108, 108, 111]

优点: 无OOV
缺点: 词汇表15万+, 序列太长
压缩比: 1.0
```

---

## 2.2 字节级

```python
def byte_tokenize(text):
    return list(text.encode('utf-8'))

# "Hello" → [72, 101, 108, 108, 111]
# "你好" → [228, 189, 160, 229, 165, 189]  # 6字节!

优点: 词汇表固定256
缺点: 中文等每字3+字节，序列爆炸
压缩比: 1.0 (最差)
```

---

## 2.3 词级

```python
def word_tokenize(text, vocab):
    words = text.split()
    return [vocab.get(w, UNK_ID) for w in words]

# "Hello world" → [1234, 5678]

优点: 语义单元，序列短
缺点: OOV问题严重，词汇表巨大
```

---

## 2.4 BPE (现代主流)

```
核心思想: 从数据自动学习词汇表

常见组合 → 单个token
罕见组合 → 多个token

"Hello" → [15496]  # 常见，1个token
"abracadabra" → [397, 1809, 324, 397]  # 罕见，多个token
```

---

## 方法对比

| 方法 | 词汇表 | 序列长度 | OOV | 现状 |
|------|--------|----------|-----|------|
| 字符 | 15万 | 最长 | 无 | 少用 |
| 字节 | 256 | 最长 | 无 | 少用 |
| 词 | 10万+ | 最短 | 严重 | 淘汰 |
| BPE | 3-10万 | 适中 | 无 | 主流 |

---

# Part 3: BPE算法详解

---

## 3.1 训练流程

```
输入: "aaabdaaabac"
初始词汇表: {a, b, c, d} (所有字符)

迭代1: 最频繁的相邻对 = "aa" (出现4次)
      合并: "aa" → "Z"
      序列: "ZabdZabac"
      词汇表: {a, b, c, d, Z}

迭代2: 最频繁的相邻对 = "Za" (出现2次)
      合并: "Za" → "Y"
      序列: "YbdYbac"
      词汇表: {a, b, c, d, Z, Y}

继续直到达到目标词汇表大小...
```

---

## 3.2 核心代码

```python
def train_bpe(text, num_merges):
    # 初始化为字节
    tokens = list(text.encode('utf-8'))
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}

    for i in range(num_merges):
        # 统计相邻对频率
        pairs = Counter(zip(tokens, tokens[1:]))
        if not pairs:
            break

        # 找最频繁的对
        best_pair = pairs.most_common(1)[0][0]

        # 创建新token
        new_id = 256 + i
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges[best_pair] = new_id

        # 应用合并
        tokens = merge(tokens, best_pair, new_id)

    return vocab, merges
```

---

## 3.3 编码过程

```python
def encode(text, merges):
    tokens = list(text.encode('utf-8'))

    # 按训练时的顺序应用合并规则
    for pair, new_id in merges.items():
        tokens = merge(tokens, pair, new_id)

    return tokens

def merge(tokens, pair, new_id):
    result = []
    i = 0
    while i < len(tokens):
        if (i < len(tokens)-1 and
            tokens[i] == pair[0] and
            tokens[i+1] == pair[1]):
            result.append(new_id)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result
```

---

## 3.4 解码过程

```python
def decode(tokens, vocab):
    # 简单: 查表 + 连接 + UTF-8解码
    byte_list = [vocab[t] for t in tokens]
    return b''.join(byte_list).decode('utf-8')
```

---

## 3.5 压缩比

```
定义: 压缩比 = 原始字节数 / token数

越高越好 = 每个token承载更多信息

GPT-2 (50K词汇):
- 英文: ~4.0 (4字节/token)
- 中文: ~1.5 (1.5字节/token) ← 中文吃亏!
- 代码: ~3.5
```

---

# Part 4: 实践考量

---

## 4.1 GPT-2的改进

```python
# Pre-tokenization: 先分割再BPE
GPT2_PATTERN = r"""
    's|'t|'re|'ve|'m|'ll|'d|  # 英语缩写
    \p{L}+|                    # 单词
    \p{N}{1,3}|               # 数字(最多3位)
    ...
"""

# 先按正则分割
segments = regex.findall(GPT2_PATTERN, text)

# 在每个segment内独立做BPE
for segment in segments:
    tokens.extend(bpe_encode(segment))
```

**好处**: 防止跨词合并，提高效率

---

## 4.2 特殊Token

```python
special_tokens = {
    '<|endoftext|>': 50256,
    '<|pad|>': 50257,
    '<|unk|>': 50258,
}

# 编码时保护特殊token，不被BPE分割
```

---

## 4.3 多语言问题

```
同样语义，不同语言token数:

"Hello, how are you?" → 6 tokens
"你好，你怎么样？" → 11 tokens
"Bonjour, comment allez-vous?" → 9 tokens

非英语语言"吃亏":
- 更长序列
- 更少有效上下文
- 更高推理成本
```

---

## 4.4 词汇表大小选择

| 大小 | 优点 | 缺点 |
|------|------|------|
| 10K | Embedding小 | 压缩差 |
| 50K | 平衡 | GPT-2默认 |
| 100K | 压缩好 | Embedding大 |

**趋势**: 100K+ (GPT-4, Claude)

---

# Part 5: 总结

---

## 核心要点

| 概念 | 一句话 |
|------|--------|
| Tokenization | 文本→整数，LLM第一步 |
| BPE | 从数据自动学习词汇表 |
| 压缩比 | 字节/token，越高越好 |
| Pre-tokenization | 先分词再BPE |

---

## 关键数字

```
GPT-2: 50,257 tokens
GPT-4: ~100,000 tokens
压缩比: 英文~4, 中文~1.5
```

---

## 常见误区

| 误区 | 正确理解 |
|------|----------|
| BPE是压缩算法 | BPE是分词算法 |
| 词汇表越大越好 | 有最优点 |
| 所有语言平等 | 非英语吃亏 |

---

## Q&A

### Q: 为什么不用词级tokenization？
**A**: OOV问题。新词、变体、拼写错误都会变成UNK。BPE保证任何输入都能编码。

### Q: Tokenization是LLM的瓶颈吗？
**A**: 某种程度上是。研究表明更好的tokenization能显著提升性能。有人在探索tokenization-free方法。

---

**感谢聆听！**
