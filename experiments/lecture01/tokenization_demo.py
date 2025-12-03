"""
Lecture 01: Tokenization 基础示例
展示不同tokenization策略的实现和对比

主要内容:
1. Character-level tokenization (字符级)
2. Word-level tokenization (单词级)
3. Subword tokenization (子词级: BPE)
4. 性能对比和可视化
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import re


class CharacterTokenizer:
    """字符级Tokenizer - 最简单的tokenization策略"""

    def __init__(self):
        self.vocab = {}
        self.id_to_token = {}

    def build_vocab(self, texts: List[str]):
        """
        构建字符级词表

        Args:
            texts: 训练文本列表
        """
        chars = set()
        for text in texts:
            chars.update(text)

        # 添加特殊token
        self.vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }

        # 添加字符
        for idx, char in enumerate(sorted(chars), start=4):
            self.vocab[char] = idx

        self.id_to_token = {v: k for k, v in self.vocab.items()}

        print(f"✅ Character Tokenizer vocabulary size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """文本 → token IDs"""
        return [self.vocab.get(char, self.vocab['<UNK>']) for char in text]

    def decode(self, ids: List[int]) -> str:
        """Token IDs → 文本"""
        return ''.join([self.id_to_token.get(id, '<UNK>') for id in ids])


class WordTokenizer:
    """单词级Tokenizer - 基于空格分词"""

    def __init__(self):
        self.vocab = {}
        self.id_to_token = {}

    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """
        构建单词级词表

        Args:
            texts: 训练文本列表
            min_freq: 最小词频阈值
        """
        word_counts = Counter()

        for text in texts:
            # 简单的单词分割（实际应用中应使用更复杂的分词）
            words = re.findall(r'\w+|[^\w\s]', text.lower())
            word_counts.update(words)

        # 添加特殊token
        self.vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }

        # 添加高频词
        idx = 4
        for word, count in word_counts.most_common():
            if count >= min_freq:
                self.vocab[word] = idx
                idx += 1

        self.id_to_token = {v: k for k, v in self.vocab.items()}

        print(f"✅ Word Tokenizer vocabulary size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """文本 → token IDs"""
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        return [self.vocab.get(word, self.vocab['<UNK>']) for word in words]

    def decode(self, ids: List[int]) -> str:
        """Token IDs → 文本"""
        tokens = [self.id_to_token.get(id, '<UNK>') for id in ids]
        # 简单重建（实际应该更智能地处理标点和空格）
        return ' '.join(tokens)


class SimpleBPETokenizer:
    """
    简化版BPE (Byte Pair Encoding) Tokenizer

    核心思想:
    1. 从字符级开始
    2. 迭代合并最频繁的token pair
    3. 构建subword词表
    """

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []  # 记录merge操作序列
        self.id_to_token = {}

    def get_stats(self, words: Dict[str, int]) -> Counter:
        """
        统计所有相邻token pair的频率

        Args:
            words: {word: frequency}

        Returns:
            Counter of pairs
        """
        pairs = Counter()

        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq

        return pairs

    def merge_vocab(self, pair: Tuple[str, str], words: Dict[str, int]) -> Dict[str, int]:
        """
        在词表中合并指定的pair

        Args:
            pair: 要合并的token pair
            words: 当前词表

        Returns:
            合并后的词表
        """
        new_words = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word in words:
            new_word = word.replace(bigram, replacement)
            new_words[new_word] = words[word]

        return new_words

    def build_vocab(self, texts: List[str], num_merges: int = 100):
        """
        训练BPE模型

        Args:
            texts: 训练文本
            num_merges: merge操作次数
        """
        print(f"🔧 Training BPE with {num_merges} merges...")

        # Step 1: 初始化为字符级
        word_freqs = Counter()
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            word_freqs.update(words)

        # 在每个字符间添加空格（BPE的标准做法）
        vocab = {' '.join(word) + ' </w>': freq
                 for word, freq in word_freqs.items()}

        # Step 2: 迭代合并最频繁的pair
        for i in range(num_merges):
            pairs = self.get_stats(vocab)

            if not pairs:
                break

            # 找到最频繁的pair
            best_pair = max(pairs, key=pairs.get)

            # 合并
            vocab = self.merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)

            if (i + 1) % 20 == 0:
                print(f"  Merge {i+1}/{num_merges}: {best_pair[0]} + {best_pair[1]} → {''.join(best_pair)}")

        # Step 3: 构建最终词表
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        idx = 4

        for word in vocab.keys():
            for token in word.split():
                if token not in self.vocab:
                    self.vocab[token] = idx
                    idx += 1

        self.id_to_token = {v: k for k, v in self.vocab.items()}

        print(f"✅ BPE Tokenizer vocabulary size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """
        文本 → token IDs (应用learned merges)
        """
        words = re.findall(r'\w+', text.lower())

        encoded = []
        for word in words:
            # 初始化为字符级
            tokens = list(word) + ['</w>']

            # 应用learned merges
            for pair in self.merges:
                i = 0
                while i < len(tokens) - 1:
                    if (tokens[i], tokens[i+1]) == pair:
                        tokens = tokens[:i] + [''.join(pair)] + tokens[i+2:]
                    else:
                        i += 1

            # 转换为IDs
            for token in tokens:
                encoded.append(self.vocab.get(token, self.vocab['<UNK>']))

        return encoded

    def decode(self, ids: List[int]) -> str:
        """Token IDs → 文本"""
        tokens = [self.id_to_token.get(id, '<UNK>') for id in ids]
        text = ''.join(tokens).replace('</w>', ' ').strip()
        return text


def compare_tokenizers():
    """对比不同tokenizer的性能"""

    print("=" * 70)
    print("🔬 Tokenization 策略对比实验")
    print("=" * 70)

    # 准备测试数据
    train_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "Where there is a will, there is a way.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning models require large amounts of data.",
        "Natural language processing enables computers to understand human language.",
        "Tokenization is the first step in text preprocessing.",
        "Subword tokenization balances vocabulary size and coverage."
    ]

    test_text = "Machine learning and deep learning are subsets of artificial intelligence."

    print(f"\n📝 Training corpus: {len(train_texts)} sentences")
    print(f"📝 Test sentence: \"{test_text}\"")

    # 1. Character-level Tokenizer
    print("\n" + "-" * 70)
    print("1️⃣  Character-level Tokenizer")
    print("-" * 70)
    char_tokenizer = CharacterTokenizer()
    char_tokenizer.build_vocab(train_texts)

    char_encoded = char_tokenizer.encode(test_text)
    char_decoded = char_tokenizer.decode(char_encoded)

    print(f"Encoded length: {len(char_encoded)} tokens")
    print(f"Sample tokens: {char_encoded[:20]}...")
    print(f"Decoded: \"{char_decoded}\"")
    print(f"Match: {'✅' if char_decoded == test_text else '❌'}")

    # 2. Word-level Tokenizer
    print("\n" + "-" * 70)
    print("2️⃣  Word-level Tokenizer")
    print("-" * 70)
    word_tokenizer = WordTokenizer()
    word_tokenizer.build_vocab(train_texts, min_freq=1)

    word_encoded = word_tokenizer.encode(test_text)
    word_decoded = word_tokenizer.decode(word_encoded)

    print(f"Encoded length: {len(word_encoded)} tokens")
    print(f"Sample tokens: {word_encoded}")
    print(f"Decoded: \"{word_decoded}\"")

    # 3. BPE Tokenizer
    print("\n" + "-" * 70)
    print("3️⃣  BPE (Byte Pair Encoding) Tokenizer")
    print("-" * 70)
    bpe_tokenizer = SimpleBPETokenizer()
    bpe_tokenizer.build_vocab(train_texts, num_merges=50)

    bpe_encoded = bpe_tokenizer.encode(test_text)
    bpe_decoded = bpe_tokenizer.decode(bpe_encoded)

    print(f"Encoded length: {len(bpe_encoded)} tokens")
    print(f"Sample tokens: {bpe_encoded[:20]}...")
    print(f"Decoded: \"{bpe_decoded}\"")

    # 4. 综合对比
    print("\n" + "=" * 70)
    print("📊 Performance Comparison")
    print("=" * 70)

    comparison_table = f"""
    {'Tokenizer':<20} {'Vocab Size':<15} {'Token Count':<15} {'Compression':<15}
    {'-' * 70}
    {'Character-level':<20} {len(char_tokenizer.vocab):<15} {len(char_encoded):<15} {len(test_text)/len(char_encoded):.2f}x
    {'Word-level':<20} {len(word_tokenizer.vocab):<15} {len(word_encoded):<15} {len(test_text)/len(word_encoded):.2f}x
    {'BPE':<20} {len(bpe_tokenizer.vocab):<15} {len(bpe_encoded):<15} {len(test_text)/len(bpe_encoded):.2f}x
    """

    print(comparison_table)

    # 5. OOV (Out of Vocabulary) 测试
    print("\n" + "=" * 70)
    print("🔍 OOV (Out of Vocabulary) Handling Test")
    print("=" * 70)

    oov_text = "Supercalifragilisticexpialidocious"  # 未见过的词

    print(f"Test word: \"{oov_text}\"")

    print(f"\n  Character-level: {len(char_tokenizer.encode(oov_text))} tokens (handles gracefully ✅)")

    word_encoded_oov = word_tokenizer.encode(oov_text)
    unk_count = sum(1 for id in word_encoded_oov if id == word_tokenizer.vocab['<UNK>'])
    print(f"  Word-level: {unk_count} <UNK> tokens (poor handling ❌)")

    bpe_encoded_oov = bpe_tokenizer.encode(oov_text)
    print(f"  BPE: {len(bpe_encoded_oov)} subword tokens (good handling ✅)")

    # 6. 总结
    print("\n" + "=" * 70)
    print("💡 Key Insights")
    print("=" * 70)
    print("""
    1. Character-level:
       ✅ 词表小 (vocab size ~100)
       ✅ 无OOV问题
       ❌ 序列长，计算量大
       ❌ 难以学习单词语义

    2. Word-level:
       ✅ 序列短，计算高效
       ✅ 直接对应单词语义
       ❌ 词表巨大 (vocab size ~50K-100K)
       ❌ OOV问题严重

    3. BPE (Subword):
       ✅ 平衡词表大小和序列长度
       ✅ 处理OOV能力强
       ✅ 学习到morphology（词法）
       ✅ 当前主流方案（GPT, BERT等）

    🏆 Winner: BPE / Subword tokenization
    """)


def demonstrate_bpe_process():
    """详细演示BPE算法过程"""

    print("\n" + "=" * 70)
    print("🔬 Detailed BPE Algorithm Demonstration")
    print("=" * 70)

    # 简单示例
    corpus = ["low", "lower", "newest", "widest"]

    print(f"\nCorpus: {corpus}")
    print("\n📝 Step-by-step BPE training:\n")

    # 初始化
    vocab = {' '.join(word) + ' </w>': 1 for word in corpus}

    print("Initial vocabulary (character-level):")
    for word in vocab:
        print(f"  {word}")

    # 手动演示几次merge
    print("\n" + "-" * 70)
    print("Merge iterations:")
    print("-" * 70)

    tokenizer = SimpleBPETokenizer()

    for iteration in range(5):
        pairs = tokenizer.get_stats(vocab)

        if not pairs:
            break

        best_pair = max(pairs, key=pairs.get)

        print(f"\nIteration {iteration + 1}:")
        print(f"  Most frequent pair: {best_pair[0]} + {best_pair[1]} (count: {pairs[best_pair]})")
        print(f"  Before merge:")
        for word in list(vocab.keys())[:3]:
            print(f"    {word}")

        vocab = tokenizer.merge_vocab(best_pair, vocab)

        print(f"  After merge ({best_pair[0]}{best_pair[1]}):")
        for word in list(vocab.keys())[:3]:
            print(f"    {word}")


if __name__ == '__main__':
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                                                                    ║")
    print("║           CS336 Lecture 01 - Tokenization Demo                    ║")
    print("║                                                                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    # 主要对比实验
    compare_tokenizers()

    # 详细BPE演示
    demonstrate_bpe_process()

    print("\n" + "=" * 70)
    print("✅ Tokenization Demo完成！")
    print("=" * 70)
    print("\n💡 Further Reading:")
    print("  - Sennrich et al. (2016): Neural Machine Translation of Rare Words with Subword Units")
    print("  - HuggingFace Tokenizers: https://huggingface.co/docs/tokenizers")
    print("\n")
