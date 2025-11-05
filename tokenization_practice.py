#!/usr/bin/env python3
"""
Tokenization实践练习
基于Lecture 01的内容，实现和测试不同的tokenizer
"""

import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict, Tuple


class Tokenizer(ABC):
    """Tokenizer抽象基类"""
    @abstractmethod
    def encode(self, string: str) -> List[int]:
        """将字符串编码为token序列"""
        pass

    @abstractmethod
    def decode(self, indices: List[int]) -> str:
        """将token序列解码为字符串"""
        pass


class CharacterTokenizer(Tokenizer):
    """字符级tokenizer"""
    def encode(self, string: str) -> List[int]:
        return list(map(ord, string))

    def decode(self, indices: List[int]) -> str:
        return "".join(map(chr, indices))


class ByteTokenizer(Tokenizer):
    """字节级tokenizer"""
    def encode(self, string: str) -> List[int]:
        string_bytes = string.encode("utf-8")
        return list(map(int, string_bytes))

    def decode(self, indices: List[int]) -> str:
        string_bytes = bytes(indices)
        return string_bytes.decode("utf-8")


class BPETokenizerParams:
    """BPE tokenizer参数"""
    def __init__(self, vocab: Dict[int, bytes], merges: Dict[Tuple[int, int], int]):
        self.vocab = vocab          # index -> bytes
        self.merges = merges        # (index1, index2) -> new_index


def merge(indices: List[int], pair: Tuple[int, int], new_index: int) -> List[int]:
    """将indices中的pair替换为new_index"""
    new_indices = []
    i = 0
    while i < len(indices):
        if (i + 1 < len(indices) and
            indices[i] == pair[0] and
            indices[i + 1] == pair[1]):
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:
    """训练BPE tokenizer"""
    print(f"训练BPE tokenizer，文本长度: {len(string)} 字符")

    # 初始化：将字符串转换为字节序列
    indices = list(map(int, string.encode("utf-8")))
    print(f"初始字节序列长度: {len(indices)}")

    # 初始化词汇表和合并规则
    vocab = {x: bytes([x]) for x in range(256)}
    merges = {}

    for i in range(num_merges):
        # 统计相邻token对频率
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):
            counts[(index1, index2)] += 1

        if not counts:
            break

        # 找到最频繁的token对
        pair = max(counts, key=counts.get)
        new_index = 256 + i

        print(f"第{i+1}次合并: {pair} -> {new_index} (频率: {counts[pair]})")

        # 记录合并规则
        merges[pair] = new_index
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]

        # 应用合并
        indices = merge(indices, pair, new_index)
        print(f"合并后序列长度: {len(indices)}")

    print(f"训练完成，词汇表大小: {len(vocab)}")
    return BPETokenizerParams(vocab=vocab, merges=merges)


class BPETokenizer(Tokenizer):
    """BPE tokenizer实现"""
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> List[int]:
        # 转换为字节序列
        indices = list(map(int, string.encode("utf-8")))

        # 应用所有合并规则（简单但低效的实现）
        for pair, new_index in self.params.merges.items():
            indices = merge(indices, pair, new_index)

        return indices

    def decode(self, indices: List[int]) -> str:
        # 将token索引转换为字节
        bytes_list = [self.params.vocab[idx] for idx in indices]
        # 解码为字符串
        return b"".join(bytes_list).decode("utf-8")


def get_compression_ratio(string: str, indices: List[int]) -> float:
    """计算压缩比"""
    num_bytes = len(bytes(string, encoding="utf-8"))
    num_tokens = len(indices)
    return num_bytes / num_tokens


def test_tokenizer(tokenizer: Tokenizer, name: str, test_strings: List[str]):
    """测试tokenizer性能"""
    print(f"\n=== 测试 {name} ===")

    for string in test_strings:
        try:
            # 编码
            indices = tokenizer.encode(string)
            # 解码
            reconstructed = tokenizer.decode(indices)
            # 验证
            success = string == reconstructed
            # 压缩比
            ratio = get_compression_ratio(string, indices)

            print(f"原文: {string[:30]}{'...' if len(string) > 30 else ''}")
            print(f"Tokens: {indices[:10]}{'...' if len(indices) > 10 else ''}")
            print(f"Token数量: {len(indices)}, 压缩比: {ratio:.2f}")
            print(f"往返测试: {'✅ 通过' if success else '❌ 失败'}")
            print()

        except Exception as e:
            print(f"处理 '{string}' 时出错: {e}")
            print()


def main():
    """主函数：演示不同tokenizer的效果"""
    print("🚀 Tokenization实践演示")
    print("=" * 50)

    # 测试字符串
    test_strings = [
        "Hello, world!",
        "Hello, 🌍! 你好!",
        "the cat in the hat",
        "supercalifragilisticexpialidocious",
        "I'll say this is amazing!"
    ]

    # 1. 测试字符级tokenizer
    char_tokenizer = CharacterTokenizer()
    test_tokenizer(char_tokenizer, "Character-based Tokenizer", test_strings)

    # 2. 测试字节级tokenizer
    byte_tokenizer = ByteTokenizer()
    test_tokenizer(byte_tokenizer, "Byte-based Tokenizer", test_strings)

    # 3. 训练和测试BPE tokenizer
    print("\n" + "=" * 50)
    print("🔧 训练BPE Tokenizer")
    print("=" * 50)

    # 使用简单的训练文本
    training_text = "the cat in the hat the cat sat on the mat the quick brown fox"
    bpe_params = train_bpe(training_text, num_merges=10)

    bpe_tokenizer = BPETokenizer(bpe_params)
    test_tokenizer(bpe_tokenizer, "BPE Tokenizer", test_strings)

    # 4. 演示BPE训练过程
    print("\n" + "=" * 50)
    print("📊 BPE训练过程详细演示")
    print("=" * 50)

    simple_text = "aaabdaaabac"
    print(f"训练文本: {simple_text}")

    # 手动演示前几步合并
    indices = list(map(int, simple_text.encode("utf-8")))
    print(f"初始字节: {indices}")
    print(f"对应字符: {[chr(b) for b in indices]}")

    # 统计相邻对
    counts = defaultdict(int)
    for i in range(len(indices)-1):
        counts[(indices[i], indices[i+1])] += 1

    print(f"相邻对频率: {dict(counts)}")
    if counts:
        most_common = max(counts, key=counts.get)
        print(f"最频繁的对: {most_common} (出现 {counts[most_common]} 次)")


if __name__ == "__main__":
    main()