#!/usr/bin/env python3
"""
🎯 BPE (Byte Pair Encoding) 核心算法实现
===========================================

简洁清晰的BPE实现，专注于算法本质，便于理解和学习。

作者: Claude Code
日期: 2025-11-06
用途: Lecture 01 - Tokenization原理与实践核心代码示例
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class BPETokenizer:
    """BPE Tokenizer核心实现"""

    def __init__(self):
        self.vocab: Dict[int, bytes] = {}
        self.merges: Dict[Tuple[int, int], int] = {}
        self._init_vocab()

    def _init_vocab(self):
        """初始化词汇表为单字节"""
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, num_merges: int) -> None:
        """
        训练BPE tokenizer

        Args:
            text: 训练文本
            num_merges: 合并次数
        """
        print(f"🚀 开始BPE训练: '{text}' (合并{num_merges}次)")

        # 1. 将文本转换为字节序列
        indices = list(map(int, text.encode("utf-8")))
        print(f"📝 初始token序列: {indices}")

        # 2. 迭代合并
        for i in range(num_merges):
            # 统计相邻token对频率
            pair_counts = self._count_pairs(indices)
            if not pair_counts:
                break

            # 找到最频繁的对
            most_frequent_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
            new_token = 256 + i

            print(f"步骤{i+1}: 合并 {most_frequent_pair} → {new_token} "
                  f"(频率: {pair_counts[most_frequent_pair]})")

            # 执行合并
            indices = self._merge(indices, most_frequent_pair, new_token)

            # 记录合并规则和更新词汇表
            self.merges[most_frequent_pair] = new_token
            self.vocab[new_token] = self.vocab[most_frequent_pair[0]] + self.vocab[most_frequent_pair[1]]

            print(f"  合并后: {indices}")

        print(f"✅ 训练完成! 最终序列: {indices}")
        print(f"📊 词汇表大小: {len(self.vocab)}, 合并规则数: {len(self.merges)}")

    def _count_pairs(self, indices: List[int]) -> Dict[Tuple[int, int], int]:
        """统计相邻token对频率"""
        counts = defaultdict(int)
        for i in range(len(indices) - 1):
            pair = (indices[i], indices[i + 1])
            counts[pair] += 1
        return counts

    def _merge(self, indices: List[int], pair: Tuple[int, int], new_token: int) -> List[int]:
        """合并指定的token对"""
        new_indices = []
        i = 0
        while i < len(indices):
            if (i + 1 < len(indices) and
                indices[i] == pair[0] and
                indices[i + 1] == pair[1]):
                new_indices.append(new_token)
                i += 2
            else:
                new_indices.append(indices[i])
                i += 1
        return new_indices

    def encode(self, text: str) -> List[int]:
        """编码文本为token序列"""
        # 1. 转换为字节序列
        indices = list(map(int, text.encode("utf-8")))

        # 2. 应用所有合并规则
        for pair, new_token in self.merges.items():
            indices = self._merge(indices, pair, new_token)

        return indices

    def decode(self, indices: List[int]) -> str:
        """解码token序列为文本"""
        # 1. 将token转换为字节
        bytes_list = [self.vocab[idx] for idx in indices]

        # 2. 解码为字符串
        return b"".join(bytes_list).decode("utf-8")

    def print_vocab(self):
        """打印词汇表"""
        print("\n📚 词汇表:")
        print("Token | Bytes")
        print("-" * 30)

        # 只显示新创建的tokens
        for token in sorted(self.vocab.keys()):
            if token >= 256:  # 新创建的tokens
                print(f"{token:5d} | {self.vocab[token]!r}")

    def print_merges(self):
        """打印合并规则"""
        print("\n🔗 合并规则:")
        print("步骤 | 合并对      → 新Token | Bytes")
        print("-" * 50)

        for i, (pair, new_token) in enumerate(self.merges.items()):
            bytes_repr = f"{self.vocab[pair[0]]!r}+{self.vocab[pair[1]]!r}"
            print(f"{i+1:4d} | [{pair[0]}]+[{pair[1]:3d}]   → [{new_token:3d}]   | {bytes_repr}")


def simple_example():
    """简单示例：aaabdaaabac"""
    print("🎯 示例1: aaabdaaabac")
    print("-" * 50)

    tokenizer = BPETokenizer()
    tokenizer.train("aaabdaaabac", num_merges=5)

    print("\n📋 训练结果:")
    tokenizer.print_merges()
    tokenizer.print_vocab()

    # 测试编码
    test_text = "aaaa"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"\n🔍 测试编码: '{test_text}'")
    print(f"   编码: {encoded}")
    print(f"   解码: '{decoded}'")
    print(f"   ✅ 成功: {test_text == decoded}")


def english_example():
    """英文示例：hello world hello"""
    print("\n🎯 示例2: hello world hello")
    print("-" * 50)

    tokenizer = BPETokenizer()
    tokenizer.train("hello world hello", num_merges=8)

    print("\n📋 训练结果:")
    tokenizer.print_merges()

    # 测试编码
    test_text = "hello"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"\n🔍 测试编码: '{test_text}'")
    print(f"   编码: {encoded}")
    print(f"   解码: '{decoded}'")
    print(f"   ✅ 成功: {test_text == decoded}")


def chinese_example():
    """中文示例：你好世界"""
    print("\n🎯 示例3: 你好世界")
    print("-" * 50)

    tokenizer = BPETokenizer()
    tokenizer.train("你好世界", num_merges=6)

    print("\n📋 训练结果:")
    tokenizer.print_merges()
    tokenizer.print_vocab()

    # 测试编码
    test_text = "你好"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"\n🔍 测试编码: '{test_text}'")
    print(f"   编码: {encoded}")
    print(f"   解码: '{decoded}'")
    print(f"   ✅ 成功: {test_text == decoded}")


def compression_analysis():
    """压缩效果分析"""
    print("\n📊 压缩效果分析")
    print("-" * 50)

    texts = [
        "aaaaaa",           # 重复字符
        "hello world",      # 英文
        "你好世界你好",      # 中文
        "abcabcabc",        # 重复模式
    ]

    for text in texts:
        print(f"\n📝 文本: '{text}'")

        # 原始字节长度
        original_bytes = len(text.encode("utf-8"))
        print(f"   原始字节: {original_bytes}")

        # 训练BPE
        tokenizer = BPETokenizer()
        tokenizer.train(text, num_merges=min(10, original_bytes // 2))

        # 编码后长度
        encoded = tokenizer.encode(text)
        compressed_length = len(encoded)
        print(f"   BPE编码: {compressed_length}")

        # 压缩比
        compression_ratio = original_bytes / compressed_length
        print(f"   压缩比: {compression_ratio:.3f}")


def main():
    """主函数"""
    print("🎯 BPE核心算法演示")
    print("=" * 60)

    # 运行示例
    simple_example()
    english_example()
    chinese_example()
    compression_analysis()

    print("\n" + "=" * 60)
    print("✅ 演示完成!")
    print("\n💡 关键洞察:")
    print("1. BPE从字节开始，逐步合并高频对")
    print("2. 常见模式被压缩为单个token")
    print("3. 压缩效果取决于文本的重复模式")
    print("4. 词汇表大小可通过合并次数控制")


if __name__ == "__main__":
    main()