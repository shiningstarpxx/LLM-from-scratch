#!/usr/bin/env python3
"""
BPE算法可视化演示
形象化展示Byte Pair Encoding的完整过程
"""

import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BPEStep:
    """BPE训练过程的一步记录"""
    iteration: int
    pair: Tuple[int, int]
    new_token: int
    frequency: int
    sequence_before: List[int]
    sequence_after: List[int]
    pair_counts: Dict[Tuple[int, int], int]


class BPEVisualizer:
    """BPE算法可视化工具"""

    def __init__(self):
        self.steps: List[BPEStep] = []
        self.vocab: Dict[int, bytes] = {}
        self.merges: Dict[Tuple[int, int], int] = {}

    def train_with_visualization(self, text: str, num_merges: int) -> None:
        """训练BPE并记录每一步的可视化信息"""
        print("🚀 BPE算法可视化演示")
        print("=" * 60)
        print(f"📝 训练文本: '{text}'")
        print(f"🔢 目标合并次数: {num_merges}")
        print()

        # 初始化
        indices = list(map(int, text.encode("utf-8")))
        self.vocab = {x: bytes([x]) for x in range(256)}

        print("📊 初始状态:")
        self._show_sequence(indices, "初始字节序列")
        self._show_byte_representation(indices)
        print()

        # 训练循环
        for i in range(num_merges):
            print(f"🔄 第 {i+1} 轮合并:")
            print("-" * 40)

            # 统计相邻对频率
            counts = self._count_pairs(indices)
            self._show_pair_counts(counts)

            if not counts:
                print("⏹️ 没有更多可以合并的token对")
                break

            # 找到最频繁的token对
            pair = max(counts, key=counts.get)
            frequency = counts[pair]
            new_token = 256 + i

            # 记录合并前的状态
            sequence_before = indices.copy()

            # 执行合并
            indices = self._merge(indices, pair, new_token)

            # 更新词汇表和合并规则
            self.vocab[new_token] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merges[pair] = new_token

            # 记录这一步
            step = BPEStep(
                iteration=i+1,
                pair=pair,
                new_token=new_token,
                frequency=frequency,
                sequence_before=sequence_before,
                sequence_after=indices.copy(),
                pair_counts=counts.copy()
            )
            self.steps.append(step)

            # 可视化这一步的结果
            self._visualize_step(step)

            print()

        print("🎉 训练完成!")
        print(f"📚 最终词汇表大小: {len(self.vocab)}")
        print(f"🔗 合并规则数量: {len(self.merges)}")
        print(f"📏 最终序列长度: {len(indices)}")

    def _count_pairs(self, indices: List[int]) -> Dict[Tuple[int, int], int]:
        """统计相邻token对的频率"""
        counts = defaultdict(int)
        for i in range(len(indices) - 1):
            counts[(indices[i], indices[i+1])] += 1
        return dict(counts)

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

    def _show_sequence(self, indices: List[int], title: str) -> None:
        """显示token序列"""
        print(f"   {title}:")
        print(f"   Tokens: {indices}")
        print(f"   长度: {len(indices)}")

    def _show_byte_representation(self, indices: List[int]) -> None:
        """显示字节对应的字符表示"""
        chars = []
        bytes_repr = []
        for idx in indices:
            if idx < 256:
                char = chr(idx) if 32 <= idx <= 126 else f"[{idx}]"
                chars.append(char)
                bytes_repr.append(f"{idx:3d}")
            else:
                chars.append(f"[{idx}]")
                bytes_repr.append(f"{idx:3d}")

        print(f"   字符: {' '.join(chars)}")
        print(f"   字节: {' '.join(bytes_repr)}")

    def _show_pair_counts(self, counts: Dict[Tuple[int, int], int]) -> None:
        """显示token对频率统计"""
        print("   📈 相邻token对频率:")
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for i, (pair, freq) in enumerate(sorted_counts[:10]):  # 只显示前10个
            pair_str = self._format_pair(pair)
            print(f"      {i+1:2d}. {pair_str} -> {freq} 次")

        if len(sorted_counts) > 10:
            print(f"      ... 还有 {len(sorted_counts) - 10} 个token对")

    def _format_pair(self, pair: Tuple[int, int]) -> str:
        """格式化token对为可读形式"""
        def format_token(token: int) -> str:
            if token < 256:
                char = chr(token) if 32 <= token <= 126 else f"[{token}]"
                return f"{token}({char})"
            else:
                return f"{token}[{self._get_token_content(token)}]"

        return f"({format_token(pair[0])}, {format_token(pair[1])})"

    def _get_token_content(self, token: int) -> str:
        """获取token的字节内容的字符串表示"""
        if token in self.vocab:
            try:
                return self.vocab[token].decode('utf-8', errors='replace')
            except:
                return str(self.vocab[token])
        return "?"

    def _visualize_step(self, step: BPEStep) -> None:
        """可视化单步合并过程"""
        pair_str = self._format_pair(step.pair)
        new_content = self._get_token_content(step.new_token)

        print(f"   🎯 选择合并: {pair_str}")
        print(f"   📊 出现频率: {step.frequency} 次")
        print(f"   🔧 创建新token: {step.new_token}[{new_content}]")

        print("\n   🔄 合并过程可视化:")
        self._show_merge_animation(step.sequence_before, step.pair, step.new_token)

        print(f"\n   📏 合并效果:")
        print(f"      合并前长度: {len(step.sequence_before)}")
        print(f"      合并后长度: {len(step.sequence_after)}")
        print(f"      压缩效果: {len(step.sequence_before) - len(step.sequence_after)} 个token")

    def _show_merge_animation(self, sequence: List[int], pair: Tuple[int, int], new_token: int) -> None:
        """显示合并过程的动画效果"""
        # 显示原始序列，标记要合并的位置
        marked_sequence = []
        i = 0
        while i < len(sequence):
            if (i + 1 < len(sequence) and
                sequence[i] == pair[0] and
                sequence[i + 1] == pair[1]):
                marked_sequence.append(f"🔴{sequence[i]}🔴{sequence[i+1]}")
                i += 2
            else:
                marked_sequence.append(f"{sequence[i]}")
                i += 1

        print(f"      原始: {' '.join(marked_sequence)}")

        # 显示合并后的序列
        result_sequence = []
        i = 0
        while i < len(sequence):
            if (i + 1 < len(sequence) and
                sequence[i] == pair[0] and
                sequence[i + 1] == pair[1]):
                result_sequence.append(f"🟢{new_token}")
                i += 2
            else:
                result_sequence.append(f"{sequence[i]}")
                i += 1

        print(f"      合并: {' '.join(result_sequence)}")

    def show_final_summary(self) -> None:
        """显示训练总结"""
        print("\n" + "=" * 60)
        print("📊 BPE训练总结")
        print("=" * 60)

        print(f"\n🔗 合并规则总览:")
        for i, step in enumerate(self.steps):
            pair_str = self._format_pair(step.pair)
            new_content = self._get_token_content(step.new_token)
            print(f"   {i+1:2d}. {pair_str} -> {step.new_token}[{new_content}] (频率: {step.frequency})")

        print(f"\n📚 词汇表演进:")
        print(f"   初始词汇表: 256 个字节token")
        for i, step in enumerate(self.steps):
            print(f"   第{i+1}步: 添加 token {step.new_token} = '{self._get_token_content(step.new_token)}'")

        print(f"\n🎯 压缩效果:")
        if self.steps:
            initial_length = len(self.steps[0].sequence_before)
            final_length = len(self.steps[-1].sequence_after)
            compression_ratio = initial_length / final_length
            print(f"   初始序列长度: {initial_length}")
            print(f"   最终序列长度: {final_length}")
            print(f"   压缩比: {compression_ratio:.2f}")


def demo_different_texts():
    """演示不同文本的BPE过程"""
    visualizer = BPEVisualizer()

    # 示例1: 简单重复模式
    print("\n" + "🎬" * 20)
    print("示例1: 简单重复模式")
    print("🎬" * 20)
    visualizer.train_with_visualization("aaabdaaabac", num_merges=5)
    visualizer.show_final_summary()

    # 示例2: 英文单词
    print("\n" + "🎬" * 20)
    print("示例2: 英文单词")
    print("🎬" * 20)
    visualizer2 = BPEVisualizer()
    visualizer2.train_with_visualization("the cat in the hat", num_merges=8)
    visualizer2.show_final_summary()

    # 示例3: 混合内容
    print("\n" + "🎬" * 20)
    print("示例3: 混合内容")
    print("🎬" * 20)
    visualizer3 = BPEVisualizer()
    visualizer3.train_with_visualization("hello hello world", num_merges=6)
    visualizer3.show_final_summary()


def interactive_demo():
    """交互式演示"""
    print("\n🎮 交互式BPE演示")
    print("输入你想要测试的文本（或按回车使用默认文本）:")

    user_input = input("> ").strip()
    if not user_input:
        user_input = "low low low lower lowest lowest newer newer newer newest newest"

    print(f"输入文本: '{user_input}'")
    print("输入合并次数（或按回车使用默认值8）:")

    try:
        num_merges = int(input("> ").strip())
    except:
        num_merges = 8

    visualizer = BPEVisualizer()
    visualizer.train_with_visualization(user_input, num_merges)
    visualizer.show_final_summary()


if __name__ == "__main__":
    print("🎨 BPE算法可视化演示工具")
    print("选择演示模式:")
    print("1. 预设示例演示")
    print("2. 交互式演示")

    choice = input("请选择 (1/2): ").strip()

    if choice == "2":
        interactive_demo()
    else:
        demo_different_texts()