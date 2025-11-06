#!/usr/bin/env python3
"""
🎯 BPE (Byte Pair Encoding) 形象化演示工具
===========================================

这个脚本通过可视化演示帮助理解BPE算法的工作原理。
包含逐步演示、动画效果和详细的统计信息。

作者: Claude Code
日期: 2025-11-06
用途: Lecture 01 - Tokenization原理与实践补充材料
"""

import json
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import argparse


class BPEVisualizer:
    """BPE算法可视化演示器"""

    def __init__(self, text: str, num_merges: int = 10, delay: float = 0.5):
        self.original_text = text
        self.num_merges = num_merges
        self.delay = delay

        # 初始化状态
        self.reset()

    def reset(self):
        """重置到初始状态"""
        # 将文本转换为字节序列
        self.indices = list(map(int, self.original_text.encode("utf-8")))
        self.vocab = {x: bytes([x]) for x in range(256)}  # 0-255 → 单字节
        self.merges = {}  # (token1, token2) → new_token
        self.merge_history = []  # 记录合并历史
        self.step = 0

    def print_separator(self, title: str = ""):
        """打印分隔线"""
        print("\n" + "="*80)
        if title:
            print(f"🎯 {title}")
            print("="*80)

    def print_tokens(self, indices: List[int], label: str = ""):
        """美化打印token序列"""
        if label:
            print(f"\n📝 {label}:")

        # 显示token序列
        tokens_str = " → ".join([f"[{idx}]" for idx in indices])
        print(f"   Tokens: {tokens_str}")

        # 显示对应的字节
        bytes_str = " → ".join([f"{self.vocab[idx]!r}" for idx in indices])
        print(f"   Bytes:  {bytes_str}")

        # 显示解码后的文本（如果可读）
        try:
            decoded = b"".join([self.vocab[idx] for idx in indices]).decode("utf-8")
            if decoded.isprintable():
                print(f"   Text:   {decoded!r}")
        except:
            print(f"   Text:   <非UTF-8序列>")

    def find_most_frequent_pair(self, indices: List[int]) -> Tuple[Tuple[int, int], int]:
        """找到最频繁的相邻token对"""
        counts = defaultdict(int)

        # 统计相邻对频率
        for i in range(len(indices) - 1):
            pair = (indices[i], indices[i + 1])
            counts[pair] += 1

        if not counts:
            return None, 0

        # 找到最频繁的对
        most_frequent = max(counts.items(), key=lambda x: x[1])
        return most_frequent  # ((token1, token2), frequency)

    def merge_pair(self, indices: List[int], pair: Tuple[int, int], new_token: int) -> List[int]:
        """合并token序列中的指定对"""
        new_indices = []
        i = 0

        while i < len(indices):
            if (i + 1 < len(indices) and
                indices[i] == pair[0] and
                indices[i + 1] == pair[1]):
                new_indices.append(new_token)
                i += 2  # 跳过已合并的两个token
            else:
                new_indices.append(indices[i])
                i += 1

        return new_indices

    def visualize_pair_frequencies(self, indices: List[int]):
        """可视化token对频率统计"""
        counts = defaultdict(int)

        for i in range(len(indices) - 1):
            pair = (indices[i], indices[i + 1])
            counts[pair] += 1

        if not counts:
            print("   📊 无相邻token对")
            return

        # 按频率排序
        sorted_pairs = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        print(f"\n📊 Token对频率统计 (共{len(sorted_pairs)}种不同的对):")
        print("   " + "-"*60)
        print("   排名 | Token对          | 频率 | 字节表示")
        print("   " + "-"*60)

        for i, (pair, freq) in enumerate(sorted_pairs[:10]):  # 只显示前10个
            token1, token2 = pair
            bytes_repr = f"{self.vocab[token1]!r}+{self.vocab[token2]!r}"
            print(f"   #{i+1:2d}  | [{token1}]+[{token2:3d}]      | {freq:3d}  | {bytes_repr}")

        if len(sorted_pairs) > 10:
            print(f"   ... 还有{len(sorted_pairs)-10}个token对")

    def step_visualize(self):
        """单步可视化BPE合并过程"""
        self.step += 1

        self.print_separator(f"第 {self.step} 步合并")

        # 显示当前状态
        self.print_tokens(self.indices, f"当前token序列 (长度: {len(self.indices)})")

        # 显示token对频率统计
        self.visualize_pair_frequencies(self.indices)

        # 找到最频繁的对
        most_frequent_pair, frequency = self.find_most_frequent_pair(self.indices)

        if most_frequent_pair is None:
            print("\n❌ 没有可以合并的token对")
            return False

        token1, token2 = most_frequent_pair
        new_token = 256 + len(self.merges)  # 新token索引

        print(f"\n🎯 选择最频繁的token对: [{token1}]+[{token2}] (出现{frequency}次)")
        print(f"🆕 创建新token: [{new_token}] = {self.vocab[token1]!r}+{self.vocab[token2]!r}")

        # 等待用户确认（如果设置了延迟）
        if self.delay > 0:
            time.sleep(self.delay)

        # 执行合并
        old_length = len(self.indices)
        self.indices = self.merge_pair(self.indices, most_frequent_pair, new_token)
        new_length = len(self.indices)

        # 更新词汇表和合并规则
        self.merges[most_frequent_pair] = new_token
        self.vocab[new_token] = self.vocab[token1] + self.vocab[token2]

        # 记录历史
        self.merge_history.append({
            'step': self.step,
            'pair': most_frequent_pair,
            'new_token': new_token,
            'frequency': frequency,
            'old_length': old_length,
            'new_length': new_length,
            'compression_ratio': old_length / new_length
        })

        # 显示合并结果
        self.print_tokens(self.indices, f"合并后token序列 (长度: {len(self.indices)})")

        # 显示压缩效果
        compression = old_length / new_length
        print(f"\n📈 压缩效果:")
        print(f"   合并前长度: {old_length}")
        print(f"   合并后长度: {new_length}")
        print(f"   压缩比: {compression:.3f}")

        return True

    def run_visualization(self):
        """运行完整的可视化过程"""
        print("🚀 BPE算法可视化演示开始")
        print(f"📝 原始文本: {self.original_text!r}")
        print(f"🎯 目标合并次数: {self.num_merges}")

        # 显示初始状态
        self.print_separator("初始状态")
        self.print_tokens(self.indices, f"初始token序列 (长度: {len(self.indices)})")

        print(f"\n💡 初始词汇表大小: {len(self.vocab)} (0-255的单字节)")

        # 逐步合并
        for i in range(self.num_merges):
            if not self.step_visualize():
                break

        # 最终总结
        self.print_final_summary()

    def print_final_summary(self):
        """打印最终总结"""
        self.print_separator("🎉 BPE训练完成总结")

        print(f"📊 训练统计:")
        print(f"   原始文本长度: {len(self.original_text)} 字符")
        print(f"   初始token数: {len(list(map(int, self.original_text.encode('utf-8'))))}")
        print(f"   最终token数: {len(self.indices)}")
        print(f"   总合并步数: {len(self.merge_history)}")
        print(f"   最终词汇表大小: {len(self.vocab)}")

        # 压缩比
        initial_length = len(list(map(int, self.original_text.encode('utf-8'))))
        final_compression = initial_length / len(self.indices)
        print(f"   总压缩比: {final_compression:.3f}")

        # 显示合并历史
        print(f"\n📜 合并历史:")
        print("   步骤 | 合并对        | 新token | 压缩比")
        print("   " + "-"*40)

        for record in self.merge_history:
            pair = record['pair']
            print(f"   #{record['step']:2d}  | [{pair[0]}]+[{pair[1]:3d}]    | [{record['new_token']:3d}]   | {record['compression_ratio']:.3f}")

        # 显示最终词汇表（新创建的tokens）
        print(f"\n📚 新创建的词汇:")
        for record in self.merge_history:
            new_token = record['new_token']
            pair = record['pair']
            print(f"   [{new_token:3d}] = {self.vocab[pair[0]]!r}+{self.vocab[pair[1]]!r}")

    def encode_with_trained_bpe(self, text: str) -> List[int]:
        """使用训练好的BPE编码新文本"""
        print(f"\n🔍 使用训练好的BPE编码新文本: {text!r}")

        # 1. 转换为字节序列
        indices = list(map(int, text.encode("utf-8")))
        self.print_tokens(indices, "原始字节序列")

        # 2. 应用所有合并规则
        for step, record in enumerate(self.merge_history):
            pair = record['pair']
            new_token = record['new_token']
            old_length = len(indices)

            indices = self.merge_pair(indices, pair, new_token)

            if len(indices) < old_length:  # 只有实际合并时才显示
                print(f"   步骤{step+1}: 应用合并 [{pair[0]}]+[{pair[1]}] → [{new_token}] (长度: {old_length}→{len(indices)})")

        self.print_tokens(indices, "最终BPE编码")
        return indices

    def decode_with_trained_bpe(self, indices: List[int]) -> str:
        """使用训练好的BPE解码"""
        print(f"\n🔄 解码token序列: {indices}")

        # 将token索引转换为字节
        bytes_list = [self.vocab[idx] for idx in indices]
        print(f"   字节序列: {b''.join(bytes_list)!r}")

        # 解码为字符串
        result = b"".join(bytes_list).decode("utf-8")
        print(f"   解码结果: {result!r}")

        return result


def demo_simple_example():
    """简单示例演示"""
    print("🎯 简单示例: 'aaabdaaabac'")

    text = "aaabdaaabac"
    visualizer = BPEVisualizer(text, num_merges=5, delay=0)
    visualizer.run_visualization()

    # 测试编码新文本
    visualizer.encode_with_trained_bpe("aaaa")
    visualizer.decode_with_trained_bpe([256, 256, 100, 256, 256, 99])


def demo_chinese_example():
    """中文示例演示"""
    print("🎯 中文示例: '你好世界，你好Python'")

    text = "你好世界，你好Python"
    visualizer = BPEVisualizer(text, num_merges=8, delay=0)
    visualizer.run_visualization()

    # 测试编码新文本
    visualizer.encode_with_trained_bpe("你好")


def demo_english_example():
    """英文示例演示"""
    print("🎯 英文示例: 'hello world hello'")

    text = "hello world hello"
    visualizer = BPEVisualizer(text, num_merges=10, delay=0)
    visualizer.run_visualization()


def interactive_mode():
    """交互模式"""
    print("🎮 交互式BPE演示")
    print("输入文本和合并次数，观察BPE算法的工作过程")

    while True:
        text = input("\n📝 请输入文本 (或 'quit' 退出): ").strip()
        if text.lower() == 'quit':
            break

        try:
            num_merges = int(input("🎯 合并次数 (默认10): ").strip() or "10")
        except ValueError:
            num_merges = 10

        try:
            delay = float(input("⏱️ 演示延迟秒数 (默认0.5): ").strip() or "0.5")
        except ValueError:
            delay = 0.5

        visualizer = BPEVisualizer(text, num_merges, delay)
        visualizer.run_visualization()

        # 测试编码
        test_text = input(f"\n🔍 测试编码文本 (默认 '{text[:5]}'): ").strip() or text[:5]
        visualizer.encode_with_trained_bpe(test_text)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="BPE算法可视化演示工具")
    parser.add_argument("--text", type=str, help="要处理的文本")
    parser.add_argument("--merges", type=int, default=10, help="合并次数")
    parser.add_argument("--delay", type=float, default=0.5, help="演示延迟")
    parser.add_argument("--demo", choices=["simple", "chinese", "english"], help="预设演示")
    parser.add_argument("--interactive", action="store_true", help="交互模式")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.demo == "simple":
        demo_simple_example()
    elif args.demo == "chinese":
        demo_chinese_example()
    elif args.demo == "english":
        demo_english_example()
    elif args.text:
        visualizer = BPEVisualizer(args.text, args.merges, args.delay)
        visualizer.run_visualization()
    else:
        print("🎯 BPE可视化演示工具")
        print("\n使用方法:")
        print("1. 预设演示: python bpe_visualizer.py --demo simple")
        print("2. 交互模式: python bpe_visualizer.py --interactive")
        print("3. 自定义文本: python bpe_visualizer.py --text 'your text' --merges 5")
        print("\n🚀 启动简单演示...")
        demo_simple_example()


if __name__ == "__main__":
    main()