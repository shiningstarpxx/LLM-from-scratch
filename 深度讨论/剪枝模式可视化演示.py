#!/usr/bin/env python3
"""
剪枝模式可视化演示
直观展示非结构化剪枝和结构化剪枝的区别
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 配置中文字体，避免中文显示警告
def setup_chinese_font():
    """设置 matplotlib 中文字体"""
    # macOS 系统字体列表（按优先级）
    mac_fonts = ['PingFang SC', 'STHeiti', 'Arial Unicode MS', 'Heiti TC', 'SimHei']
    
    # 尝试找到可用的中文字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_font = None
    
    for font_name in mac_fonts:
        if font_name in available_fonts:
            chinese_font = font_name
            break
    
    if chinese_font:
        plt.rcParams['font.sans-serif'] = [chinese_font]
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print(f"✓ 已设置中文字体: {chinese_font}")
    else:
        # 如果找不到系统字体，尝试使用 matplotlib 的默认设置
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        print("⚠ 未找到中文字体，可能无法正确显示中文")

# 初始化字体设置
setup_chinese_font()

def demonstrate_pruning_patterns():
    """演示不同剪枝模式的具体效果"""

    print("🎯 剪枝模式对比演示")
    print("=" * 50)

    # 创建示例权重矩阵 (4x4)
    torch.manual_seed(42)
    original_weight = torch.randn(4, 4)

    print("📊 原始权重矩阵:")
    print(original_weight)
    print(f"权重范围: [{original_weight.min():.3f}, {original_weight.max():.3f}]")

    # === 1. 非结构化剪枝演示 ===
    print("\n" + "="*50)
    print("🔬 非结构化剪枝 (L1 Unstructured)")
    print("="*50)

    # 创建线性层并设置权重
    linear_unstructured = nn.Linear(4, 4, bias=False)
    linear_unstructured.weight.data = original_weight.clone()

    # 执行非结构化剪枝
    prune.l1_unstructured(linear_unstructured, name='weight', amount=0.25)

    print("剪枝后的权重矩阵:")
    print(linear_unstructured.weight)
    print("\n剪枝掩码 (1=保留, 0=剪枝):")
    print(linear_unstructured.weight_mask)

    # 统计被剪枝的位置
    pruned_positions = torch.where(linear_unstructured.weight_mask == 0)
    print(f"\n被剪枝的位置: {list(zip(pruned_positions[0].tolist(), pruned_positions[1].tolist()))}")

    # 计算实际稀疏度
    actual_sparsity = 1.0 - torch.sum(linear_unstructured.weight_mask).item() / linear_unstructured.weight_mask.numel()
    print(f"实际稀疏度: {actual_sparsity:.2%}")

    # === 2. 结构化剪枝演示 ===
    print("\n" + "="*50)
    print("🏗️ 结构化剪枝 (按行剪枝)")
    print("="*50)

    # 创建新的线性层
    linear_structured = nn.Linear(4, 4, bias=False)
    linear_structured.weight.data = original_weight.clone()

    # 执行结构化剪枝 (按行，dim=0)
    prune.ln_structured(linear_structured, name='weight', amount=0.25, n=2, dim=0)

    print("剪枝后的权重矩阵:")
    print(linear_structured.weight)
    print("\n剪枝掩码:")
    print(linear_structured.weight_mask)

    # 分析被剪枝的行
    fully_pruned_rows = torch.all(linear_structured.weight_mask == 0, dim=1)
    partially_pruned_rows = torch.any(linear_structured.weight_mask == 0, dim=1) & ~fully_pruned_rows
    print(f"\n完全剪枝的行: {torch.where(fully_pruned_rows)[0].tolist()}")
    print(f"部分剪枝的行: {torch.where(partially_pruned_rows)[0].tolist()}")

    # === 3. 对比分析 ===
    print("\n" + "="*50)
    print("📊 两种模式对比分析")
    print("="*50)

    print("非结构化剪枝特点:")
    print("✅ 零散置零单个权重元素")
    print("✅ 保持矩阵结构完整")
    print("✅ 精度损失相对较小")
    print("❌ 硬件难以直接加速")
    print("❌ 内存节省有限")

    print("\n结构化剪枝特点:")
    print("✅ 整行/整列置零")
    print("✅ 产生真正的稀疏结构")
    print("✅ 硬件友好，易于加速")
    print("✅ 显著减少内存占用")
    print("❌ 精度损失相对较大")

    return {
        'original': original_weight,
        'unstructured': linear_unstructured.weight,
        'structured': linear_structured.weight
    }

def visualize_pruning_effects():
    """可视化剪枝效果"""

    # 创建更大的矩阵用于可视化
    torch.manual_seed(123)
    weight_matrix = torch.randn(16, 16)

    # 非结构化剪枝
    weight_unstructured = weight_matrix.clone()
    mask_unstructured = torch.rand(weight_matrix.shape) > 0.3
    weight_unstructured = weight_unstructured * mask_unstructured.float()

    # 结构化剪枝 (按行)
    weight_structured = weight_matrix.clone()
    n_rows = int(16 * 0.3)  # 剪枝30%的行
    row_indices = torch.randperm(16)[:n_rows]
    weight_structured[row_indices, :] = 0

    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 原始矩阵
    im1 = axes[0, 0].imshow(weight_matrix, cmap='RdBu', vmin=-3, vmax=3)
    axes[0, 0].set_title('原始权重矩阵', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('输出维度')
    axes[0, 0].set_ylabel('输入维度')
    plt.colorbar(im1, ax=axes[0, 0])

    # 非结构化剪枝
    im2 = axes[0, 1].imshow(weight_unstructured, cmap='RdBu', vmin=-3, vmax=3)
    axes[0, 1].set_title('非结构化剪枝 (30%稀疏)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('输出维度')
    axes[0, 1].set_ylabel('输入维度')
    plt.colorbar(im2, ax=axes[0, 1])

    # 结构化剪枝
    im3 = axes[1, 0].imshow(weight_structured, cmap='RdBu', vmin=-3, vmax=3)
    axes[1, 0].set_title('结构化剪枝 (按行30%)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('输出维度')
    axes[1, 0].set_ylabel('输入维度')
    plt.colorbar(im3, ax=axes[1, 0])

    # 剪枝模式对比
    comparison_data = np.array([
        ['非结构化', '零散置零', '精度高', '硬件不友好'],
        ['结构化', '整行置零', '精度中', '硬件友好']
    ])

    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=comparison_data,
                             colLabels=['模式', '置零方式', '精度影响', '硬件友好度'],
                             cellLoc='center',
                             loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 1].set_title('剪枝模式对比', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # 保存图片
    save_path = '/Users/peixingxin/code/spring2025-lectures/深度讨论/剪枝模式可视化.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 可视化图片已保存: {save_path}")

    plt.show()

def pruning_performance_impact():
    """分析剪枝对性能的实际影响"""

    print("\n" + "="*60)
    print("⚡ 剪枝性能影响分析")
    print("="*60)

    # 创建不同规模的权重矩阵
    sizes = [(100, 100), (500, 500), (1000, 1000)]
    pruning_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

    print(f"{'矩阵规模':<12} {'剪枝比例':<10} {'非结构化FLOP':<15} {'结构化FLOP':<15} {'理论加速比':<12}")
    print("-" * 70)

    for size in sizes:
        rows, cols = size

        for ratio in pruning_ratios:
            # 原始FLOP (矩阵乘法: rows * cols * 2)
            original_flops = rows * cols * 2

            # 非结构化剪枝FLOP (大部分硬件仍需计算零元素)
            unstructured_flops = original_flops * 0.95  # 假设5%的优化

            # 结构化剪枝FLOP (真正减少计算)
            structured_flops = original_flops * (1 - ratio)

            # 理论加速比
            speedup = original_flops / structured_flops

            print(f"{rows}x{cols}{'':<6} {ratio:<10.1f} {unstructured_flops:<15.0f} {structured_flops:<15.0f} {speedup:<12.2f}x")

def main():
    """主函数"""
    print("🔪 剪枝技术深度解析演示")
    print("=" * 60)

    # 1. 演示剪枝模式
    results = demonstrate_pruning_patterns()

    # 2. 可视化剪枝效果
    try:
        visualize_pruning_effects()
    except Exception as e:
        print(f"可视化失败: {e}")
        print("可能需要安装matplotlib: pip install matplotlib")

    # 3. 性能影响分析
    pruning_performance_impact()

    print("\n" + "="*60)
    print("🎯 核心结论")
    print("="*60)
    print("1. 非结构化剪枝: 零散置零，不是整行/整列")
    print("2. 结构化剪枝: 整体删除，才是真正的行/列置零")
    print("3. 硬件友好性: 结构化 >> 非结构化")
    print("4. 精度保持: 非结构化 > 结构化")
    print("5. 实际应用: 需要根据具体场景权衡选择")

if __name__ == "__main__":
    main()