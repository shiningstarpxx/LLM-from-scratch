#!/usr/bin/env python3
"""
PyTorch Building Blocks & Resource Accounting - 可视化实践演示
这个脚本将Lecture 02的核心概念转化为可执行的、可视化的演示

作者: CS336 Deep Learning Systems
日期: 2025-11-07
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import time
import functools
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 配置中文字体，避免中文显示警告
import matplotlib.font_manager as fm
import os

def setup_chinese_font():
    """设置 matplotlib 中文字体 - 增强版"""
    # 优先尝试直接使用字体文件路径（最可靠的方法）
    system_font_paths = [
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
        '/System/Library/Fonts/STHeiti Medium.ttc',
        '/Library/Fonts/Arial Unicode.ttf',
    ]
    
    font_path = None
    chinese_font_name = None
    
    # 查找可用的字体文件
    for path in system_font_paths:
        if os.path.exists(path):
            font_path = path
            try:
                # 获取字体名称
                font_prop = fm.FontProperties(fname=path)
                chinese_font_name = font_prop.get_name()
                break
            except:
                continue
    
    # 如果找到字体文件，直接使用文件路径
    if font_path:
        try:
            # 将字体添加到matplotlib的字体列表
            try:
                fm.fontManager.addfont(font_path)
            except (AttributeError, ValueError):
                # 旧版本或字体已存在
                pass
            
            # 使用字体文件路径设置全局字体
            plt.rcParams['font.sans-serif'] = [chinese_font_name] + ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✓ 已设置中文字体: {chinese_font_name} (来自 {font_path})")
            return font_path  # 返回字体路径，用于后续显式指定
        except Exception as e:
            print(f"⚠ 设置字体文件失败: {e}")
    
    # 备用方案：通过字体名称查找
    mac_fonts = ['PingFang SC', 'STHeiti', 'Arial Unicode MS', 'Heiti TC', 'SimHei', 'Hiragino Sans GB']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font_name in mac_fonts:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name] + ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✓ 已设置中文字体: {font_name}")
            return font_name
    
    # 最后的备用方案
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("⚠ 未找到中文字体，使用备用字体设置")
    return None

# 初始化字体设置（在导入其他模块之前）
chinese_font_name = setup_chinese_font()

# 创建字体属性对象，用于显式指定字体
def get_chinese_font_prop(size=12):
    """获取中文字体属性对象"""
    if chinese_font_name:
        # 如果是字体文件路径，直接使用
        if isinstance(chinese_font_name, str) and os.path.exists(chinese_font_name):
            return fm.FontProperties(fname=chinese_font_name, size=size)
        else:
            # 如果是字体名称，使用family参数
            return fm.FontProperties(family=chinese_font_name, size=size)
    else:
        return fm.FontProperties(family='sans-serif', size=size)

# 设置绘图风格
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

print("🚀 PyTorch Building Blocks & Resource Accounting - 可视化实践演示")
print("=" * 80)

class FLOPCalculator:
    """PyTorch模型FLOP计算器 - 增强版"""

    def __init__(self):
        self.flops = 0
        self.layer_flops = {}
        self.hooks = []

    def _conv_flop(self, input_shape: tuple, output_shape: tuple,
                   kernel_shape: tuple, groups: int = 1) -> int:
        """计算卷积操作FLOP"""
        batch_size = input_shape[0]
        output_dims = output_shape[2:]
        kernel_dims = kernel_shape[2:]
        in_channels = input_shape[1]
        out_channels = output_shape[1]

        filters_per_channel = out_channels // groups
        conv_per_position_flops = functools.reduce(
            lambda a, b: a * b, kernel_dims) * in_channels // groups
        active_elements_count = batch_size * functools.reduce(
            lambda a, b: a * b, output_dims)

        overall_conv_flops = conv_per_position_flops * active_elements_count * filters_per_channel
        bias_flops = out_channels * active_elements_count

        return overall_conv_flops + bias_flops

    def _linear_flop(self, input_shape: tuple, weight_shape: tuple,
                     has_bias: bool = True) -> int:
        """计算线性层FLOP"""
        batch_size = input_shape[0]
        in_features = input_shape[1]
        out_features = weight_shape[0]

        mul_flops = batch_size * in_features * out_features
        add_flops = batch_size * out_features if has_bias else 0

        return mul_flops + add_flops

    def _create_hook(self, layer_name: str, layer_type: str):
        """创建FLOP计算钩子"""
        def hook_fn(module, input, output):
            if layer_type == 'conv2d':
                if hasattr(module, 'weight'):
                    kernel_shape = module.weight.shape
                    input_shape = input[0].shape
                    output_shape = output.shape
                    flops = self._conv_flop(input_shape, output_shape,
                                          kernel_shape, module.groups)
                    self.flops += flops
                    self.layer_flops[layer_name] = flops

            elif layer_type == 'linear':
                if hasattr(module, 'weight'):
                    input_shape = input[0].shape
                    weight_shape = module.weight.shape
                    has_bias = hasattr(module, 'bias') and module.bias is not None
                    flops = self._linear_flop(input_shape, weight_shape, has_bias)
                    self.flops += flops
                    self.layer_flops[layer_name] = flops

        return hook_fn

    def analyze_model(self, model: nn.Module, input_shape: tuple) -> Dict[str, Any]:
        """分析模型FLOP"""
        self.flops = 0
        self.layer_flops = {}
        self.hooks = []

        # 注册钩子
        for name, module in model.named_modules():
            if name:  # 跳过根模块
                if isinstance(module, nn.Conv2d):
                    hook = module.register_forward_hook(
                        self._create_hook(name, 'conv2d'))
                    self.hooks.append(hook)

                elif isinstance(module, nn.Linear):
                    hook = module.register_forward_hook(
                        self._create_hook(name, 'linear'))
                    self.hooks.append(hook)

        # 运行前向传播
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_shape, device=device)

        with torch.no_grad():
            _ = model(dummy_input)

        # 清理钩子
        for hook in self.hooks:
            hook.remove()

        return {
            'total_flops': self.flops,
            'layer_flops': self.layer_flops,
            'flops_readable': self._format_flops(self.flops),
            'parameters': sum(p.numel() for p in model.parameters()),
            'parameters_readable': self._format_count(
                sum(p.numel() for p in model.parameters()))
        }

    def _format_flops(self, flops: int) -> str:
        """格式化FLOP显示"""
        if flops >= 1e15:
            return f"{flops/1e15:.2f} PFLOP"
        elif flops >= 1e12:
            return f"{flops/1e12:.2f} TFLOP"
        elif flops >= 1e9:
            return f"{flops/1e9:.2f} GFLOP"
        elif flops >= 1e6:
            return f"{flops/1e6:.2f} MFLOP"
        elif flops >= 1e3:
            return f"{flops/1e3:.2f} KFLOP"
        else:
            return f"{flops} FLOP"

    def _format_count(self, count: int) -> str:
        """格式化数量显示"""
        if count >= 1e9:
            return f"{count/1e9:.2f} B"
        elif count >= 1e6:
            return f"{count/1e6:.2f} M"
        elif count >= 1e3:
            return f"{count/1e3:.2f} K"
        else:
            return str(count)


class MemoryProfiler:
    """GPU/CPU内存使用分析器 - 增强版"""

    def __init__(self):
        self.snapshots = []
        self.peak_memory = {'cpu': 0, 'gpu': 0}

    def snapshot(self, label: str = ""):
        """拍摄内存快照"""
        # CPU内存
        cpu_memory = psutil.virtual_memory()
        cpu_used = cpu_memory.used / 1024**3  # GB

        # GPU内存
        gpu_memory = {'allocated': 0, 'cached': 0, 'max_allocated': 0}
        if torch.cuda.is_available():
            gpu_memory['allocated'] = torch.cuda.memory_allocated() / 1024**3
            gpu_memory['cached'] = torch.cuda.memory_reserved() / 1024**3
            gpu_memory['max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3

        snapshot = {
            'timestamp': time.time(),
            'label': label,
            'cpu_used_gb': cpu_used,
            'cpu_percent': cpu_memory.percent,
            **gpu_memory
        }

        self.snapshots.append(snapshot)

        # 更新峰值
        self.peak_memory['cpu'] = max(self.peak_memory['cpu'], cpu_used)
        if torch.cuda.is_available():
            self.peak_memory['gpu'] = max(self.peak_memory['gpu'], gpu_memory['allocated'])

        return snapshot

    def reset(self):
        """重置分析器"""
        self.snapshots = []
        self.peak_memory = {'cpu': 0, 'gpu': 0}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def get_memory_analysis(self, model: nn.Module, param_memory: float,
                          activation_memory: float) -> Dict[str, float]:
        """获取内存分析结果"""
        grad_memory = param_memory  # 梯度内存与参数相同
        optimizer_memory = param_memory * 2  # Adam优化器状态

        return {
            'parameter_memory_gb': param_memory,
            'gradient_memory_gb': grad_memory,
            'optimizer_memory_gb': optimizer_memory,
            'activation_memory_gb': activation_memory,
            'total_training_memory_gb': param_memory + grad_memory + optimizer_memory + activation_memory,
            'total_inference_memory_gb': param_memory + activation_memory
        }


def visualize_flop_analysis():
    """可视化FLOP分析"""
    print("\n🧮 FLOP分析可视化演示")
    print("=" * 50)

    # 创建不同规模的模型
    models = {
        'Small CNN': nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(32, 10)
        ),
        'Medium CNN': nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(128, 10)
        ),
        'Large CNN': nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(512, 10)
        )
    }

    input_shape = (1, 3, 32, 32)
    results = {}

    # 分析每个模型
    for name, model in models.items():
        calculator = FLOPCalculator()
        result = calculator.analyze_model(model, input_shape)
        results[name] = result
        print(f"{name}: {result['flops_readable']}, {result['parameters_readable']} 参数")

    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('FLOP分析可视化', fontsize=16, fontweight='bold', 
                 fontproperties=get_chinese_font_prop(16))

    # 1. FLOP对比柱状图
    model_names = list(results.keys())
    flop_values = [results[name]['total_flops'] / 1e6 for name in model_names]  # MFLOP

    bars = axes[0, 0].bar(model_names, flop_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 0].set_title('模型FLOP对比', fontsize=14, fontweight='bold',
                         fontproperties=get_chinese_font_prop(14))
    axes[0, 0].set_ylabel('FLOP (MFLOP)')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 添加数值标签
    for bar, value in zip(bars, flop_values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(flop_values)*0.01,
                       f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

    # 2. 参数数量对比
    param_values = [results[name]['parameters'] for name in model_names]

    bars = axes[0, 1].bar(model_names, param_values, color=['#95E77E', '#FFD93D', '#FF6BCB'])
    axes[0, 1].set_title('模型参数数量对比', fontsize=14, fontweight='bold',
                         fontproperties=get_chinese_font_prop(14))
    axes[0, 1].set_ylabel('参数数量', fontproperties=get_chinese_font_prop())
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 添加数值标签
    for bar, value in zip(bars, param_values):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(param_values)*0.01,
                       f'{value:,}', ha='center', va='bottom', fontweight='bold')

    # 3. FLOP vs 参数散点图
    axes[1, 0].scatter(param_values, flop_values,
                       s=200, c=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                       alpha=0.7, edgecolors='black')

    for i, name in enumerate(model_names):
        axes[1, 0].annotate(name, (param_values[i], flop_values[i]),
                           xytext=(5, 5), textcoords='offset points', fontweight='bold')

    axes[1, 0].set_xlabel('参数数量', fontproperties=get_chinese_font_prop())
    axes[1, 0].set_ylabel('FLOP (MFLOP)')
    axes[1, 0].set_title('FLOP vs 参数数量关系', fontsize=14, fontweight='bold',
                         fontproperties=get_chinese_font_prop(14))

    # 4. 效率指标（FLOP/参数）
    efficiency = [flop/param for flop, param in zip(flop_values, param_values)]

    bars = axes[1, 1].bar(model_names, efficiency, color=['#C9B6E5', '#FFB6B9', '#B6E5D8'])
    axes[1, 1].set_title('计算效率 (FLOP/参数)', fontsize=14, fontweight='bold',
                         fontproperties=get_chinese_font_prop(14))
    axes[1, 1].set_ylabel('FLOP per 参数', fontproperties=get_chinese_font_prop())
    axes[1, 1].tick_params(axis='x', rotation=45)

    # 添加数值标签
    for bar, value in zip(bars, efficiency):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(efficiency)*0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('/Users/peixingxin/code/spring2025-lectures/深度讨论/FLOP分析可视化.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def visualize_memory_analysis():
    """可视化内存分析"""
    print("\n💾 内存分析可视化演示")
    print("=" * 50)

    # 创建不同规模的模型进行内存分析
    model_configs = [
        {'name': 'Small', 'layers': [256, 128, 64], 'batch_size': 32},
        {'name': 'Medium', 'layers': [512, 256, 128], 'batch_size': 16},
        {'name': 'Large', 'layers': [1024, 512, 256], 'batch_size': 8}
    ]

    results = {}

    for config in model_configs:
        # 创建模型
        layers = []
        input_dim = config['layers'][0]
        for hidden_dim in config['layers'][1:]:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 10))

        model = nn.Sequential(*layers)

        # 计算参数内存
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3

        # 估算激活内存（简化计算）
        activation_memory = param_memory * 0.5  # 粗略估算

        # 获取内存分析
        profiler = MemoryProfiler()
        analysis = profiler.get_memory_analysis(model, param_memory, activation_memory)

        results[config['name']] = {
            'param_memory': param_memory,
            'activation_memory': activation_memory,
            'training_memory': analysis['total_training_memory_gb'],
            'inference_memory': analysis['total_inference_memory_gb']
        }

        print(f"{config['name']} 模型:")
        print(f"  参数内存: {param_memory:.3f} GB")
        print(f"  激活内存: {activation_memory:.3f} GB")
        print(f"  训练内存: {analysis['total_training_memory_gb']:.3f} GB")
        print(f"  推理内存: {analysis['total_inference_memory_gb']:.3f} GB")

    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('内存分析可视化', fontsize=16, fontweight='bold',
                 fontproperties=get_chinese_font_prop(16))

    model_names = list(results.keys())

    # 1. 内存组成堆叠图（训练）
    param_mem = [results[name]['param_memory'] for name in model_names]
    act_mem = [results[name]['activation_memory'] for name in model_names]
    grad_mem = param_mem.copy()  # 梯度内存=参数内存
    opt_mem = [p * 2 for p in param_mem]  # 优化器内存=2*参数内存

    x = np.arange(len(model_names))
    width = 0.6
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD93D']
    labels = ['参数', '激活', '梯度', '优化器']
    
    # 创建堆叠柱状图
    axes[0, 0].bar(x, param_mem, width, label=labels[0], color=colors[0], alpha=0.8)
    bottom = param_mem
    axes[0, 0].bar(x, act_mem, width, bottom=bottom, label=labels[1], color=colors[1], alpha=0.8)
    bottom = [b + a for b, a in zip(bottom, act_mem)]
    axes[0, 0].bar(x, grad_mem, width, bottom=bottom, label=labels[2], color=colors[2], alpha=0.8)
    bottom = [b + g for b, g in zip(bottom, grad_mem)]
    axes[0, 0].bar(x, opt_mem, width, bottom=bottom, label=labels[3], color=colors[3], alpha=0.8)
    
    axes[0, 0].set_title('训练内存组成', fontsize=14, fontweight='bold',
                         fontproperties=get_chinese_font_prop(14))
    axes[0, 0].set_ylabel('内存 (GB)', fontproperties=get_chinese_font_prop())
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names)
    axes[0, 0].legend(prop=get_chinese_font_prop(10))

    # 2. 训练 vs 推理内存对比
    training_mem = [results[name]['training_memory'] for name in model_names]
    inference_mem = [results[name]['inference_memory'] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    axes[0, 1].bar(x - width/2, training_mem, width, label='训练', color='#FF6B6B', alpha=0.8)
    axes[0, 1].bar(x + width/2, inference_mem, width, label='推理', color='#4ECDC4', alpha=0.8)

    axes[0, 1].set_title('训练 vs 推理内存对比', fontsize=14, fontweight='bold',
                        fontproperties=get_chinese_font_prop(14))
    axes[0, 1].set_ylabel('内存 (GB)', fontproperties=get_chinese_font_prop())
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(model_names)
    axes[0, 1].legend(prop=get_chinese_font_prop(10))

    # 3. 内存增长趋势
    model_sizes = [sum(p.numel() for p in nn.Sequential(
        *[nn.Linear(config['layers'][i] if i == 0 else config['layers'][i-1],
                    config['layers'][i]) for i in range(len(config['layers']))]
    ).parameters()) for config in model_configs]

    axes[1, 0].plot(model_sizes, training_mem, 'o-', linewidth=3, markersize=8,
                    color='#FF6B6B', label='训练内存')
    axes[1, 0].plot(model_sizes, inference_mem, 's-', linewidth=3, markersize=8,
                    color='#4ECDC4', label='推理内存')

    axes[1, 0].set_xlabel('模型参数数量', fontproperties=get_chinese_font_prop())
    axes[1, 0].set_ylabel('内存 (GB)', fontproperties=get_chinese_font_prop())
    axes[1, 0].set_title('内存增长趋势', fontsize=14, fontweight='bold',
                         fontproperties=get_chinese_font_prop(14))
    axes[1, 0].legend(prop=get_chinese_font_prop(10))
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 内存效率（推理内存/参数）
    memory_efficiency = [results[name]['inference_memory'] / results[name]['param_memory']
                        for name in model_names]

    bars = axes[1, 1].bar(model_names, memory_efficiency,
                          color=['#95E77E', '#FFD93D', '#FF6BCB'])
    axes[1, 1].set_title('内存效率 (推理内存/参数内存)', fontsize=14, fontweight='bold',
                         fontproperties=get_chinese_font_prop(14))
    axes[1, 1].set_ylabel('效率比率', fontproperties=get_chinese_font_prop())
    axes[1, 1].tick_params(axis='x', rotation=45)

    # 添加数值标签
    for bar, value in zip(bars, memory_efficiency):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(memory_efficiency)*0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('/Users/peixingxin/code/spring2025-lectures/深度讨论/内存分析可视化.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def visualize_training_performance():
    """可视化训练性能"""
    print("\n🔄 训练性能可视化演示")
    print("=" * 50)

    # 模拟训练数据
    epochs = 50

    # 模拟不同的训练场景
    scenarios = {
        '基线训练': {
            'train_loss': np.linspace(2.0, 0.5, epochs) + np.random.normal(0, 0.05, epochs),
            'val_loss': np.linspace(2.1, 0.6, epochs) + np.random.normal(0, 0.08, epochs),
            'throughput': np.random.uniform(50, 60, epochs),
            'color': '#FF6B6B'
        },
        '混合精度训练': {
            'train_loss': np.linspace(2.0, 0.45, epochs) + np.random.normal(0, 0.04, epochs),
            'val_loss': np.linspace(2.1, 0.55, epochs) + np.random.normal(0, 0.06, epochs),
            'throughput': np.random.uniform(80, 95, epochs),
            'color': '#4ECDC4'
        },
        '梯度累积': {
            'train_loss': np.linspace(2.0, 0.48, epochs) + np.random.normal(0, 0.045, epochs),
            'val_loss': np.linspace(2.1, 0.58, epochs) + np.random.normal(0, 0.07, epochs),
            'throughput': np.random.uniform(45, 55, epochs),
            'color': '#45B7D1'
        }
    }

    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('训练性能可视化对比', fontsize=16, fontweight='bold')

    epoch_range = range(1, epochs + 1)

    # 1. 训练损失曲线
    for name, data in scenarios.items():
        axes[0, 0].plot(epoch_range, data['train_loss'], linewidth=2, label=name,
                       color=data['color'], alpha=0.8)

    axes[0, 0].set_title('训练损失变化', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 验证损失曲线
    for name, data in scenarios.items():
        axes[0, 1].plot(epoch_range, data['val_loss'], linewidth=2, label=name,
                       color=data['color'], alpha=0.8)

    axes[0, 1].set_title('验证损失变化', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('损失')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 吞吐量对比
    for name, data in scenarios.items():
        axes[1, 0].plot(epoch_range, data['throughput'], linewidth=2, label=name,
                       color=data['color'], alpha=0.8)

    axes[1, 0].set_title('训练吞吐量', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('样本/秒')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 最终性能对比雷达图
    categories = ['最终精度', '训练速度', '内存效率', '稳定性', '收敛速度']

    # 计算各项指标（归一化到0-1）
    final_scores = {}
    for name, data in scenarios.items():
        final_loss = data['val_loss'][-1]
        avg_throughput = np.mean(data['throughput'])
        loss_std = np.std(data['val_loss'][-10:])  # 最后10个epoch的标准差

        scores = [
            1.0 - (final_loss / 2.1),  # 精度 (损失越低越好)
            avg_throughput / 100,       # 速度 (归一化到100)
            0.8 if name == '混合精度训练' else 0.6,  # 内存效率
            1.0 - (loss_std / 0.1),    # 稳定性
            1.0 - (data['val_loss'][10] / data['val_loss'][0])  # 收敛速度
        ]
        final_scores[name] = scores

    # 绘制雷达图
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    ax_radar = plt.subplot(2, 2, 4, projection='polar')

    for name, scores in final_scores.items():
        scores += scores[:1]  # 闭合图形
        ax_radar.plot(angles, scores, 'o-', linewidth=2, label=name,
                     color=scenarios[name]['color'], markersize=6)
        ax_radar.fill(angles, scores, alpha=0.25, color=scenarios[name]['color'])

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('综合性能对比', fontsize=14, fontweight='bold', pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig('/Users/peixingxin/code/spring2025-lectures/深度讨论/训练性能可视化.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def demonstrate_mixed_precision_benefits():
    """演示混合精度训练的实际好处"""
    print("\n⚡ 混合精度训练效益演示")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 创建测试模型
    model = nn.Sequential(
        nn.Linear(2048, 4096),
        nn.ReLU(),
        nn.Linear(4096, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1000)
    ).to(device)

    batch_size = 64
    input_data = torch.randn(batch_size, 2048, device=device)
    target = torch.randint(0, 1000, (batch_size,), device=device)

    results = {}

    # FP32 基准测试
    print("\n🔢 FP32 基准测试:")
    model_fp32 = model.float()
    optimizer_fp32 = optim.Adam(model_fp32.parameters())

    # 预热
    for _ in range(5):
        optimizer_fp32.zero_grad()
        output = model_fp32(input_data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer_fp32.step()

    # 正式测试
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    total_loss = 0

    for _ in range(50):
        optimizer_fp32.zero_grad()
        output = model_fp32(input_data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer_fp32.step()
        total_loss += loss.item()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    fp32_time = time.time() - start_time
    fp32_memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

    results['FP32'] = {
        'time': fp32_time,
        'memory': fp32_memory,
        'loss': total_loss / 50
    }

    print(f"  平均时间: {fp32_time/50*1000:.2f}ms/iter")
    print(f"  峰值内存: {fp32_memory:.2f}GB")
    print(f"  平均损失: {results['FP32']['loss']:.4f}")

    if torch.cuda.is_available():
        # 混合精度测试
        print("\n🔀 混合精度训练测试:")
        model_amp = model
        optimizer_amp = optim.Adam(model_amp.parameters())
        scaler = torch.cuda.amp.GradScaler()

        torch.cuda.reset_peak_memory_stats()

        # 预热
        for _ in range(5):
            optimizer_amp.zero_grad()
            with torch.cuda.amp.autocast():
                output = model_amp(input_data)
                loss = nn.CrossEntropyLoss()(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer_amp)
            scaler.update()

        torch.cuda.synchronize()
        start_time = time.time()
        total_loss = 0

        for _ in range(50):
            optimizer_amp.zero_grad()
            with torch.cuda.amp.autocast():
                output = model_amp(input_data)
                loss = nn.CrossEntropyLoss()(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer_amp)
            scaler.update()
            total_loss += loss.item()

        torch.cuda.synchronize()
        amp_time = time.time() - start_time
        amp_memory = torch.cuda.max_memory_allocated() / 1024**3

        results['AMP'] = {
            'time': amp_time,
            'memory': amp_memory,
            'loss': total_loss / 50
        }

        print(f"  平均时间: {amp_time/50*1000:.2f}ms/iter")
        print(f"  峰值内存: {amp_memory:.2f}GB")
        print(f"  平均损失: {results['AMP']['loss']:.4f}")

        # 性能对比
        speedup = fp32_time / amp_time
        memory_saved = (fp32_memory - amp_memory) / fp32_memory * 100

        print(f"\n📊 性能提升:")
        print(f"  加速比: {speedup:.2f}x")
        print(f"  内存节省: {memory_saved:.1f}%")

        # 可视化对比
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('混合精度训练效益对比', fontsize=16, fontweight='bold')

        # 时间对比
        times = [fp32_time, amp_time]
        labels = ['FP32', '混合精度']
        colors = ['#FF6B6B', '#4ECDC4']

        bars = axes[0].bar(labels, times, color=colors, alpha=0.8)
        axes[0].set_title('训练时间对比 (50 iterations)', fontweight='bold')
        axes[0].set_ylabel('时间 (秒)')

        for bar, value in zip(bars, times):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                        f'{value:.3f}s', ha='center', va='bottom', fontweight='bold')

        # 内存对比
        memories = [fp32_memory, amp_memory]

        bars = axes[1].bar(labels, memories, color=colors, alpha=0.8)
        axes[1].set_title('内存使用对比', fontweight='bold')
        axes[1].set_ylabel('内存 (GB)')

        for bar, value in zip(bars, memories):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(memories)*0.01,
                        f'{value:.2f}GB', ha='center', va='bottom', fontweight='bold')

        # 效益指标
        metrics = [f'{speedup:.2f}x', f'{memory_saved:.1f}%']
        metric_labels = ['加速比', '内存节省']

        bars = axes[2].bar(metric_labels, [float(m.rstrip('x%')) for m in metrics],
                          color=['#45B7D1', '#FFD93D'], alpha=0.8)
        axes[2].set_title('性能提升指标', fontweight='bold')
        axes[2].set_ylabel('提升幅度')

        for bar, value, label in zip(bars, [float(m.rstrip('x%')) for m in metrics], metrics):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() +
                        float(metrics[0].rstrip('x%'))*0.01, label,
                        ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('/Users/peixingxin/code/spring2025-lectures/深度讨论/混合精度效益对比.png',
                    dpi=300, bbox_inches='tight')
        plt.show()


def create_comprehensive_dashboard():
    """创建综合性能仪表板"""
    print("\n📊 综合性能仪表板")
    print("=" * 50)

    # 模拟不同规模模型的综合数据
    model_configs = [
        {'name': 'ResNet-18', 'params': '11.7M', 'flops': '1.8 GFLOP', 'memory': '0.8GB', 'accuracy': '69.8%'},
        {'name': 'ResNet-34', 'params': '21.8M', 'flops': '3.7 GFLOP', 'memory': '1.2GB', 'accuracy': '73.3%'},
        {'name': 'ResNet-50', 'params': '25.6M', 'flops': '4.1 GFLOP', 'memory': '1.5GB', 'accuracy': '76.2%'},
        {'name': 'MobileNet-V2', 'params': '3.5M', 'flops': '0.3 GFLOP', 'memory': '0.3GB', 'accuracy': '71.8%'},
        {'name': 'EfficientNet-B0', 'params': '5.3M', 'flops': '0.4 GFLOP', 'memory': '0.4GB', 'accuracy': '77.1%'}
    ]

    # 解析数据
    names = [config['name'] for config in model_configs]
    params = [float(config['params'].rstrip('M')) for config in model_configs]
    flops = [float(config['flops'].split()[0]) for config in model_configs]
    memory = [float(config['memory'].rstrip('GB')) for config in model_configs]
    accuracy = [float(config['accuracy'].rstrip('%')) for config in model_configs]

    # 创建综合仪表板
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    fig.suptitle('深度学习模型综合性能仪表板', fontsize=20, fontweight='bold')

    # 1. 参数 vs FLOP 散点图
    ax1 = fig.add_subplot(gs[0, :2])
    scatter = ax1.scatter(params, flops, s=[m * 500 for m in memory], c=accuracy,
                         cmap='viridis', alpha=0.7, edgecolors='black')

    for i, name in enumerate(names):
        ax1.annotate(name, (params[i], flops[i]), xytext=(5, 5),
                    textcoords='offset points', fontweight='bold')

    ax1.set_xlabel('参数数量 (M)')
    ax1.set_ylabel('FLOP (GFLOP)')
    ax1.set_title('模型复杂度分析 (气泡大小=内存, 颜色=精度)', fontweight='bold')
    plt.colorbar(scatter, ax=ax1, label='精度 (%)')

    # 2. 性能指标雷达图
    ax2 = fig.add_subplot(gs[0, 2:], projection='polar')

    # 归一化指标
    max_params, max_flops, max_memory, max_acc = max(params), max(flops), max(memory), max(accuracy)

    categories = ['参数效率', '计算效率', '内存效率', '精度']

    for i, config in enumerate(model_configs):
        scores = [
            1 - (params[i] / max_params),      # 参数效率 (越小越好)
            1 - (flops[i] / max_flops),        # 计算效率 (越小越好)
            1 - (memory[i] / max_memory),      # 内存效率 (越小越好)
            accuracy[i] / max_acc              # 精度 (越大越好)
        ]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        scores += scores[:1]
        angles += angles[:1]

        ax2.plot(angles, scores, 'o-', linewidth=2, label=config['name'], markersize=6)
        ax2.fill(angles, scores, alpha=0.15)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, 1)
    ax2.set_title('综合性能雷达图', fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # 3. 精度 vs 参数关系
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(params, accuracy, 'o-', linewidth=3, markersize=8, color='#FF6B6B')
    ax3.set_xlabel('参数数量 (M)')
    ax3.set_ylabel('精度 (%)')
    ax3.set_title('精度 vs 参数数量', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. 精度 vs FLOP关系
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(flops, accuracy, 'o-', linewidth=3, markersize=8, color='#4ECDC4')
    ax4.set_xlabel('FLOP (GFLOP)')
    ax4.set_ylabel('精度 (%)')
    ax4.set_title('精度 vs 计算量', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. 内存使用对比
    ax5 = fig.add_subplot(gs[1, 2])
    bars = ax5.bar(names, memory, color=['#95E77E', '#FFD93D', '#FF6BCB', '#C9B6E5', '#FFB6B9'])
    ax5.set_title('内存使用对比', fontweight='bold')
    ax5.set_ylabel('内存 (GB)')
    ax5.tick_params(axis='x', rotation=45)

    for bar, value in zip(bars, memory):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(memory)*0.01,
                f'{value}GB', ha='center', va='bottom', fontweight='bold')

    # 6. 效率排行榜
    ax6 = fig.add_subplot(gs[1, 3])

    # 计算综合效率分数
    efficiency_scores = []
    for i in range(len(names)):
        score = (accuracy[i] / max_acc) / ((params[i] / max_params + flops[i] / max_flops + memory[i] / max_memory) / 3)
        efficiency_scores.append(score)

    sorted_indices = np.argsort(efficiency_scores)[::-1]
    sorted_names = [names[i] for i in sorted_indices]
    sorted_scores = [efficiency_scores[i] for i in sorted_indices]

    bars = ax6.barh(sorted_names, sorted_scores, color='#45B7D1', alpha=0.8)
    ax6.set_title('综合效率排行榜', fontweight='bold')
    ax6.set_xlabel('效率分数')

    # 7. 推荐矩阵 (精度 vs 效率)
    ax7 = fig.add_subplot(gs[2, :2])

    for i, name in enumerate(names):
        color = 'red' if efficiency_scores[i] > np.mean(efficiency_scores) else 'blue'
        size = 100 + accuracy[i] * 10
        ax7.scatter(efficiency_scores[i], accuracy[i], s=size, alpha=0.7,
                   color=color, edgecolors='black')
        ax7.annotate(name, (efficiency_scores[i], accuracy[i]), xytext=(5, 5),
                    textcoords='offset points', fontweight='bold')

    # 添加推荐区域
    ax7.axhline(y=np.mean(accuracy), color='gray', linestyle='--', alpha=0.5, label='平均精度')
    ax7.axvline(x=np.mean(efficiency_scores), color='gray', linestyle='--', alpha=0.5, label='平均效率')

    ax7.set_xlabel('效率分数')
    ax7.set_ylabel('精度 (%)')
    ax7.set_title('模型选择推荐矩阵', fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. 关键洞察文本
    ax8 = fig.add_subplot(gs[2, 2:])
    ax8.axis('off')

    best_accuracy_idx = np.argmax(accuracy)
    best_efficiency_idx = np.argmax(efficiency_scores)
    lowest_memory_idx = np.argmin(memory)

    insights = f"""
    🎯 关键洞察:

    🏆 最高精度: {names[best_accuracy_idx]}
       精度: {accuracy[best_accuracy_idx]}%
       参数: {params[best_accuracy_idx]}M

    ⚡ 最高效率: {names[best_efficiency_idx]}
       效率分数: {efficiency_scores[best_efficiency_idx]:.3f}
       内存: {memory[best_efficiency_idx]}GB

    💾 最低内存: {names[lowest_memory_idx]}
       内存: {memory[lowest_memory_idx]}GB
       精度: {accuracy[lowest_memory_idx]}%

    📊 推荐选择:
    • 追求精度: {names[best_accuracy_idx]}
    • 平衡性能: {names[best_efficiency_idx]}
    • 移动部署: {names[lowest_memory_idx]}
    """

    ax8.text(0.1, 0.5, insights, fontsize=12, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5",
            facecolor="lightgray", alpha=0.8))

    plt.savefig('/Users/peixingxin/code/spring2025-lectures/深度讨论/综合性能仪表板.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主演示函数"""
    print("🚀 开始PyTorch可视化实践演示")
    print("=" * 80)

    try:
        # 1. FLOP分析可视化
        visualize_flop_analysis()

        # 2. 内存分析可视化
        visualize_memory_analysis()

        # 3. 训练性能可视化
        visualize_training_performance()

        # 4. 混合精度效益演示
        demonstrate_mixed_precision_benefits()

        # 5. 综合性能仪表板
        create_comprehensive_dashboard()

        print("\n✅ 所有可视化演示完成！")
        print("\n📁 生成的图片文件:")
        print("  📊 FLOP分析可视化.png")
        print("  💾 内存分析可视化.png")
        print("  🔄 训练性能可视化.png")
        print("  ⚡ 混合精度效益对比.png")
        print("  📈 综合性能仪表板.png")

        print("\n💡 关键学习收获:")
        print("  1. 🔍 学会了精确分析和可视化模型的计算复杂度")
        print("  2. 💾 理解了内存使用的组成和优化策略")
        print("  3.  🚀 掌握了训练性能的监控和对比方法")
        print("  4.  ⚡ 体验了混合精度训练的实际效益")
        print("  5.  📊 建立了模型性能评估的综合视角")

        print("\n🎯 下一步建议:")
        print("  • 尝试将这些工具应用到自己的模型上")
        print("  • 实验不同的优化策略组合")
        print("  • 深入学习Transformer架构的FLOP分析")
        print("  • 探索分布式训练的性能分析")

    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        print("💡 请检查依赖包是否正确安装:")
        print("   pip install torch matplotlib seaborn numpy psutil")


if __name__ == "__main__":
    main()