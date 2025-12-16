#!/usr/bin/env python3
"""
Roofline Model Visualization
支持 NVIDIA GPU 和 Apple MPS (MacBook)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from typing import List, Tuple, Optional

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class RooflineModel:
    """Roofline Model 分析器"""
    
    def __init__(self, device='cuda'):
        """
        初始化Roofline Model
        
        Args:
            device: 'cuda', 'mps', 或 'cpu'
        """
        self.device = device
        self.specs = self._get_gpu_specs(device)
        
    def _get_gpu_specs(self, device: str) -> dict:
        """获取GPU规格"""
        if device == 'cuda':
            # NVIDIA A100 规格
            return {
                'name': 'NVIDIA A100',
                'peak_flops': 312e12,      # TF32
                'peak_bandwidth': 1.5e12,  # 1.5 TB/s
                'ai_critical': 312e12 / 1.5e12  # 208 FLOP/Byte
            }
        elif device == 'mps':
            # Apple Silicon (M1/M2/M3) 规格
            # 注意: 这些是近似值，实际可能因型号而异
            return {
                'name': 'Apple MPS',
                'peak_flops': 20e12,       # 约20 TFLOP/s (FP32)
                'peak_bandwidth': 400e9,   # 约400 GB/s (统一内存)
                'ai_critical': 20e12 / 400e9  # 50 FLOP/Byte
            }
        else:  # cpu
            return {
                'name': 'CPU',
                'peak_flops': 0.5e12,     # 约0.5 TFLOP/s
                'peak_bandwidth': 50e9,    # 约50 GB/s
                'ai_critical': 0.5e12 / 50e9  # 10 FLOP/Byte
            }
    
    def plot_roofline(self, kernels: Optional[List[Tuple]] = None, 
                     save_path: Optional[str] = None):
        """
        绘制Roofline Model
        
        Args:
            kernels: [(name, ai, actual_perf_tflops), ...]
            save_path: 保存路径
        """
        peak_flops = self.specs['peak_flops']
        peak_bandwidth = self.specs['peak_bandwidth']
        ai_critical = self.specs['ai_critical']
        
        # AI范围 (对数刻度)
        ai_range = np.logspace(-1, 4, 1000)  # 0.1 到 10000
        
        # 计算Roofline
        performance = np.minimum(
            peak_bandwidth * ai_range,  # 内存限制
            peak_flops                  # 计算限制
        )
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 绘制主Roofline线
        ax.loglog(ai_range, performance / 1e12, 'b-', linewidth=3, 
                 label=f'Roofline ({self.specs["name"]})')
        
        # 标注转折点
        ridge_perf = peak_bandwidth * ai_critical / 1e12
        ax.axvline(x=ai_critical, color='r', linestyle='--', linewidth=2,
                  label=f'Ridge Point (AI={ai_critical:.0f})')
        ax.axhline(y=peak_flops/1e12, color='g', linestyle='--', linewidth=2,
                  label=f'Peak Compute ({peak_flops/1e12:.0f} TFLOP/s)')
        
        # 区域标注
        ax.text(ai_critical * 0.3, peak_flops / 1e12 * 0.3, 
               'Memory-Bound\nRegion', 
               fontsize=14, color='blue', ha='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.text(ai_critical * 3, peak_flops / 1e12 * 0.7, 
               'Compute-Bound\nRegion', 
               fontsize=14, color='green', ha='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # 添加kernel点
        if kernels:
            for name, ai, actual_perf in kernels:
                # 理论性能
                theoretical = min(peak_bandwidth * ai, peak_flops) / 1e12
                
                # 绘制点
                ax.plot(ai, actual_perf, 'ro', markersize=12, zorder=5)
                
                # 标注
                ax.annotate(name, (ai, actual_perf), 
                          xytext=(15, 15), 
                          textcoords='offset points',
                          fontsize=11,
                          bbox=dict(boxstyle='round,pad=0.5', 
                                   facecolor='yellow', alpha=0.8),
                          arrowprops=dict(arrowstyle='->', 
                                        connectionstyle='arc3,rad=0.2'))
                
                # 绘制优化空间箭头
                if actual_perf < theoretical * 0.95:  # 如果距离屋顶>5%
                    gap = theoretical - actual_perf
                    ax.arrow(ai, actual_perf, 0, gap * 0.8,
                           head_width=ai*0.15, head_length=gap*0.1,
                           fc='red', ec='red', alpha=0.6,
                           length_includes_head=True,
                           label='Optimization Space' if name == kernels[0][0] else '')
        
        # 设置
        ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Performance (TFLOP/s)', fontsize=14, fontweight='bold')
        ax.set_title(f'Roofline Model - {self.specs["name"]}', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=12, loc='upper left')
        ax.set_xlim(0.1, 10000)
        ax.set_ylim(0.01, max(peak_flops/1e12 * 1.2, 500))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ 图表已保存到: {save_path}")
        
        return fig, ax


def benchmark_operation(name: str, flop: float, bytes_transferred: float,
                       operation_func, device='mps', warmup=5, repeat=100):
    """
    基准测试操作
    
    Args:
        name: 操作名称
        flop: FLOP数量
        bytes_transferred: 数据传输量 (bytes)
        operation_func: 执行操作的函数
        device: 设备
        warmup: 预热次数
        repeat: 重复次数
    """
    # 预热
    for _ in range(warmup):
        operation_func()
    
    # 同步
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
    
    # 计时
    start = time.time()
    for _ in range(repeat):
        operation_func()
    
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
    
    elapsed = time.time() - start
    avg_time = elapsed / repeat
    
    # 计算性能
    actual_perf = flop / avg_time  # FLOP/s
    actual_perf_tflops = actual_perf / 1e12
    
    # 计算AI
    ai = flop / bytes_transferred
    
    return {
        'name': name,
        'ai': ai,
        'actual_perf_tflops': actual_perf_tflops,
        'time_ms': avg_time * 1000,
        'flop': flop,
        'bytes': bytes_transferred
    }


def test_operations_mps():
    """测试MPS上的各种操作"""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"测试设备: {device}")
    print(f"{'='*60}\n")
    
    if device == 'cpu':
        print("⚠️  MPS不可用，使用CPU")
    
    results = []
    
    # 1. 向量加法
    print("📊 测试1: 向量加法")
    size = 10_000_000
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    def vec_add():
        c = a + b
        return c
    
    flop = size  # N次加法
    bytes_transferred = 3 * size * 4  # 读a, b, 写c (FP32)
    
    result = benchmark_operation('VecAdd', flop, bytes_transferred, vec_add, device)
    results.append(result)
    print(f"  AI: {result['ai']:.3f} FLOP/Byte")
    print(f"  性能: {result['actual_perf_tflops']:.3f} TFLOP/s")
    print(f"  时间: {result['time_ms']:.3f} ms\n")
    
    # 2. 矩阵乘法 (小)
    print("📊 测试2: 小矩阵乘法 (256×256)")
    size = 256
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    def matmul_small():
        c = torch.matmul(a, b)
        return c
    
    flop = 2 * size ** 3
    bytes_transferred = 3 * size * size * 4  # 读a, b, 写c
    
    result = benchmark_operation('MatMul-256', flop, bytes_transferred, 
                                 matmul_small, device)
    results.append(result)
    print(f"  AI: {result['ai']:.1f} FLOP/Byte")
    print(f"  性能: {result['actual_perf_tflops']:.3f} TFLOP/s")
    print(f"  时间: {result['time_ms']:.3f} ms\n")
    
    # 3. 矩阵乘法 (中)
    print("📊 测试3: 中矩阵乘法 (1024×1024)")
    size = 1024
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    def matmul_medium():
        c = torch.matmul(a, b)
        return c
    
    flop = 2 * size ** 3
    bytes_transferred = 3 * size * size * 4
    
    result = benchmark_operation('MatMul-1024', flop, bytes_transferred,
                                 matmul_medium, device)
    results.append(result)
    print(f"  AI: {result['ai']:.1f} FLOP/Byte")
    print(f"  性能: {result['actual_perf_tflops']:.3f} TFLOP/s")
    print(f"  时间: {result['time_ms']:.3f} ms\n")
    
    # 4. 矩阵乘法 (大)
    print("📊 测试4: 大矩阵乘法 (2048×2048)")
    size = 2048
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    def matmul_large():
        c = torch.matmul(a, b)
        return c
    
    flop = 2 * size ** 3
    bytes_transferred = 3 * size * size * 4
    
    result = benchmark_operation('MatMul-2048', flop, bytes_transferred,
                                 matmul_large, device)
    results.append(result)
    print(f"  AI: {result['ai']:.1f} FLOP/Byte")
    print(f"  性能: {result['actual_perf_tflops']:.3f} TFLOP/s")
    print(f"  时间: {result['time_ms']:.3f} ms\n")
    
    # 5. LayerNorm
    print("📊 测试5: LayerNorm")
    batch, seq, hidden = 32, 512, 768
    x = torch.randn(batch, seq, hidden, device=device)
    
    def layernorm():
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        y = (x - mean) / torch.sqrt(var + 1e-5)
        return y
    
    flop = batch * seq * hidden * 3  # mean, var, normalize
    bytes_transferred = 2 * batch * seq * hidden * 4  # 读x, 写y
    
    result = benchmark_operation('LayerNorm', flop, bytes_transferred,
                                layernorm, device)
    results.append(result)
    print(f"  AI: {result['ai']:.3f} FLOP/Byte")
    print(f"  性能: {result['actual_perf_tflops']:.3f} TFLOP/s")
    print(f"  时间: {result['time_ms']:.3f} ms\n")
    
    return results


def analyze_on_roofline(results, device='mps'):
    """在Roofline图上分析结果"""
    roofline = RooflineModel(device=device)
    
    # 准备kernel数据
    kernels = [(r['name'], r['ai'], r['actual_perf_tflops']) 
               for r in results]
    
    # 绘制
    save_path = f'roofline_{device}.png'
    fig, ax = roofline.plot_roofline(kernels, save_path)
    
    # 打印分析报告
    print(f"\n{'='*80}")
    print(f"Roofline 分析报告 - {roofline.specs['name']}")
    print(f"{'='*80}\n")
    
    specs = roofline.specs
    print(f"GPU规格:")
    print(f"  峰值计算: {specs['peak_flops']/1e12:.1f} TFLOP/s")
    print(f"  峰值带宽: {specs['peak_bandwidth']/1e9:.1f} GB/s")
    print(f"  临界AI: {specs['ai_critical']:.1f} FLOP/Byte\n")
    
    print(f"{'操作':<15} {'AI':<10} {'实际性能':<12} {'理论性能':<12} {'效率':<8} {'瓶颈':<15}")
    print(f"{'-'*80}")
    
    for r in results:
        ai = r['ai']
        actual = r['actual_perf_tflops']
        theoretical = min(specs['peak_bandwidth'] * ai, specs['peak_flops']) / 1e12
        efficiency = actual / theoretical * 100
        
        if ai < specs['ai_critical']:
            bottleneck = "Memory-Bound"
        else:
            bottleneck = "Compute-Bound"
        
        print(f"{r['name']:<15} {ai:<10.1f} {actual:<12.3f} {theoretical:<12.3f} "
              f"{efficiency:<7.1f}% {bottleneck:<15}")
    
    print(f"\n{'='*80}\n")
    
    plt.show()
    
    return roofline, results


if __name__ == '__main__':
    print("🚀 Roofline Model 性能分析")
    print("=" * 60)
    
    # 检查MPS可用性
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal) 可用")
        device = 'mps'
    else:
        print("⚠️  MPS不可用，将使用CPU")
        device = 'cpu'
    
    # 运行测试
    results = test_operations_mps()
    
    # 在Roofline上分析
    roofline, results = analyze_on_roofline(results, device=device)
    
    print("✅ 分析完成!")
