"""
Lecture 02: PyTorch 基础构建块
展示张量操作、计算图、自动微分等核心概念

主要内容:
1. Tensor 操作基础
2. 自动微分 (Autograd)
3. FLOP 计算
4. 内存分析
5. 简单训练循环

支持设备: CUDA / MPS (Apple Silicon) / CPU
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import time
from typing import Dict, Tuple

from utils.device_utils import DeviceManager, get_device


class TensorBasics:
    """张量操作基础演示"""

    def __init__(self, device_manager: DeviceManager):
        self.dm = device_manager

    def demonstrate_tensor_creation(self):
        """演示各种tensor创建方法"""

        print("\n" + "=" * 60)
        print("📦 Tensor Creation Methods")
        print("=" * 60)

        # 1. 从Python列表创建
        t1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        print(f"\n1. From list: shape={t1.shape}, dtype={t1.dtype}")
        print(f"   {t1}")

        # 2. 全零/全一
        t2 = torch.zeros(3, 4)
        t3 = torch.ones(3, 4)
        print(f"\n2. Zeros: shape={t2.shape}")
        print(f"3. Ones: shape={t3.shape}")

        # 3. 随机初始化
        t4 = torch.randn(3, 4)  # 正态分布
        t5 = torch.rand(3, 4)   # 均匀分布 [0, 1)
        print(f"\n4. Random Normal: mean={t4.mean():.3f}, std={t4.std():.3f}")
        print(f"5. Random Uniform: min={t5.min():.3f}, max={t5.max():.3f}")

        # 4. 在设备上创建
        t6 = torch.randn(3, 4, device=self.dm.device)
        print(f"\n6. On device '{self.dm.device}': shape={t6.shape}")

        return t6

    def demonstrate_operations(self, x: torch.Tensor):
        """演示tensor运算"""

        print("\n" + "=" * 60)
        print("⚙️  Tensor Operations")
        print("=" * 60)

        # 确保在正确设备上
        x = self.dm.to_device(x)

        # 1. 逐元素运算
        print("\n1. Element-wise operations:")
        y = torch.randn_like(x)

        add_result = x + y
        mul_result = x * y
        print(f"   x + y: shape={add_result.shape}")
        print(f"   x * y: shape={mul_result.shape}")

        # 2. 矩阵运算
        print("\n2. Matrix operations:")
        a = torch.randn(3, 4, device=self.dm.device)
        b = torch.randn(4, 5, device=self.dm.device)

        matmul = torch.matmul(a, b)  # 或 a @ b
        print(f"   matmul(3x4, 4x5): shape={matmul.shape}")

        # 3. Reduction operations
        print("\n3. Reduction operations:")
        print(f"   sum: {x.sum():.3f}")
        print(f"   mean: {x.mean():.3f}")
        print(f"   max: {x.max():.3f}")
        print(f"   argmax: {x.argmax().item()}")

        # 4. 形状变换
        print("\n4. Shape transformations:")
        x_flat = x.reshape(-1)
        x_expand = x.unsqueeze(0)  # 增加维度
        print(f"   reshape to 1D: {x_flat.shape}")
        print(f"   unsqueeze(0): {x_expand.shape}")

        # 5. 广播机制
        print("\n5. Broadcasting:")
        a = torch.randn(3, 1, device=self.dm.device)
        b = torch.randn(1, 4, device=self.dm.device)
        c = a + b  # 自动广播
        print(f"   (3,1) + (1,4) → {c.shape}")


class AutogradDemo:
    """自动微分演示"""

    def __init__(self, device_manager: DeviceManager):
        self.dm = device_manager

    def demonstrate_autograd(self):
        """演示PyTorch自动微分机制"""

        print("\n" + "=" * 60)
        print("🔄 Autograd - Automatic Differentiation")
        print("=" * 60)

        # 1. 基础梯度计算
        print("\n1. Basic gradient computation:")
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = x ** 2  # y = x^2
        z = y.sum()  # z = x1^2 + x2^2

        z.backward()

        print(f"   x = {x.tolist()}")
        print(f"   y = x^2 = {y.tolist()}")
        print(f"   z = sum(y) = {z.item()}")
        print(f"   dz/dx = 2x = {x.grad.tolist()}")  # 应该是 [4.0, 6.0]

        # 2. 计算图演示
        print("\n2. Computation graph:")
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = torch.tensor([3.0, 4.0], requires_grad=True)

        c = a + b       # c = a + b
        d = c * a       # d = c * a = (a+b) * a
        e = d.sum()     # e = sum(d)

        e.backward()

        print(f"   a = {a.tolist()}, b = {b.tolist()}")
        print(f"   e = sum((a+b)*a)")
        print(f"   de/da = 2a + b = {a.grad.tolist()}")  # [5.0, 8.0]
        print(f"   de/db = a = {b.grad.tolist()}")       # [1.0, 2.0]

        # 3. 梯度累积
        print("\n3. Gradient accumulation (记得清零!):")
        x = torch.tensor([1.0], requires_grad=True)

        for i in range(3):
            y = x ** 2
            y.backward()
            print(f"   After backward {i+1}: grad = {x.grad.item()}")

        print("   ⚠️ 注意：梯度会累积！训练时需要optimizer.zero_grad()")

        # 4. detach和no_grad
        print("\n4. detach() and no_grad():")
        x = torch.tensor([2.0], requires_grad=True)
        y = x ** 2

        y_detached = y.detach()  # 切断梯度
        print(f"   y.requires_grad: {y.requires_grad}")
        print(f"   y.detach().requires_grad: {y_detached.requires_grad}")

        with torch.no_grad():
            z = x ** 3
        print(f"   with no_grad: z.requires_grad = {z.requires_grad}")


class FLOPCalculator:
    """FLOP（浮点运算次数）计算器"""

    @staticmethod
    def count_matmul_flops(m: int, k: int, n: int) -> int:
        """
        计算矩阵乘法FLOP

        C[m, n] = A[m, k] @ B[k, n]

        每个输出元素需要:
        - k次乘法
        - k-1次加法
        ≈ 2k FLOPs

        总计: m * n * 2k = 2mnk FLOPs
        """
        return 2 * m * n * k

    @staticmethod
    def count_linear_flops(batch: int, in_features: int, out_features: int, has_bias: bool = True) -> int:
        """
        计算Linear层FLOP

        Y = X @ W^T + b
        X: [batch, in_features]
        W: [out_features, in_features]
        """
        matmul_flops = FLOPCalculator.count_matmul_flops(batch, in_features, out_features)
        bias_flops = batch * out_features if has_bias else 0
        return matmul_flops + bias_flops

    @staticmethod
    def count_attention_flops(batch: int, seq_len: int, d_model: int, num_heads: int) -> Dict[str, int]:
        """
        计算Multi-Head Attention的FLOP分解

        详细分解：
        1. Q, K, V projection: 3 * 2*B*L*D*D FLOPs
        2. QK^T: B*H * 2*L*L*(D/H) FLOPs
        3. Softmax: ~5*B*H*L*L FLOPs (approx)
        4. Attention * V: B*H * 2*L*(D/H)*L FLOPs
        5. Output projection: 2*B*L*D*D FLOPs
        """
        d_head = d_model // num_heads
        B, L, D, H = batch, seq_len, d_model, num_heads

        flops = {
            'qkv_proj': 3 * 2 * B * L * D * D,
            'qk_matmul': B * H * 2 * L * L * d_head,
            'softmax': 5 * B * H * L * L,  # 近似
            'attn_v': B * H * 2 * L * d_head * L,
            'out_proj': 2 * B * L * D * D
        }

        flops['total'] = sum(flops.values())
        return flops

    @staticmethod
    def demonstrate():
        """FLOP计算演示"""

        print("\n" + "=" * 60)
        print("🧮 FLOP (Floating Point Operations) Calculation")
        print("=" * 60)

        # 1. 矩阵乘法
        m, k, n = 64, 512, 256  # batch, input, output
        matmul_flops = FLOPCalculator.count_matmul_flops(m, k, n)
        print(f"\n1. Matrix multiplication [{m}x{k}] @ [{k}x{n}]:")
        print(f"   FLOPs: {matmul_flops:,} = {matmul_flops/1e6:.2f}M")

        # 2. Linear层
        batch, in_f, out_f = 32, 768, 3072  # 典型FFN扩展
        linear_flops = FLOPCalculator.count_linear_flops(batch, in_f, out_f)
        print(f"\n2. Linear layer (batch={batch}, {in_f}→{out_f}):")
        print(f"   FLOPs: {linear_flops:,} = {linear_flops/1e6:.2f}M")

        # 3. Attention
        batch, seq_len, d_model, num_heads = 32, 512, 768, 12
        attn_flops = FLOPCalculator.count_attention_flops(batch, seq_len, d_model, num_heads)
        print(f"\n3. Multi-Head Attention (batch={batch}, seq={seq_len}, d={d_model}, h={num_heads}):")
        print(f"   QKV Projection: {attn_flops['qkv_proj']/1e9:.2f}G")
        print(f"   QK^T:           {attn_flops['qk_matmul']/1e9:.2f}G")
        print(f"   Softmax:        {attn_flops['softmax']/1e6:.2f}M")
        print(f"   Attention×V:    {attn_flops['attn_v']/1e9:.2f}G")
        print(f"   Output Proj:    {attn_flops['out_proj']/1e9:.2f}G")
        print(f"   -" * 30)
        print(f"   Total:          {attn_flops['total']/1e9:.2f}G FLOPs")

        # 4. 复杂度分析
        print("\n📊 Complexity Analysis:")
        print(f"   Attention O(n²·d): 随sequence length二次增长")
        print(f"   seq_len=512:  {FLOPCalculator.count_attention_flops(1, 512, 768, 12)['total']/1e9:.2f}G")
        print(f"   seq_len=1024: {FLOPCalculator.count_attention_flops(1, 1024, 768, 12)['total']/1e9:.2f}G")
        print(f"   seq_len=2048: {FLOPCalculator.count_attention_flops(1, 2048, 768, 12)['total']/1e9:.2f}G")


class MemoryProfiler:
    """内存使用分析器"""

    def __init__(self, device_manager: DeviceManager):
        self.dm = device_manager

    def estimate_tensor_memory(self, shape: Tuple, dtype: torch.dtype = torch.float32) -> int:
        """估算tensor内存占用（bytes）"""
        numel = 1
        for dim in shape:
            numel *= dim

        dtype_size = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.int8: 1,
            torch.int32: 4,
            torch.int64: 8,
        }

        return numel * dtype_size.get(dtype, 4)

    def estimate_model_memory(self, model: nn.Module) -> Dict[str, float]:
        """估算模型内存占用"""
        param_bytes = 0
        param_count = 0

        for name, param in model.named_parameters():
            param_bytes += param.numel() * param.element_size()
            param_count += param.numel()

        buffer_bytes = 0
        for name, buf in model.named_buffers():
            buffer_bytes += buf.numel() * buf.element_size()

        return {
            'param_count': param_count,
            'param_mb': param_bytes / 1024 / 1024,
            'buffer_mb': buffer_bytes / 1024 / 1024,
            'total_mb': (param_bytes + buffer_bytes) / 1024 / 1024
        }

    def demonstrate(self):
        """内存分析演示"""

        print("\n" + "=" * 60)
        print("💾 Memory Profiling")
        print("=" * 60)

        # 1. Tensor内存估算
        print("\n1. Tensor memory estimation:")

        shapes_dtypes = [
            ((1024, 1024), torch.float32, "1K×1K FP32"),
            ((1024, 1024), torch.float16, "1K×1K FP16"),
            ((8192, 8192), torch.float32, "8K×8K FP32"),
            ((32, 512, 768), torch.float32, "Typical hidden states"),
        ]

        for shape, dtype, desc in shapes_dtypes:
            mem_bytes = self.estimate_tensor_memory(shape, dtype)
            print(f"   {desc}: {mem_bytes/1024/1024:.2f} MB")

        # 2. 模型内存估算
        print("\n2. Model memory estimation:")

        class SimpleTransformerBlock(nn.Module):
            def __init__(self, d_model=768, n_heads=12, ff_dim=3072):
                super().__init__()
                self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
                self.ff = nn.Sequential(
                    nn.Linear(d_model, ff_dim),
                    nn.GELU(),
                    nn.Linear(ff_dim, d_model)
                )
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)

            def forward(self, x):
                x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
                x = x + self.ff(self.norm2(x))
                return x

        model = SimpleTransformerBlock()
        mem_info = self.estimate_model_memory(model)

        print(f"   Single Transformer Block (d=768, h=12, ff=3072):")
        print(f"     Parameters: {mem_info['param_count']:,} ({mem_info['param_count']/1e6:.2f}M)")
        print(f"     Memory: {mem_info['total_mb']:.2f} MB")

        # 3. 完整模型估算
        print("\n3. Full model estimation (GPT-2 scale):")
        num_layers = 12
        estimated_total = mem_info['total_mb'] * num_layers
        print(f"   {num_layers} layers: ~{estimated_total:.0f} MB ({estimated_total/1024:.2f} GB)")

        # 4. 激活内存（训练时）
        print("\n4. Activation memory during training:")
        batch, seq_len, d_model = 32, 512, 768
        activation_size = self.estimate_tensor_memory((batch, seq_len, d_model))
        print(f"   Hidden states (batch={batch}, seq={seq_len}, d={d_model}): {activation_size/1024/1024:.2f} MB")
        print(f"   ⚠️ 训练时需要保存中间激活用于反向传播")
        print(f"   12层Transformer激活内存约: ~{activation_size*12*3/1024/1024:.0f} MB")


class SimpleTrainingLoop:
    """简单训练循环演示"""

    def __init__(self, device_manager: DeviceManager):
        self.dm = device_manager

    def demonstrate(self):
        """演示完整的训练循环"""

        print("\n" + "=" * 60)
        print("🏋️ Simple Training Loop Demo")
        print("=" * 60)

        # 1. 创建简单模型
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.dm.device)

        print(f"\n1. Model created on {self.dm.device}")

        # 2. 准备数据（简单回归任务）
        torch.manual_seed(42)
        X = torch.randn(1000, 10, device=self.dm.device)
        y = X.sum(dim=1, keepdim=True) + torch.randn(1000, 1, device=self.dm.device) * 0.1
        print(f"2. Data: X={X.shape}, y={y.shape}")

        # 3. 定义优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # 4. 训练循环
        print("\n3. Training loop:")
        epochs = 100
        batch_size = 64
        num_batches = len(X) // batch_size

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0

            for i in range(num_batches):
                # 获取batch
                start = i * batch_size
                end = start + batch_size
                batch_x = X[start:end]
                batch_y = y[start:end]

                # Forward pass
                pred = model(batch_x)
                loss = criterion(pred, batch_y)

                # Backward pass
                optimizer.zero_grad()  # ⚠️ 重要：清零梯度
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / num_batches

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.4f}")

        # 5. 评估
        print("\n4. Evaluation:")
        model.eval()
        with torch.no_grad():
            test_pred = model(X[:10])
            print(f"   Predictions vs Targets (first 5):")
            for i in range(5):
                print(f"   pred={test_pred[i].item():.3f}, target={y[i].item():.3f}")

        # 6. 训练要点总结
        print("\n📝 Training Loop Key Points:")
        print("""
        1. model.train() / model.eval() - 切换模式
        2. optimizer.zero_grad() - 清零梯度（每个batch前）
        3. loss.backward() - 反向传播计算梯度
        4. optimizer.step() - 更新参数
        5. with torch.no_grad() - 推理时禁用梯度计算
        """)


def run_all_demos():
    """运行所有演示"""

    # 初始化设备
    dm = DeviceManager()

    # 1. Tensor基础
    tensor_demo = TensorBasics(dm)
    tensor_demo.demonstrate_tensor_creation()
    x = torch.randn(3, 4, device=dm.device)
    tensor_demo.demonstrate_operations(x)

    # 2. Autograd
    autograd_demo = AutogradDemo(dm)
    autograd_demo.demonstrate_autograd()

    # 3. FLOP计算
    FLOPCalculator.demonstrate()

    # 4. 内存分析
    memory_demo = MemoryProfiler(dm)
    memory_demo.demonstrate()

    # 5. 训练循环
    training_demo = SimpleTrainingLoop(dm)
    training_demo.demonstrate()


if __name__ == '__main__':
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                                                                    ║")
    print("║           CS336 Lecture 02 - PyTorch Basics                       ║")
    print("║                                                                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    run_all_demos()

    print("\n" + "=" * 70)
    print("✅ PyTorch Basics Demo完成！")
    print("=" * 70)
    print("\n💡 Next Steps:")
    print("  - Explore torch.nn modules")
    print("  - Learn about DataLoader and Dataset")
    print("  - Practice with real datasets")
    print("\n")
