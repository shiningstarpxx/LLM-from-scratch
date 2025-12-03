"""
设备检测工具类
支持自动检测并使用最优设备：CUDA (NVIDIA GPU) / MPS (Apple Silicon) / CPU
"""

import torch
import sys


class DeviceManager:
    """统一的设备管理类，自动检测并配置最优计算设备"""

    def __init__(self, prefer_device=None):
        """
        初始化设备管理器

        Args:
            prefer_device: 偏好设备类型 ('cuda', 'mps', 'cpu')，None则自动检测
        """
        self.device = self._detect_device(prefer_device)
        self.device_type = self.device.type

        # 打印设备信息
        self._print_device_info()

    def _detect_device(self, prefer_device=None):
        """
        检测并返回最优设备

        优先级: CUDA > MPS > CPU
        """
        if prefer_device:
            # 用户指定设备
            if prefer_device == 'cuda' and torch.cuda.is_available():
                return torch.device('cuda')
            elif prefer_device == 'mps' and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                print(f"Warning: {prefer_device} not available, falling back to CPU")
                return torch.device('cpu')

        # 自动检测最优设备
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def _print_device_info(self):
        """打印当前使用的设备信息"""
        print("=" * 60)
        print(f"🖥️  Device Configuration")
        print("=" * 60)
        print(f"Selected Device: {self.device}")
        print(f"Device Type: {self.device_type.upper()}")

        if self.device_type == 'cuda':
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
        elif self.device_type == 'mps':
            print(f"Platform: Apple Silicon (MPS Backend)")
            print(f"PyTorch MPS: Enabled")
        else:
            print(f"Platform: CPU only")
            print(f"Warning: No GPU acceleration available")

        print(f"PyTorch Version: {torch.__version__}")
        print(f"Python Version: {sys.version.split()[0]}")
        print("=" * 60)

    def to_device(self, *tensors):
        """
        将tensor或模型移动到设备

        Args:
            *tensors: 一个或多个tensor/module

        Returns:
            如果输入一个，返回一个；如果输入多个，返回tuple
        """
        if len(tensors) == 1:
            return tensors[0].to(self.device)
        return tuple(t.to(self.device) for t in tensors)

    def is_cuda(self):
        """是否使用CUDA"""
        return self.device_type == 'cuda'

    def is_mps(self):
        """是否使用MPS"""
        return self.device_type == 'mps'

    def is_cpu(self):
        """是否使用CPU"""
        return self.device_type == 'cpu'

    def empty_cache(self):
        """清空设备缓存"""
        if self.is_cuda():
            torch.cuda.empty_cache()
        elif self.is_mps():
            torch.mps.empty_cache()

    def synchronize(self):
        """同步设备（用于准确计时）"""
        if self.is_cuda():
            torch.cuda.synchronize()
        elif self.is_mps():
            torch.mps.synchronize()

    def get_memory_info(self):
        """获取设备内存使用情况"""
        if self.is_cuda():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            return {
                'allocated': f'{allocated:.2f} GB',
                'reserved': f'{reserved:.2f} GB'
            }
        elif self.is_mps():
            # MPS目前没有直接的内存查询API
            return {'info': 'MPS memory info not directly available'}
        else:
            return {'info': 'CPU mode'}


# 全局设备管理器实例
_device_manager = None


def get_device_manager(prefer_device=None):
    """
    获取全局设备管理器单例

    Args:
        prefer_device: 偏好设备类型 ('cuda', 'mps', 'cpu')

    Returns:
        DeviceManager实例
    """
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager(prefer_device)
    return _device_manager


def get_device(prefer_device=None):
    """
    快捷函数：获取设备

    Args:
        prefer_device: 偏好设备类型

    Returns:
        torch.device对象
    """
    return get_device_manager(prefer_device).device


# 便捷函数
def to_device(*tensors, prefer_device=None):
    """
    快捷函数：将tensor移动到设备

    Args:
        *tensors: 一个或多个tensor
        prefer_device: 偏好设备类型

    Returns:
        移动后的tensor(s)
    """
    return get_device_manager(prefer_device).to_device(*tensors)


if __name__ == '__main__':
    # 测试设备检测
    print("\n测试1: 自动检测设备")
    dm = DeviceManager()

    print("\n测试2: 创建tensor并移动到设备")
    x = torch.randn(3, 3)
    print(f"Original device: {x.device}")
    x = dm.to_device(x)
    print(f"After to_device: {x.device}")

    print("\n测试3: 同时移动多个tensor")
    a = torch.randn(2, 2)
    b = torch.randn(2, 2)
    a, b = dm.to_device(a, b)
    print(f"Tensor a device: {a.device}")
    print(f"Tensor b device: {b.device}")

    print("\n测试4: 设备类型检查")
    print(f"Is CUDA: {dm.is_cuda()}")
    print(f"Is MPS: {dm.is_mps()}")
    print(f"Is CPU: {dm.is_cpu()}")

    print("\n测试5: 内存信息")
    print(dm.get_memory_info())

    print("\n✅ 设备工具测试完成！")
