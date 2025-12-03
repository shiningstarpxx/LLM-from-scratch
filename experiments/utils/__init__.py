"""
通用工具模块

包含:
- device_utils: 设备检测与管理（CUDA/MPS/CPU）
"""

from .device_utils import DeviceManager, get_device, get_device_manager, to_device

__all__ = ['DeviceManager', 'get_device', 'get_device_manager', 'to_device']
