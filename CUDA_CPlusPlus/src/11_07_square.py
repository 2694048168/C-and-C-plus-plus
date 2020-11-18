#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: weili
# @filename: 11_07_square.py
# @copyright: https://gitee.com/weili_yzzcq/C-and-C-plus-plus/CUDA_CPlusPlus/
# @copyright: https://github.com/2694048168/C-and-C-plus-plus/CUDA_CPlusPlus/
# @function: pycuda 对矩阵元素进行平方运算。

# 配置 PyCUDA 需要的 path 环境变量 和 CUDA_PATH 环境变量
# import os
# path_add = r'D:\Nvidia\NVIDIA GPU Computing Toolkit\CUDA\V11.0\bin;D:\Nvidia\NVIDIA GPU Computing Toolkit\CUDA\V11.0\libnvvp;'
# cuda_path = r"D:\Nvidia\NVIDIA GPU Computing Toolkit\CUDA\V11.0"
# path = os.getenv('path')
# os.environ['path'] = path_add + path
# os.environ['CUDA_PATH'] = cuda_path

# driver 包含内存管理功能、设备属性、数据方向功能等等。
import pycuda.driver as drv

# autoinit 用于设备初始化、上下文创建和内存清理，可以手动完成，不需要该模块。
import pycuda.autoinit

# 导入 compiler 模块中的 SourceModule 类，
# SourceModule 类用于在 PyCUDA 中定义类 C 的内核函数。
from pycuda.compiler import SourceModule

import numpy as np

# C/C++ 内核代码作为构造函数被传入 SourceModule 类中并实例化对象 mod。
# 该内核函数由 nvcc 编译器进行编译。

mod = SourceModule("""
  __global__ void square(float *device_a)
  {
    int idx = threadIdx.x + threadIdx.y * 5;
    device_a[idx] = device_a[idx] * device_a[idx];
  }
""")

# 使用 PyCUDA 中 driver 类的 Event 函数来创建 CUDA 事件。 
start = drv.Event()
end = drv.Event()

# 正态分布函数生成随机数，astype(numpy.float32) 方法转换为单精度浮点数字。
host_a = np.random.randint(1,5,(5, 5))
host_a = host_a.astype(np.float32)
# 拷贝复制一份数据。
host_b = host_a.copy()

# 开始记录 CDUA Event。
start.record()

# drive 类 mem_alloc 方法分配设备上显存。
device_a = drv.mem_alloc(host_a.size * host_a.dtype.itemsize)
# drive 类 memcpy_htod 方法拷贝数据从主机到设备。
drv.memcpy_htod(device_a, host_a)

# 内核调用。
square = mod.get_function("square")
square(device_a, block=(5, 5, 1), grid=(1, 1), shared=0)

# 保存结果，在主机和设备上。
host_result = np.empty_like(host_a)
drv.memcpy_dtoh(host_result, device_a)

# CUDA Event 结束记录。
end.record()
end.synchronize()

# 度量计算记录的时间戳之差，单位转换从 ms 到 s 。
secs = start.time_till(end) * 1e-3

print("Time of Squaring on GPU without inout")
print("%fs" % (secs))
print("original array:")
print(host_a)
print("Square with kernel:")
print(host_result)

# ############ Using inout functionality of driver class ############
start.record()
start.synchronize()
square(drv.InOut(host_a), block=(5, 5, 1))
end.record()
end.synchronize()

print("Square with InOut:")
print(host_a)
secs = start.time_till(end) * 1e-3
print("Time of Squaring on GPU with inout")
print("%fs" % (secs))

# ############## Using gpuarray class ##############
# python 为数值计算提供了 numpy 库；
# PyCUDA 提供了一个类似 numpy 的 gpuarray 类，用来存储数据并在GPU设备上执行计算。
import pycuda.gpuarray as gpuarray

start.record()
start.synchronize()
host_b = np.random.randint(1,5,(5, 5))
#host_b = host_b.astype(np.float32)
device_b = gpuarray.to_gpu(host_b.astype(np.float32))
host_result = (device_b ** 2).get()
end.record()
end.synchronize()
print("original array:")
print(host_b)
print("Squared with gpuarray:")
print(host_result)
secs = start.time_till(end) * 1e-3
print("Time of Squaring on GPU with gpuarray")
print("%fs" % (secs))
