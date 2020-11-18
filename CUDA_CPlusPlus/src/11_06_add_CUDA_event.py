#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: weili
# @filename: 11_06_add_CUDA_event.py
# @copyright: https://gitee.com/weili_yzzcq/C-and-C-plus-plus/CUDA_CPlusPlus/
# @copyright: https://github.com/2694048168/C-and-C-plus-plus/CUDA_CPlusPlus/
# @function: pycuda 使用 CUDA Event 度量其程序的性能。

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
import math
import time

# 定义常量 类似 #define N 1
N = 1000000

# C/C++ 内核代码作为构造函数被传入 SourceModule 类中并实例化对象 mod。
# 该内核函数由 nvcc 编译器进行编译。

mod = SourceModule("""
  __global__ void add_num(float *device_result, float *device_a, float *device_b,int N)
  {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;	
	  while (tid < N)
    {
      // 确保线程索引不会超出数组元素的索引，数组越界情况。
      device_result[tid] = device_a[tid] + device_b[tid];
      tid = tid + blockDim.x * gridDim.x;
    }
  }
""")

# 使用 PyCUDA 中 driver 类的 Event 函数来创建 CUDA 事件。 
start = drv.Event()
end = drv.Event()

# SourceModule 类实例化对象 mod 的 get_function 方法创建指向函数的指针。
add_num = mod.get_function("add_num")

# 正态分布函数生成随机数，astype(numpy.float32) 方法转换为单精度浮点数字。
host_a = np.random.randn(N).astype(np.float32)
host_b = np.random.randn(N).astype(np.float32)
# 存储结果值，初始化为零。
host_result = np.zeros_like(host_a)
host_result1 = np.zeros_like(host_a)

# 线程块总数的计算结果可能是一个浮点数，
# 使用 numpy 的 ceil 函数方法将其转换为下一个最高的整数值。
n_blocks = math.ceil((N / 1024) + 1)

# 开始记录 CDUA Event。
start.record()


# Python 中内核调用，简单语法：
# kernel (parameters for kernel, block=(tx, ty, tz), grid=(bx, by, bz))
# PyCDUA 为内核调用提供简单 API，不需要内存分配和复制，有 API 隐式完成的，
# 通过使用 PyCDUA 中 driver 类的 In 和 Out 函数来实现。
# 直接调用内核，通过 driver.In 和 driver.Out 指定数据的方向来修改内核函数。
add_num(drv.Out(host_result), drv.In(host_a), drv.In(host_b), block=(1024,1,1), grid=(n_blocks,1))

# 结束 CUDA Event 的记录。
end.record()
end.synchronize() # 同步
# 计算 CDUA Event 的度量事件，ms 转化为 s。
# 计算内核执行的事件戳之间的差异，使用 time_till 方法测量。
secs = start.time_till(end) * 1e-3

print("Addition of %d element of GPU"%N)
print("%fs" % (secs))

start = time.time()
for i in range(0,N):
    host_result1[i] = host_a[i] + host_b[i]
end = time.time()
print("Addition of %d element of CPU"%N)
print(end-start,"s")
