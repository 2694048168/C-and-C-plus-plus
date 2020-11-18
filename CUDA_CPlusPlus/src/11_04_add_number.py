#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: weili
# @filename: 11_04_add_number.py
# @copyright: https://gitee.com/weili_yzzcq/C-and-C-plus-plus/CUDA_CPlusPlus/
# @copyright: https://github.com/2694048168/C-and-C-plus-plus/CUDA_CPlusPlus/
# @function: pycuda 使用 Python 利用 CDUA 进行两个数相加。

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
  #include <stdio.h>

  __global__ void add_num(float *device_result, float *device_a, float *device_b)
  {
    const int i = threadIdx.x;  
    device_result[i] = device_a[i] + device_b[i];
  }
""")

# SourceModule 类实例化对象 mod 的 get_function 方法创建指向函数的指针。
add_num = mod.get_function("add_num")

# 正态分布函数生成随机数，astype(numpy.float32) 方法转换为单精度浮点数字。
host_a = np.random.randn(1).astype(np.float32)
host_b = np.random.randn(1).astype(np.float32)
# 存储结果值，初始化为零。
host_result = np.zeros_like(host_a)

# 使用 PyCUDA 中 driver 类的 mem_alloc 函数来分配设备 GPU 上的内存。
device_a = drv.mem_alloc(host_a.nbytes)
device_b = drv.mem_alloc(host_b.nbytes)
device_result = drv.mem_alloc(host_result.nbytes)

# 使用 PyCUDA 中 driver 类的 memcpy_htod 函数将数据从主机内存复制到设备显存。
drv.memcpy_htod(device_a,host_a)
drv.memcpy_htod(device_b,host_b)

# Python 中内核调用，简单语法：
# kernel (parameters for kernel, block=(tx, ty, tz), grid=(bx, by, bz))
# parameters for kernel 表示内核参数；
# block=(tx, ty, tz) 表示启动的每个块里面的线程数；
# grid=(bx, by, bz) 表示启动的块的数；
# 元组 表示三维中的线程块和线程，内核启动的总线程是这两个数字的乘积。
add_num(device_result, device_a, device_b, block=(1,1,1), grid=(1,1))

# 使用 PyCUDA 中 driver 类的 memcpy_htod 函数将数据从设备显存复制到主机内存。
drv.memcpy_dtoh(host_result,device_result)

print("Addition on GPU:")
print(host_a[0],"+", host_b[0] , "=" , host_result[0])
