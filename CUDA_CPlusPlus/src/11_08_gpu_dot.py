#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: weili
# @filename: 11_08_gpu_dot.py
# @copyright: https://gitee.com/weili_yzzcq/C-and-C-plus-plus/CUDA_CPlusPlus/
# @copyright: https://github.com/2694048168/C-and-C-plus-plus/CUDA_CPlusPlus/
# @function: pycuda 计算数值的点乘运算，内积运算。

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

# GPU设备上 类似 numpy 功能的 gpuarray 类。
import pycuda.gpuarray as gpuarray

import numpy as np
import time

# 定义常量 类似 #define N 1
n = 10000
# 正态分布函数生成随机数，astype(numpy.float32) 方法转换为单精度浮点数字。
a = np.float32(np.random.randint(1,5,(n,n)))
b = np.float32(np.random.randint(1,5,(n,n)))
    
tic = time.time()
axb = a * b

#print(numpy.dot(a,b))
toc = time.time() - tic
print("Dot Product on CPU")
print(toc,"s")

# CUDA Event.
start = drv.Event()
end=drv.Event()
start.record()

# GPU设备上 类似 numpy 功能的 gpuarray 类。
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
axbGPU = gpuarray.dot(a_gpu, b_gpu)

# CUDA Event.
end.record()
# Sync.
end.synchronize()
secs = start.time_till(end) * 1e-3

print("Dot Product on GPU")
print("%fs" % (secs))
if(np.sum(axb) == axbGPU.get()):
    print("The computed dor product is correct")
