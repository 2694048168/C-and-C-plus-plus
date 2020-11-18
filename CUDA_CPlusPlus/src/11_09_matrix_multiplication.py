#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: weili
# @filename: 11_09_matrix_multiplication.py
# @copyright: https://gitee.com/weili_yzzcq/C-and-C-plus-plus/CUDA_CPlusPlus/
# @copyright: https://github.com/2694048168/C-and-C-plus-plus/CUDA_CPlusPlus/
# @function: pycuda 计算矩阵乘法运算。

# 配置 PyCUDA 需要的 path 环境变量 和 CUDA_PATH 环境变量
# import os
# path_add = r'D:\Nvidia\NVIDIA GPU Computing Toolkit\CUDA\V11.0\bin;D:\Nvidia\NVIDIA GPU Computing Toolkit\CUDA\V11.0\libnvvp;'
# cuda_path = r"D:\Nvidia\NVIDIA GPU Computing Toolkit\CUDA\V11.0"
# path = os.getenv('path')
# os.environ['path'] = path_add + path
# os.environ['CUDA_PATH'] = cuda_path

# driver 包含内存管理功能、设备属性、数据方向功能等等。
from pycuda import driver, gpuarray

# autoinit 用于设备初始化、上下文创建和内存清理，可以手动完成，不需要该模块。
import pycuda.autoinit

# 导入 compiler 模块中的 SourceModule 类，
# SourceModule 类用于在 PyCUDA 中定义类 C 的内核函数。
from pycuda.compiler import SourceModule

import numpy as np

# 方阵大小。
MATRIX_SIZE = 3 

matrix_mul_kernel = """
  __global__ void Matrix_Mul_Kernel(float *d_a, float *d_b, float *d_c)
  {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float value = 0;
  
    for (int i = 0; i < %(MATRIX_SIZE)s; ++i) 
    {
      float d_a_element = d_a[ty * %(MATRIX_SIZE)s + i];
      float d_b_element = d_b[i * %(MATRIX_SIZE)s + tx];
        value += d_a_element * d_b_element;
    }
 
    d_c[ty * %(MATRIX_SIZE)s + tx] = value;
  } """
  
matrix_mul = matrix_mul_kernel % {'MATRIX_SIZE': MATRIX_SIZE}
  
mod = SourceModule(matrix_mul)

h_a = np.random.randint(1,5,(MATRIX_SIZE, MATRIX_SIZE)).astype(np.float32)
h_b = np.random.randint(1,5,(MATRIX_SIZE, MATRIX_SIZE)).astype(np.float32)
 
# compute on the CPU to verify GPU computation
h_c_cpu = np.dot(h_a, h_b)

d_a = gpuarray.to_gpu(h_a) 
d_b = gpuarray.to_gpu(h_b)

d_c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

matrixmul = mod.get_function("Matrix_Mul_Kernel")
 
matrixmul(d_a, d_b,d_c_gpu, block = (MATRIX_SIZE, MATRIX_SIZE, 1), )
  
print("*" * 100)
print("Matrix A:")
print(d_a.get())

print("*" * 100)
print("Matrix B:")
print(d_b.get())

print("*" * 100)
print("Matrix Multiplication result:")
print(d_c_gpu.get())

if (h_c_cpu.all() == d_c_gpu.get().all()) :
    print("\n\nThe computed matrix multiplication is correct")
