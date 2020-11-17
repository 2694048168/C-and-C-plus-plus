#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: weili
# @filename: 11_01_hello_pycdua.py
# @copyright: https://gitee.com/weili_yzzcq/C-and-C-plus-plus/CUDA_CPlusPlus/
# @copyright: https://github.com/2694048168/C-and-C-plus-plus/CUDA_CPlusPlus/
# @function: pip install pycuda 使用 Python 利用 CDUA 进行 GPU 加速
# @function: pip install pyopencl 使用 Python 利用 OpenCL 进行 GPU 加速

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

# C/C++ 内核代码作为构造函数被传入 SourceModule 类中并实例化对象 mod。
# 该内核函数由 nvcc 编译器进行编译。
mod = SourceModule("""
  #include <stdio.h>

  __global__ void first_kernel()
  {
    printf("Hello,PyCUDA.");
  }
""")

# SourceModule 类实例化对象 mod 的 get_function 方法创建指向函数的指针。
function = mod.get_function("first_kernel")

# Python 中内核调用，简单语法：
# kernel (parameters for kernel, block=(tx, ty, tz), grid=(bx, by, bz))
# parameters for kernel 表示内核参数；
# block=(tx, ty, tz) 表示启动的每个块里面的线程数；
# grid=(bx, by, bz) 表示启动的块的数；
# 元组 表示三维中的线程块和线程，内核启动的总线程是这两个数字的乘积。
function(block=(1,1,1))