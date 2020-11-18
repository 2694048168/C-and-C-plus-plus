#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: weili
# @filename: 11_03_thread_execution.py
# @copyright: https://gitee.com/weili_yzzcq/C-and-C-plus-plus/CUDA_CPlusPlus/
# @copyright: https://github.com/2694048168/C-and-C-plus-plus/CUDA_CPlusPlus/
# @function: 使用 Python 利用 PyCUDA 执行 GPU 线程和块。

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
    printf("I am in block no: %d \\n", blockIdx.x);
  }
""")

function = mod.get_function("first_kernel")
function(grid=(10,1),block=(1,1,1))