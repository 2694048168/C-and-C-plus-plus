#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: weili
# @filename: 11_10_element_wise_addition.py
# @copyright: https://gitee.com/weili_yzzcq/C-and-C-plus-plus/CUDA_CPlusPlus/
# @copyright: https://github.com/2694048168/C-and-C-plus-plus/CUDA_CPlusPlus/
# @function: PyCUDA 的高级内核函数之 PyCDUA 的元素级内核函数。

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel
from pycuda.curandom import rand as curand

add = ElementwiseKernel(
        "float *d_a, float *d_b, float *d_c",
        "d_c[i] = d_a[i] + d_b[i]",
        "add")

# create a couple of random matrices with a given shape
shape = 1000000
d_a = curand(shape)
d_b = curand(shape)

d_c = gpuarray.empty_like(d_a)

start = drv.Event()
end=drv.Event()
start.record()

add(d_a, d_b, d_c)

end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Addition of %d element of GPU"%shape)
print("%fs" % (secs))

# check the result
if d_c == (d_a + d_b):
    print("The sum computed on GPU is correct")
