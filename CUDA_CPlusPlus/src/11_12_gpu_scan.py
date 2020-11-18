#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: weili
# @filename: 11_12_gpu_scan.py
# @copyright: https://gitee.com/weili_yzzcq/C-and-C-plus-plus/CUDA_CPlusPlus/
# @copyright: https://github.com/2694048168/C-and-C-plus-plus/CUDA_CPlusPlus/
# @function: pycuda 高级内核函数之 scan 内核函数。

import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy
from pycuda.scan import InclusiveScanKernel
import pycuda.autoinit

n = 10
start = drv.Event()
end = drv.Event()
start.record()

kernel = InclusiveScanKernel(numpy.uint32,"a+b")

h_a = numpy.random.randint(1,10,n).astype(numpy.int32)
d_a = gpuarray.to_gpu(h_a)
kernel(d_a)

end.record()
end.synchronize()
secs = start.time_till(end) * 1e-3

assert(d_a.get() == numpy.cumsum(h_a,axis=0)).all()
print("The input data:")
print(h_a)

print("The computed cumulative sum using Scan:")
print(d_a.get())

print("Cumulative Sum on GPU")
print("%fs" % (secs))
