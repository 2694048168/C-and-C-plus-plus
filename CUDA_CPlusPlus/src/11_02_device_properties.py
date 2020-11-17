#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: weili
# @filename: 11_02_device_properties.py
# @copyright: https://gitee.com/weili_yzzcq/C-and-C-plus-plus/CUDA_CPlusPlus/
# @copyright: https://github.com/2694048168/C-and-C-plus-plus/CUDA_CPlusPlus/
# @function: 使用 Python 利用 PyCUDA 获取 GPU 设备属性。

# driver 包含内存管理功能、设备属性、数据方向功能等等。
import pycuda.driver as drv
# autoinit 用于设备初始化、上下文创建和内存清理，可以手动完成，不需要该模块。
import pycuda.autoinit

# 设备初始化。
drv.init()
print("%d device(s) found." % drv.Device.count())
for i in range(drv.Device.count()):
    dev = drv.Device(i)
    print("Device #%d: %s" % (i, dev.name()))
    print("  Compute Capability: %d.%d" % dev.compute_capability())
    print("  Total Memory: %s GB" % (dev.total_memory()//(1024*1024*1024)))
    attributes = [(str(prop), value) 
            for prop, value in list(dev.get_attributes().items())]
    attributes.sort()
    n=0
    for prop, value in attributes:
        print("  %s: %s " % (prop, value),end=" ")
        n = n+1
        if(n%2 == 0):
            print(" ")