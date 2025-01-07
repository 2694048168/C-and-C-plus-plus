#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: intersection_line_segments.py
@Python Version: 3.12.1
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2025-01-07.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V2.1
@License: Apache License Version 2.0, January 2004
    Copyright 2025. All rights reserved.

@Description: 
'''

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def line_intersection(A, B, C, D):
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    x4, y4 = D

    # 计算分母
    denominator = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    if denominator == 0:
        return None  # 线段平行或重合

    # 计算t和s
    t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / denominator
    s = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / denominator

    # 检查t和s是否在[0, 1]区间内
    if 0 <= t <= 1 and 0 <= s <= 1:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    else:
        return None  # 线段不相交


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    point.set_data([], [])
    text.set_text('')
    return line1, line2, point, text


def animate(i):
    A, B, C, D = segments[i]
    intersection = line_intersection(A, B, C, D)

    line1.set_data([A[0], B[0]], [A[1], B[1]])
    line2.set_data([C[0], D[0]], [C[1], D[1]])

    if intersection:
        line1.set_color('r')
        line2.set_color('r')
        point.set_data([intersection[0]], [intersection[1]])
        text.set_text(f'Intersection: ({intersection[0]:.2f}, {intersection[1]:.2f})')
    else:
        line1.set_color('b')
        line2.set_color('b')
        point.set_data([], [])  # 修改这里
        text.set_text('No Intersection')

    return line1, line2, point, text


# ---------------------------
if __name__ == "__main__":
    # 手动定义线段
    segments = [
        ((0.3, 0.3), (0.8, 0.8), (0.2, 0.8), (0.8, 0.2)),
        ((0.2, 0.2), (0.8, 0.2), (0.5, 0.5), (0.8, 0.5)),
        ((0.1, 0.1), (0.7, 0.7), (0.6, 0.22), (0.8, 0.2)),
        ((0.3, 0.3), (0.7, 0.7), (0.4, 0.84), (0.8, 0.2)),
        ((0.2, 0.5), (0.8, 0.4), (0.6, 0.8), (0.7, 0.2)),
        ((0.2, 0.3), (0.8, 0.8), (0.5, 0.2), (0.8, 0.2))
    ]


    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # 初始化绘图
    line1, = ax.plot([], [], 'b-', animated=True)
    line2, = ax.plot([], [], 'b-', animated=True)
    point, = ax.plot([], [], 'ro', animated=True)
    text = ax.text(0.5, 0.9, '', transform=ax.transAxes, ha='center')

    ani = FuncAnimation(fig, animate, init_func=init, frames=len(segments), interval=500, blit=True)

    # 保存动图
    ani.save('images/line_segments.gif', writer='pillow', fps=1)
    plt.show()
