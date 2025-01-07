#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: point_line_segments.py
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

import math

def vector(a, b):
    return (b[0] - a[0], b[1] - a[1])

def dot_product(a, b):
    return a[0] * b[0] + a[1] * b[1]

def cross_product(a, b):
    return a[0] * b[1] - a[1] * b[0]

def magnitude(a):
    return math.sqrt(a[0]**2 + a[1]**2)

def point_to_segment_distance(A, B, P):
    AP = vector(A, P)
    AB = vector(A, B)
    cross = abs(cross_product(AP, AB))
    ab_magnitude = magnitude(AB)
    return cross / ab_magnitude

def point_to_segment_projection(A, B, P):
    AP = vector(A, P)
    AB = vector(A, B)
    dot = dot_product(AP, AB)
    ab_squared = AB[0]**2 + AB[1]**2
    lambda_ = dot / ab_squared
    P_prime = (A[0] + lambda_ * AB[0], A[1] + lambda_ * AB[1])
    return P_prime

def determine_position(A, B, P):
    AP = vector(A, P)
    AB = vector(A, B)
    cross = cross_product(AB, AP)

    if cross > 0:
        print("点P在线段AB的左侧")
    elif cross < 0:
        print("点P在线段AB的右侧")
    else:
        print("点P在直线AB上")


# ---------------------------
if __name__ == "__main__":
    # 示例点
    A = (1, 2)
    B = (4, 6)
    P = (3, 4)

    print("点P到线段AB的距离:", point_to_segment_distance(A, B, P))

    P_prime = point_to_segment_projection(A, B, P)
    print("点P到线段AB的投影点:", P_prime)
    
    determine_position(A, B, P)
