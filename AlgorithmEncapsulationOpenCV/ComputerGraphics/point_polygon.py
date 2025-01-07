#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: point_polygon.py
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

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def cross(A, B, C):
    # 计算叉积
    return (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x)

def is_point_in_polygon(polygon, P):
    wn = 0  # 交点数
    n = len(polygon)
    for i in range(n):
        A = polygon[i]
        B = polygon[(i + 1) % n]
        if A.y <= P.y and B.y > P.y and cross(B, P, A) > 0:
            wn += 1
        elif A.y > P.y and B.y <= P.y and cross(B, P, A) < 0:
            wn -= 1
    return wn != 0  # 如果交点数不为0，则点在多边形内部


def winding_number(polygon, P):
    wn = 0
    for i in range(len(polygon)):
        A = polygon[i]
        B = polygon[(i + 1) % len(polygon)]
        if A.y <= P.y and B.y > P.y:
            if cross(P, A, B) > 0:
                wn += 1
        elif A.y > P.y and B.y <= P.y:
            if cross(P, A, B) < 0:
                wn -= 1
    return wn


def is_point_in_polygon_winding(polygon, P):
    return winding_number(polygon, P) != 0


# ---------------------------
if __name__ == "__main__":
    # 多边形顶点坐标
    polygon = [Point(0, 0), Point(2, 0), Point(1, 2)]
    # 待判断点的坐标
    P = Point(1, 1)
    # 判断点与多边形的位置关系
    if is_point_in_polygon(polygon, P):
        print("Point P is inside the polygon.")
    else:
        print("Point P is outside the polygon.")


    print("----------------------------------------")
    polygon = [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]
    P = Point(0.5, 1.5)
    if is_point_in_polygon(polygon, P):
        print("Point is inside the polygon.")
    else:
        print("Point is outside the polygon.")
