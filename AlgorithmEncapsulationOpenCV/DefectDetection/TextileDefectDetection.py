#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: TextileDefectDetection.py
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

import cv2
import numpy as np

# --------------------------
if __name__ == "__main__":
    # -------------------------------------------------------
    # 脏污缺陷图片, 肉眼可见明显的几处脏污
    img = cv2.imread("../images/smudge.png")
    # 【1】使用高斯滤波消除背景纹理的干扰, 将原图放大后会发现纺织物自带的纹理比较明显,
    # 这会影响后续处理结果, 所以先做滤波平滑
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    cv2.imwrite("../images/smudge_blur.png", blur)
    # 【2】Canny边缘检测凸显缺陷,
    # Canny边缘检测对低对比度缺陷检测有很好的效果, 这里注意高低阈值的设置
    edged = cv2.Canny(blur, 10, 30)
    cv2.imwrite("../images/smudge_edged.png", edged)
    # 【3】轮廓查找、筛选与结果标记,
    # 轮廓筛选可以根据面积、长度过滤掉部分干扰轮廓, 找到真正的缺陷
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        length = cv2.arcLength(cnt,True)
        if length >= 1:
            cv2.drawContours(img, cnt, -1, (0, 0, 255), 2)
    cv2.imwrite("../images/smudge_defect.png", img)

    # -------------------------------------------------------
    # 油污缺陷图片, 肉眼可见明显的两处油污
    img = cv2.imread("../images/oil_stain.png")
    # 【1】将图像从RGB颜色空间转到Lab颜色空间,
    # 对于类似油污和一些亮团的情况,将其转换到Lab或YUV等颜色空间的色彩通道常常能更好的凸显其轮廓
    LabImg = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L, A, B = cv2.split(LabImg)
    cv2.imwrite("../images/oil_stain_B_channel.png", B)
    # 【2】高斯滤波 + 二值化
    blur = cv2.GaussianBlur(B, (3, 3), 0)
    ret, thresh = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY)
    cv2.imwrite("../images/oil_stain_thresh.png", thresh)
    # 【3】形态学开运算滤除杂讯
    k1 = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k1)
    cv2.imwrite("../images/oil_stain_thresh_mask.png", thresh)
    # 【4】轮廓查找、筛选与结果标记,
    # 轮廓筛选可以根据面积、宽高过滤掉部分干扰轮廓, 找到真正的缺陷
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= 50:
            cv2.drawContours(img, cnt, -1, (0, 0, 255), 2)
    cv2.imwrite("../images/oil_stain_defect.png", img)

    # -------------------------------------------------------
    # 线条破损缺陷图片, 肉眼可见明显的一处脏污
    img = cv2.imread("../images/Line_damage.png")
    # 【1】将图像从RGB颜色空间转到Lab颜色空间 + 高斯滤波
    LabImg = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L, A, B = cv2.split(LabImg)
    blur = cv2.GaussianBlur(B, (3, 3), 0)
    cv2.imwrite("../images/Line_damage_B_blur.png", B)
    # 【2】Canny边缘检测凸显缺陷
    edged = cv2.Canny(blur, 5, 10)
    cv2.imwrite("../images/Line_damage_edged.png", edged)
    # 【3】轮廓查找、筛选与结果标记,
    # 轮廓筛选可以根据面积、长度过滤掉部分干扰轮廓, 找到真正的缺陷
    contours,hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        length = cv2.arcLength(cnt,True)
        if length >= 10:
            cv2.drawContours(img, cnt, -1, (0, 0, 255), 2)
    cv2.imwrite("../images/Line_damage_defect.png", img)
