#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: SplitCount.py
@Python Version: 3.12.1
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2025-07-22
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V2.1
@License: Apache License Version 2.0, January 2004
    Copyright 2025. All rights reserved.

@Description: 
'''

# import cv2
# import numpy as np


# # 基于分水岭算法
# def SplitCount_WatershedAlgorithm(image):
#     # 1: 高斯滤波 + 二值化 + 开运算
#     # if 3 == image.ndim:
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)
#     ret, binary = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY)
    
#     kernel = np.ones((5, 5), np.uint8)
#     binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

#     # cv2.imshow('threshold_image', binary)
#     # cv2.waitKey()

#     # 2: 距离变换 + 提取前景
#     dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
#     dist_out = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)

#     # cv2.imshow('distance-Transform', dist_out * 100)
#     # cv2.waitKey()

#     ret, surface = cv2.threshold(dist_out, 0.35*dist_out.max(), 255, cv2.THRESH_BINARY)

#     # cv2.imshow('surface', surface)
#     # cv2.waitKey()

#     sure_fg = np.uint8(surface) # 转成8位整型

#     # cv2.imshow('Sure foreground', sure_fg)
#     # cv2.waitKey()

#     # 3: 标记位置区域
#     kernel = np.ones((5, 5), np.uint8)
#     binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=1)
#     unknown = binary - sure_fg
    
#     cv2.imshow('unknown',unknown)
#     cv2.waitKey()

#     # 创建标记图
#     _, markers = cv2.connectedComponents(sure_fg)
#     # 为分水岭算法准备：未知区域标记为0
#     markers = markers + 1
#     markers[unknown == 255] = 0
    
#     # 4: 分水岭算法分割
#     markers = cv2.watershed(gray, markers=markers)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(markers)

#     # 5: 轮廓查找和标记
#     contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     for cnt in contours:
#         M = cv2.moments(cnt)
#         cx = int(M['m10']/M['m00'])
#         cx = int(M['m10']/M['m00'])
#         cy = int(M['m01']/M['m00']) # 轮廓重心
#         cv2.drawContours(img, contours, -1, colors[rd.randint(0,5)], 2)
#         cv2.drawMarker(img, (cx, cy),(0, 255, 0), 1, 8, 2)

#     return img



# ----------------------------------------
import cv2
import numpy as np
from matplotlib import pyplot as plt

def process_image(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图像，请检查路径")
        return
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 预处理：高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 二值化：Otsu自动阈值
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 形态学操作：去除小噪点
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # 距离变换
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # 获取前景区域
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # 确定未知区域（边界）
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # 创建标记图
    _, markers = cv2.connectedComponents(sure_fg)
    
    # 为分水岭算法准备：未知区域标记为0
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # 应用分水岭算法
    markers = cv2.watershed(img, markers)
    
    # 在原始图像上绘制边界（-1表示边界）
    img[markers == -1] = [0, 0, 255]  # 用红色标记边界
    
    # 统计大米数量（排除背景标记0和边界-1）
    unique_markers = np.unique(markers)
    rice_count = len(unique_markers) - 2  # 减去背景(1)和边界(-1)，但注意背景标记从1开始
    
    # 显示结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(231), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), 
    plt.title(f'最终结果 (计数: {rice_count})'), plt.axis('off')
    
    plt.subplot(232), plt.imshow(thresh, 'gray'), 
    plt.title('二值化结果'), plt.axis('off')
    
    plt.subplot(233), plt.imshow(dist_transform, 'gray'), 
    plt.title('距离变换'), plt.axis('off')
    
    plt.subplot(234), plt.imshow(sure_fg, 'gray'), 
    plt.title('确定前景'), plt.axis('off')
    
    plt.subplot(235), plt.imshow(unknown, 'gray'), 
    plt.title('未知区域'), plt.axis('off')
    
    plt.subplot(236), plt.imshow(markers, cmap='jet'), 
    plt.title('分水岭标记'), plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return rice_count

# 轮廓凸包缺陷方法: 基于轮廓凸包缺陷分割步骤
def ConvexHull_DefectSegmentation(image):
    # Step1. 高斯滤波 + 二值化 + 开运算
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, binary = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    # cv2.imshow('threshold_image', binary)
    # cv2.waitKey()

    # Step2. 轮廓遍历 + 筛选轮廓含有凸包缺陷的轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        hull = cv2.convexHull(cnt,returnPoints=False) # 默认 return Points=True
        defects = cv2.convexityDefects(cnt,hull)
    #print defects
    pt_list = []
    if defects is not None:
            flag = False
    for i in range(0,defects.shape[0]):
                s,e,f,d = defects[i,0]
    if d > 4500:
        flag = True

    # Step3. 将距离d最大的两个凸包缺陷点连起来，将二值图中对应的粘连区域分割开，红色圆标注为分割开的部分
    if len(pt_list) > 0:
        cv2.line(binary, pt_list[0], pt_list[1],0,2)  
    # cv2.imshow('binary2', binary)
    # cv2.waitKey()

    # Step4. 重新查找轮廓并标记结果
    contours,hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        try:
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])#轮廓重心

            cv2.drawContours(img,cnt,-1,colors[rd.randint(0,5)],2)
            cv2.drawMarker(img, (cx,cy),(0,0,255),1,8,2)
        except:
            pass

    cv2.imshow('render-image',img)
    cv2.waitKey()

# --------------------------
if __name__ == "__main__":
    # img = cv2.imread("../images/StickyRice.png")
    # SplitCount_WatershedAlgorithm(image=img)
    # cv2.imwrite("../images/StickyRice_defect.png", img)

    # ---------------------------------------
    # image_path = "../images/StickyRice.png"
    # count = process_image(image_path)
    # print(f"大米计数结果: {count}")

    # ---------------------------------------
    img = cv2.imread("../images/StickyRice.png")
    ConvexHull_DefectSegmentation(image=img)

