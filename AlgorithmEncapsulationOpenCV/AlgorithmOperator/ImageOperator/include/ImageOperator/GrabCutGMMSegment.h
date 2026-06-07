/**
 * @file GrabCutGMMSegment.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-11-04
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "ImageOperator/SymbolExport.h"
#include "opencv2/opencv.hpp"

namespace Ithaca {

/**
 * @brief GrabCut 交互式图像分割(基于图论的图像分割算法-高斯混合模型GMM)算法实现
 * GrabCut 是一个基于图论的图像分割算法，用于从复杂背景中分离前景(例如叶子、人物、物体等).
 * 基本思想是: 用最小割(Graph Cut)方法在图像像素图上找到最优的前景/背景划分.
 * 1. 背景与前景的建模, GrabCut 认为图像的像素属于两类:
 * --- 前景(foreground)
 * --- 背景(background)
 * 为了区分它们, GrabCut 用 高斯混合模型(GMM)来建模两类像素的颜色分布:
 * ---- 一个 GMM 模型用于前景(通常 K=5 个高斯)
 * ---- 一个 GMM 模型用于背景(K=5 个高斯)
 * 2. 用户提供初始提示: 算法需要用户提供一个 初始矩形区域：
 * ---- 矩形外 ---> 肯定是背景
 * ---- 矩形内 ---> 混合了前景和背景
 * 接下来算法自动学习两类像素的统计特征
 * 3. 迭代优化(EM + Graph Cut)
 * GrabCut 会 交替执行两个步骤：
 * 1. E 步(期望步)：根据当前分割结果, 更新每个像素属于前景/背景的概率分布;
 * 2. M 步(最大化步)：更新 GMM 模型的参数;
 * 3. Graph Cut 步: 利用图割算法, 最小化能量函数(能量函数包含颜色差异项 + 邻域平滑项), 重新划分像素标签;
 * 反复迭代, 直到分割收敛.
 * 4. 最小割(Graph Cut) 图像被视为一个 有权无向图:
 * 1. 每个像素是一个节点;
 * 2. 与邻域像素之间的边表示相似度;
 * 3. 还连接两个超级节点：源点(前景)和汇点(背景);
 * 4. Graph Cut 算法通过最小化代价函数,在图上找到一个 最小割(min-cut), 即最优的前景/背景划分.
 * 
 */

class GrabCutGMMSegment
{
public:
    static bool Run(const cv::Mat &srcImg, cv::Mat &dstImg);

public:
    GrabCutGMMSegment()  = default;
    ~GrabCutGMMSegment() = default;
};

bool OPERATOR_EXPORT CALLING_CONVENTIONS GrabCut_GMM_Segment(const cv::Mat &srcImg, cv::Mat &dstImg, float factor = 1.f,
                                                         int blurWidth = 3, int blurHeight = 3);

} // namespace Ithaca
