/**
 * @file ImageEmphasize.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief Halcon 图像增强算子-emphasize 的 OpenCV & Modern C++实现
 * @version 0.1
 * @date 2025-08-31
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "ImageOperator/SymbolExport.h"
#include "opencv2/opencv.hpp"

namespace Ithaca {

/**
 * @brief 图像增强算子 ImageEmphasize
 * 
 */
class OPERATOR_EXPORT ImageEmphasize
{
public:
    static bool Run(const cv::Mat &srcImg, cv::Mat &dstImg, float factor = 1.f, int blurWidth = 3, int blurHeight = 3);

public:
    ImageEmphasize()  = default;
    ~ImageEmphasize() = default;
};

bool OPERATOR_EXPORT CALLING_CONVENTIONS image_emphasize(const cv::Mat &srcImg, cv::Mat &dstImg, float factor = 1.f,
                                                         int blurWidth = 3, int blurHeight = 3);

} // namespace Ithaca