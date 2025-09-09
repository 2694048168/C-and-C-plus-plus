/**
 * @file ImageStitch.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-09-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#pragma once

#include "ImageOperator/SymbolExport.h"
#include "opencv2/opencv.hpp"

#include <vector>

namespace Ithaca {

/**
 * @brief 多视角图像全景拼接算法 Image Stitch
 * 
 */
class OPERATOR_EXPORT ImageStitch
{
public:
    static bool Run(const std::vector<cv::Mat> &srcImgVec, cv::Mat &dstImg);

public:
    ImageStitch()  = default;
    ~ImageStitch() = default;
};

bool OPERATOR_EXPORT CALLING_CONVENTIONS image_stitch(const std::vector<cv::Mat> &srcImgVec, cv::Mat &dstImg);

} // namespace Ithaca
