/**
 * @file ContoursDetector.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 轮廓检测
 * @version 0.1
 * @date 2025-09-18
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#pragma once

#include "ImageOperator/SymbolExport.h"
#include "opencv2/opencv.hpp"

namespace Ithaca {

struct ContoursDetectorParameter
{
    int    blurKernelX  = 5;
    int    blurKernelY  = 5;
    double minThreshold = 127.0;
    double maxThreshold = 255.0;
};

class OPERATOR_EXPORT ContoursDetector
{
public:
    bool Run(const cv::Mat &srcImg, cv::Mat &dstImg);

public:
    ContoursDetectorParameter mControlParameter;

public:
    ContoursDetector()  = default;
    ~ContoursDetector() = default;
};

} // namespace Ithaca
