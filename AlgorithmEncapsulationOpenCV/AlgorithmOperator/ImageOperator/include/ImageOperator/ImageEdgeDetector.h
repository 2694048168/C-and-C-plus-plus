/**
 * @file ImageEdgeDetector.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief image edge detector operator
 * @version 0.1
 * @date 2025-09-05
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "ImageOperator/SymbolExport.h"
#include "opencv2/opencv.hpp"

namespace Ithaca {

/**
 * @brief 图像边缘检测算法-Sobel
 * 
 */
class OPERATOR_EXPORT ImageEdgeDetector
{
public:
    static bool Run(const cv::Mat &srcImg, cv::Mat &dstImg);
    static bool RunExt(const cv::Mat &srcImg, cv::Mat &dstImg);
    static bool RunExtOpti(const cv::Mat &srcImg, cv::Mat &dstImg);

public:
    ImageEdgeDetector()  = default;
    ~ImageEdgeDetector() = default;
};

bool OPERATOR_EXPORT CALLING_CONVENTIONS image_edge_detector(const cv::Mat &srcImg, cv::Mat &dstImg);
bool OPERATOR_EXPORT CALLING_CONVENTIONS image_edge_detector_Ext(const cv::Mat &srcImg, cv::Mat &dstImg);
bool OPERATOR_EXPORT CALLING_CONVENTIONS image_edge_detector_ExtOpti(const cv::Mat &srcImg, cv::Mat &dstImg);

} // namespace Ithaca
