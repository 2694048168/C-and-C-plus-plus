/**
 * @file ImageTemplateMatch.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-09-08
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#pragma once

#include "ImageOperator/SymbolExport.h"
#include "opencv2/opencv.hpp"

namespace Ithaca {

/**
 * @brief 图像模板匹配算法 Image Template Match
 * 
 */
class OPERATOR_EXPORT ImageTemplateMatch
{
public:
    static bool Run(const cv::Mat &srcImg, const cv::Mat &templImg, cv::Mat &resultImg);
    static bool RunOpti(const cv::Mat &srcImg, const cv::Mat &templImg, cv::Mat &resultImg);

public:
    ImageTemplateMatch()  = default;
    ~ImageTemplateMatch() = default;
};

bool OPERATOR_EXPORT CALLING_CONVENTIONS image_template_match(const cv::Mat &srcImg, const cv::Mat &templImg,
                                                              cv::Mat &resultImg);

bool OPERATOR_EXPORT CALLING_CONVENTIONS image_template_match_Opti(const cv::Mat &srcImg, const cv::Mat &templImg,
                                                                   cv::Mat &resultImg);

} // namespace Ithaca
