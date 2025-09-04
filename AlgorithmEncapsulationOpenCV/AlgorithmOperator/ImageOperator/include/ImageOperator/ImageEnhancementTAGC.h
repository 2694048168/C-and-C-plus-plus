/**
 * @file ImageEnhancementTAGC.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief TAGC Image Enhancement algorithm
 * @version 0.1
 * @date 2025-09-04
 * 
 * @copyright Copyright (c) 2025
 * 
 * @paper name: Tuning adaptive gamma correction (TAGC) for enhancing images in low ligh 
 * @paper link: https://www.arxiv.org/abs/2507.19574
 * 
 */

#include "ImageOperator/SymbolExport.h"
#include "opencv2/opencv.hpp"

namespace Ithaca {

/**
 * @brief 低照度图像增强算法-TAGC Image Enhancement
 * 
 */
class OPERATOR_EXPORT ImageEnhancementTAGC
{
public:
    static bool Run(const cv::Mat &srcImg, cv::Mat &dstImg);

public:
    ImageEnhancementTAGC()  = default;
    ~ImageEnhancementTAGC() = default;
};

bool OPERATOR_EXPORT CALLING_CONVENTIONS image_enhancement_TAGC(const cv::Mat &srcImg, cv::Mat &dstImg);

} // namespace Ithaca
