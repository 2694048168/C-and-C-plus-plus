#include "ImageOperator/ImageEmphasize.h"

namespace Ithaca {

bool ImageEmphasize::Run(const cv::Mat &srcImg, cv::Mat &dstImg, float factor, int blurWidth, int blurHeight)
{
    if (srcImg.empty())
        return false;

    cv::Mat meanImg;
    cv::blur(srcImg, meanImg, cv::Size(blurWidth, blurHeight));
    cv::Mat srcFloat, meanFloat;
    srcImg.convertTo(srcFloat, CV_32F);
    meanImg.convertTo(meanFloat, CV_32F);
    cv::Mat dstFloat = (srcFloat - meanFloat) * factor + srcFloat;
    dstFloat.convertTo(dstImg, srcImg.type());

    return true;
}

bool image_emphasize(const cv::Mat &srcImg, cv::Mat &dstImg, float factor, int blurWidth, int blurHeight)
{
    if (srcImg.empty())
        return false;

    cv::Mat meanImg;
    cv::blur(srcImg, meanImg, cv::Size(blurWidth, blurHeight));
    cv::Mat srcFloat, meanFloat;
    srcImg.convertTo(srcFloat, CV_32F);
    meanImg.convertTo(meanFloat, CV_32F);
    cv::Mat dstFloat = (srcFloat - meanFloat) * factor + srcFloat;
    dstFloat.convertTo(dstImg, srcImg.type());

    return true;
}

} // namespace Ithaca
