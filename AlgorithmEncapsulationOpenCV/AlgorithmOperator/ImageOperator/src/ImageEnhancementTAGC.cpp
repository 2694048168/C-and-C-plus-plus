#include "ImageOperator/ImageEnhancementTAGC.h"

namespace Ithaca {

bool ImageEnhancementTAGC::Run(const cv::Mat &srcImg, cv::Mat &dstImg)
{
    // CV_Assert(srcImg.type() == CV_8UC3);
    if (CV_8UC3 != srcImg.type())
        return false;

    // 转换为浮点数 并归一化到 [0, 1]
    cv::Mat srcImgFloat;
    srcImg.convertTo(srcImgFloat, CV_32FC3, 1.0 / 255.0);
    dstImg.create(srcImg.size(), srcImg.type());

    for (int y{0}; y < srcImgFloat.rows; ++y)
    {
        const float *pSrcLine = srcImgFloat.ptr<float>(y);
        uchar       *pDstLine = dstImg.ptr<uchar>(y);

        for (int x{0}; x < srcImgFloat.cols; ++x)
        {
            float blue  = pSrcLine[0];
            float green = pSrcLine[1];
            float red   = pSrcLine[2];

            float L     = 0.2126f * red + 0.7152f * green + 0.0722f * blue;
            float A     = (blue + green + red) / 3.f;
            float Gamma = 5.0f + (0.5f - L) * (1.0f - A) - 2.0f * L;

            // 从归一化转化为 8bit 像素, 并自动截断
            pDstLine[0] = cvRound(std::pow(blue, 2.0f / Gamma) * 255.0f);
            pDstLine[1] = cvRound(std::pow(green, 2.0f / Gamma) * 255.0f);
            pDstLine[2] = cvRound(std::pow(red, 2.0f / Gamma) * 255.0f);

            // pointer offset address
            pSrcLine += 3;
            pDstLine += 3;
        }
    }

    return true;
}

bool image_enhancement_TAGC(const cv::Mat &srcImg, cv::Mat &dstImg)
{
    // CV_Assert(srcImg.type() == CV_8UC3);
    if (CV_8UC3 != srcImg.type())
        return false;

    // 转换为浮点数 并归一化到 [0, 1]
    cv::Mat srcImgFloat;
    srcImg.convertTo(srcImgFloat, CV_32FC3, 1.0 / 255.0);
    dstImg.create(srcImg.size(), srcImg.type());

    for (int y{0}; y < srcImgFloat.rows; ++y)
    {
        const float *pSrcLine = srcImgFloat.ptr<float>(y);
        uchar       *pDstLine = dstImg.ptr<uchar>(y);

        for (int x{0}; x < srcImgFloat.cols; ++x)
        {
            float blue  = pSrcLine[0];
            float green = pSrcLine[1];
            float red   = pSrcLine[2];

            float L     = 0.2126f * red + 0.7152f * green + 0.0722f * blue;
            float A     = (blue + green + red) / 3.f;
            float Gamma = 5.0f + (0.5f - L) * (1.0f - A) - 2.0f * L;

            // 从归一化转化为 8bit 像素, 并自动截断
            pDstLine[0] = cvRound(std::pow(blue, 2.0f / Gamma) * 255.0f);
            pDstLine[1] = cvRound(std::pow(green, 2.0f / Gamma) * 255.0f);
            pDstLine[2] = cvRound(std::pow(red, 2.0f / Gamma) * 255.0f);

            // pointer offset address
            pSrcLine += 3;
            pDstLine += 3;
        }
    }

    return true;
}

} // namespace Ithaca
