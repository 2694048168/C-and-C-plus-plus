#include "ImageOperator/ImageStitch.h"

namespace Ithaca {

bool ImageStitch::Run(const std::vector<cv::Mat> &srcImgVec, cv::Mat &dstImg)
{
    if (srcImgVec.empty())
        return false;

    cv::Ptr<cv::Stitcher> pStitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);
    pStitcher->setPanoConfidenceThresh(0.6);

    cv::Stitcher::Status status = pStitcher->stitch(srcImgVec, dstImg);
    return status == cv::Stitcher::OK ? true : false;
}

bool image_stitch(const std::vector<cv::Mat> &srcImgVec, cv::Mat &dstImg)
{
    if (srcImgVec.empty())
        return false;

    cv::Ptr<cv::Stitcher> pStitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);
    pStitcher->setPanoConfidenceThresh(0.6);

    cv::Stitcher::Status status = pStitcher->stitch(srcImgVec, dstImg);
    return status == cv::Stitcher::OK ? true : false;
}

} // namespace Ithaca
