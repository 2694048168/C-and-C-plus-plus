#include "BlobTool.h"

BlobTool::BlobTool(const BlobParams &params)
    : params(params)
{
    updateDetector();
}

BlobTool::BlobTool()
{
    updateDetector();
}

void BlobTool::setParams(const BlobParams &newParams)
{
    params = newParams;
    updateDetector();
}

std::vector<cv::KeyPoint> BlobTool::detect(const cv::Mat &image)
{
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(image, keypoints);
    return keypoints;
}

void BlobTool::updateDetector()
{
    cv::SimpleBlobDetector::Params detectorParams;

    detectorParams.minThreshold        = params.minThreshold;
    detectorParams.maxThreshold        = params.maxThreshold;
    detectorParams.filterByArea        = true;
    detectorParams.minArea             = params.minArea;
    detectorParams.maxArea             = params.maxArea;
    detectorParams.filterByColor       = params.filterByColor;
    detectorParams.blobColor           = params.blobColor;
    detectorParams.filterByCircularity = params.filterByCircularity;
    detectorParams.minCircularity      = params.minCircularity;
    detectorParams.filterByConvexity   = params.filterByConvexity;
    detectorParams.minConvexity        = params.minConvexity;
    detectorParams.filterByInertia     = params.filterByInertia;
    detectorParams.minInertiaRatio     = params.minInertiaRatio;

    detector = cv::SimpleBlobDetector::create(detectorParams);
}
