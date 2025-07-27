#pragma once

#include "BlobToolParams.h"

#include <vector>

class BlobTool
{
public:
    BlobTool();
    BlobTool(const BlobParams &params);

    void setParams(const BlobParams &newParams);

    std::vector<cv::KeyPoint> detect(const cv::Mat &image);

private:
    void updateDetector();

    BlobParams                      params;
    cv::Ptr<cv::SimpleBlobDetector> detector;
};
