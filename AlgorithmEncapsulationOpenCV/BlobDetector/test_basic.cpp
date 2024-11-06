/**
 * @file test_basic.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-11-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "BlobDetector.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>

// -----------------------------------
int main(int argc, const char **argv)
{
    std::cout << "blob detection" << std::endl;
    cv::SimpleBlobDetector::Params blob_params;
    blob_params.blobColor           = 255;
    blob_params.maxArea             = 500;
    blob_params.minArea             = 100;
    blob_params.minThreshold        = 130;
    blob_params.maxThreshold        = 170;
    blob_params.thresholdStep       = 10;
    blob_params.filterByCircularity = false;
    blob_params.filterByConvexity   = false;
    blob_params.filterByInertia     = false;

    auto blob_detect = SmartUltra::BlobDetectorModule::BlobDetect::CreateInstance(blob_params);

    cv::Mat img = cv::imread("images/blob.jpg", 0);
    if (img.empty())
        return -1;

    blob_detect->Init(img);
    // blob_detect->SetParams();
    int s = cv::getTickCount();
    blob_detect->Run();
    int e = cv::getTickCount();
    std::cout << "BlobDetect cost time: " << static_cast<double>(e - s) / cv::getTickFrequency() * 1000 << "ms"
              << std::endl;
    blob_detect->PrintResultInfo();

    blob_detect->PrintParameter();
    blob_detect->DrawOutline();
    blob_detect.release();

    cv::Ptr<cv::SimpleBlobDetector> s_blob_detect = cv::SimpleBlobDetector::create();
    std::vector<cv::KeyPoint>       kps;
    s = cv::getTickCount();
    s_blob_detect->detect(img, kps);
    e = cv::getTickCount();
    std::cout << "SimpleBlobDetector cost time: " << static_cast<double>(e - s) / cv::getTickFrequency() * 1000 << "ms"
              << std::endl;
    std::cout << "keypoints size: " << kps.size() << std::endl;
    for (int i = 0; i < kps.size(); ++i)
    {
        std::cout << i + 1 << ": location:" << kps[i].pt << std::endl;
    }
    cv::Mat im_with_keypoints;
    cv::drawKeypoints(img, kps, im_with_keypoints, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imshow("keypoints", im_with_keypoints);
    cv::waitKey(0);
    s_blob_detect.release();

    return 0;
}
