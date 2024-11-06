/**
 * @file BlobDetector.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Blob 检测器 
 * @details 基于OpenCV SimpleBlobDetector拓展的 blob 检测算子
 * @version 0.1
 * @date 2024-11-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include "opencv2/features2d.hpp"

#include <vector>

namespace SmartUltra { namespace BlobDetectorModule {

// Single blob info
struct BlobInfo
{
    cv::Point2d              location;   //center point's coordinate
    std::vector<cv::Point2d> outline;    //outline
    double                   area;       //area
    double                   radius;     //radius
    double                   confidence; //blob detection confidence
};

struct BlobDetectResult
{
    std::vector<BlobInfo> blobList;
};

class BlobDetect : protected cv::SimpleBlobDetector
{
public:
    explicit BlobDetect(const cv::SimpleBlobDetector::Params &parameters = cv::SimpleBlobDetector::Params());

    static cv::Ptr<BlobDetect> CreateInstance(const cv::SimpleBlobDetector::Params &parameters
                                              = cv::SimpleBlobDetector::Params());

    // Workflow
    // 1.
    void Init(cv::Mat &inputImage); // input image

    // 2.
    // CV_WRAP virtual void setParams(const SimpleBlobDetector::Params& params ) = 0;
    // CV_WRAP virtual SimpleBlobDetector::Params getParams() const = 0;
    void setParams(const SimpleBlobDetector::Params &params) override;

    cv::SimpleBlobDetector::Params getParams() const override;

    void SetParams();                              // default value
    void SetParams(std::string name, float value); // set value by param name
    // 3. detection
    // ....

    void Run(); // run detection processing

    // --------------debug---------------------------
    void PrintResultInfo() const; // print result information of detection
    void PrintParameter() const;
    void DrawOutline() const;

    // class members for user
    cv::Mat          m_inputImage;
    BlobDetectResult m_resultVec;

protected:
private:
    void FindBlobs(cv::InputArray image, cv::InputArray binaryImage, std::vector<BlobInfo> &centers) const;
    void Detect(cv::InputArray image, cv::InputArray mask = cv::noArray());

    std::vector<cv::KeyPoint>      m_keyPoints;
    cv::SimpleBlobDetector::Params m_params;
};

}} // namespace SmartUltra::BlobDetectorModule
