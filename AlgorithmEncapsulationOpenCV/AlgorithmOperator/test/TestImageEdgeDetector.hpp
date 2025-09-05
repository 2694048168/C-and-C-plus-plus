/**
 * @file TestImageEdgeDetector.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-09-05
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "ImageOperator/ImageEdgeDetector.h"
#include "test/Stopwatch.hpp"

namespace Ithaca {

void TestImageEdgeDetector()
{
    auto sw = Stopwatch{"Ithaca::TestImageEdgeDetector"};

    std::string filepath = "D:/Development/SmartUltra/images/1.jpg";

    // auto srcImg = cv::imread(filepath, cv::IMREAD_UNCHANGED);
    auto srcImg = cv::imread(filepath, 0);

    bool    runState = false;
    cv::Mat dstImg;
    {
        auto sw  = Stopwatch{"Ithaca::ImageEdgeDetector::Run"};
        runState = Ithaca::ImageEdgeDetector::Run(srcImg, dstImg);
    }
    if (runState)
        std::cout << "Run Successfully\n";
    else
        std::cout << "Run NOT Successfully\n";

    bool flag = cv::imwrite("./dst_edge_detector.jpg", dstImg);
    if (flag)
        std::cout << "Save Successfully\n";
    else
        std::cout << "Save NOT Successfully\n";

    cv::Mat dstImg1;
    {
        auto sw  = Stopwatch{"Ithaca::image_edge_detector_ExtOpti"};
        runState = Ithaca::image_edge_detector_ExtOpti(srcImg, dstImg1);
    }
    if (runState)
        std::cout << "Run Successfully\n";
    else
        std::cout << "Run NOT Successfully\n";

    flag = cv::imwrite("./dst1_edge_detector.jpg", dstImg1);
    if (flag)
        std::cout << "Save Successfully\n";
    else
        std::cout << "Save NOT Successfully\n";

    cv::Mat dstImg2;
    {
        auto sw  = Stopwatch{"Ithaca::image_edge_detector_Ext"};
        runState = Ithaca::image_edge_detector_Ext(srcImg, dstImg2);
    }
    if (runState)
        std::cout << "Run Successfully\n";
    else
        std::cout << "Run NOT Successfully\n";

    flag = cv::imwrite("./dst2_edge_detector.jpg", dstImg2);
    if (flag)
        std::cout << "Save Successfully\n";
    else
        std::cout << "Save NOT Successfully\n";
}

} // namespace Ithaca
