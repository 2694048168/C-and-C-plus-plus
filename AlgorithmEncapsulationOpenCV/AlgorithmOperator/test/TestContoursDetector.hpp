/**
 * @file TestContoursDetector.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-09-18
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "ImageOperator/ContoursDetector.h"
#include "test/Stopwatch.hpp"

namespace Ithaca {

void TestContoursDetector()
{
    auto sw = Stopwatch{"Ithaca::TestContoursDetector"};

    std::string src_filepath = "D:/Development/Image/nested_circles.png";
    auto        srcImg       = cv::imread(src_filepath, cv::IMREAD_UNCHANGED);

    bool    runState = false;
    cv::Mat dstImg;
    {
        auto                     sw = Stopwatch{"Ithaca::ContoursDetector::Run"};
        Ithaca::ContoursDetector detector;
        runState = detector.Run(srcImg, dstImg);
    }
    if (runState)
        std::cout << "Run Successfully\n";
    else
        std::cout << "Run NOT Successfully\n";

    bool flag = cv::imwrite("./dst_findContours.jpg", dstImg);
    if (flag)
        std::cout << "Save Successfully\n";
    else
        std::cout << "Save NOT Successfully\n";
}

} // namespace Ithaca