/**
 * @file TestImageEmphasize.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-09-04
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "ImageOperator/ImageEmphasize.h"
#include "test/Stopwatch.hpp"

#include <iostream>
#include <string>
#include <thread>

namespace Ithaca {
void TestImageEmphasize()
{
    {
        auto sw = Stopwatch("Ithaca::baseline");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    auto sw = Stopwatch{"Ithaca::TestImageEmphasize"};

    std::string filepath = "D:/Development/SmartUltra/images/1.jpg";

    auto srcImg = cv::imread(filepath, cv::IMREAD_UNCHANGED);

    bool    runState = false;
    cv::Mat dstImg;
    {
        auto sw  = Stopwatch{"Ithaca::ImageEmphasize::Run"};
        runState = Ithaca::ImageEmphasize::Run(srcImg, dstImg, 3.0f);
    }
    if (runState)
        std::cout << "Run Successfully\n";
    else
        std::cout << "Run NOT Successfully\n";

    bool flag = cv::imwrite("./dst1.jpg", dstImg);
    if (flag)
        std::cout << "Save Successfully\n";
    else
        std::cout << "Save NOT Successfully\n";

    cv::Mat dstImg1;
    {
        auto sw  = Stopwatch{"Ithaca::image_emphasize"};
        runState = Ithaca::image_emphasize(srcImg, dstImg1, 3.0f);
    }
    if (runState)
        std::cout << "Run Successfully\n";
    else
        std::cout << "Run NOT Successfully\n";

    flag = cv::imwrite("./dst2.jpg", dstImg1);
    if (flag)
        std::cout << "Save Successfully\n";
    else
        std::cout << "Save NOT Successfully\n";
}

} // namespace Ithaca
