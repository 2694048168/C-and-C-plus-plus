/**
 * @file TestImageStitch.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-09-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "ImageOperator/ImageStitch.h"
#include "test/Stopwatch.hpp"

#include <vector>

namespace Ithaca {

void TestImageStitch()
{
    auto sw = Stopwatch{"Ithaca::TestImageStitch"};

    std::string fileFolder = "D:/Development/Image/boat/";
    // std::string fileFolder = "D:/Development/Image/stitch/";

    std::vector<std::string> fileNames;
    cv::glob(fileFolder, fileNames);
    std::vector<cv::Mat> imgVec;
    for (auto filepath : fileNames)
    {
        auto img = cv::imread(filepath);
        if (img.empty())
            continue;
        imgVec.emplace_back(img);
    }

    bool    runState = false;
    cv::Mat dstImg;
    {
        auto sw  = Stopwatch{"Ithaca::ImageStitch::Run"};
        runState = Ithaca::ImageStitch::Run(imgVec, dstImg);
    }
    if (runState)
        std::cout << "Run Successfully\n";
    else
        std::cout << "Run NOT Successfully\n";

    bool flag = cv::imwrite("./dst_stitch.jpg", dstImg);
    if (flag)
        std::cout << "Save Successfully\n";
    else
        std::cout << "Save NOT Successfully\n";
}

} // namespace Ithaca