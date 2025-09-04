/**
 * @file TestImageEnhancementTAGC.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-09-04
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "ImageOperator/ImageEnhancementTAGC.h"
#include "test/Stopwatch.hpp"

#include <filesystem>
#include <vector>

namespace Ithaca {

void TestImageEnhancementTAGC()
{
    auto sw = Stopwatch{"Ithaca::TestImageEnhancementTAGC"};

    std::string filepath = "D:/ImageProcessingTeam/ChinaMM2021/eval15/low/test/input/179.png";

    auto srcImg = cv::imread(filepath, cv::IMREAD_UNCHANGED);

    bool    runState = false;
    cv::Mat dstImg;
    {
        auto sw  = Stopwatch{"Ithaca::ImageEnhancementTAGC::Run"};
        runState = Ithaca::ImageEnhancementTAGC::Run(srcImg, dstImg);
    }
    if (runState)
        std::cout << "Run Successfully\n";
    else
        std::cout << "Run NOT Successfully\n";

    bool flag = cv::imwrite("./dst_enhancement.jpg", dstImg);
    if (flag)
        std::cout << "Save Successfully\n";
    else
        std::cout << "Save NOT Successfully\n";
}

void TestImageEnhancementTAGCFolder()
{
    auto sw = Stopwatch{"Ithaca::TestImageEnhancementTAGCFolder"};

    std::string              imageFolder = "D:/ImageProcessingTeam/ChinaMM2021/eval15/low/test/input/";
    std::vector<std::string> imgFilepathVec;
    cv::glob(imageFolder, imgFilepathVec);

    for (const auto &filepath : imgFilepathVec)
    {
        auto srcImg = cv::imread(filepath, cv::IMREAD_UNCHANGED);

        bool    runState = false;
        cv::Mat dstImg;
        {
            auto sw  = Stopwatch{"Ithaca::image_enhancement_TAGC"};
            runState = Ithaca::image_enhancement_TAGC(srcImg, dstImg);
        }
        if (runState)
            std::cout << "Run Successfully\n";
        else
            std::cout << "Run NOT Successfully\n";

        std::string saveFilename = std::filesystem::path(filepath).filename().string();

        bool flag = cv::imwrite(saveFilename, dstImg);
        if (flag)
            std::cout << "Save Successfully\n";
        else
            std::cout << "Save NOT Successfully\n";
    }
}

} // namespace Ithaca
