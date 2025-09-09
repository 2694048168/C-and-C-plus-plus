/**
 * @file TestImageTemplateMatch.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-09-09
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "ImageOperator/ImageTemplateMatch.h"
#include "test/Stopwatch.hpp"

#include <vector>

namespace Ithaca {

void TestImageTemplateMatch()
{
    auto sw = Stopwatch{"Ithaca::TestImageTemplateMatch"};

    std::string src_filepath   = "D:/Development/Image/TemplateMatch/lena.jpg";
    std::string templ_filepath = "D:/Development/Image/TemplateMatch/lena_tmpl.jpg";

    auto srcImg   = cv::imread(src_filepath, cv::IMREAD_UNCHANGED);
    auto templImg = cv::imread(templ_filepath, cv::IMREAD_UNCHANGED);

    bool    runState = false;
    cv::Mat dstImg;
    {
        auto sw  = Stopwatch{"Ithaca::ImageTemplateMatch::Run"};
        runState = Ithaca::ImageTemplateMatch::Run(srcImg, templImg, dstImg);
    }
    if (runState)
        std::cout << "Run Successfully\n";
    else
        std::cout << "Run NOT Successfully\n";

    bool flag = cv::imwrite("./dst_match.jpg", dstImg);
    if (flag)
        std::cout << "Save Successfully\n";
    else
        std::cout << "Save NOT Successfully\n";

    // ----------------------------------------
    cv::Mat dstImg1;
    {
        auto sw  = Stopwatch{"Ithaca::image_template_match_Opti"};
        runState = Ithaca::image_template_match_Opti(srcImg, templImg, dstImg1);
    }
    if (runState)
        std::cout << "Run Successfully\n";
    else
        std::cout << "Run NOT Successfully\n";

    flag = cv::imwrite("./dst1_match.jpg", dstImg1);
    if (flag)
        std::cout << "Save Successfully\n";
    else
        std::cout << "Save NOT Successfully\n";
}

} // namespace Ithaca