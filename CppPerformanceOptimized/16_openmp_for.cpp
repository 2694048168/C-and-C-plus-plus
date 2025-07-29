/**
 * @file 16_openmp_for.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief OpenMP parallel for for-loop
 * @version 0.1
 * @date 2025-07-29
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "opencv2/opencv.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

void warmUpCPU() {}

void warmUpGPU() {}

void computeStatisticalInfo(const std::vector<double> &methodVec, int iterationNum)
{
    auto average = std::reduce(methodVec.begin(), methodVec.end(), 0.0) / iterationNum;
    std::cout << "The average time: " << average << " ms for " << iterationNum << " Iteration\n";
    std::cout << "The max time: " << *std::max_element(methodVec.begin(), methodVec.end()) << " ms for " << iterationNum
              << " Iteration\n";
    std::cout << "The min time: " << *std::min_element(methodVec.begin(), methodVec.end()) << " ms for " << iterationNum
              << " Iteration\n";

    double accum = 0.0;
    std::for_each(methodVec.begin(), methodVec.end(), [&](const double d) { accum += (d - average) * (d - average); });

    auto variance_           = accum / (iterationNum - 1);
    auto standard_deviation_ = std::sqrt(accum / (iterationNum - 1));
    std::cout << "The variance time: " << variance_ << " ms for " << iterationNum << " Iteration\n";
    std::cout << "The standard deviation time: " << standard_deviation_ << " ms for " << iterationNum
              << " Iteration\n\n";
}

// Debug & Release 耗时不一样, 优化方向和策略
// ------------------------------------
int main(int argc, const char *argv[])
{
    std::string filepath
        = R"(D:\Development\GitRepository\C-and-C-plus-plus\AlgorithmEncapsulationOpenCV\images/smudge.png)";
    // auto image = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
    auto image = cv::imread(filepath);
    int  w     = image.cols;
    int  h     = image.rows;
    int  c     = image.channels();

    int                 iterationNum = 1000; // 10, 100, 1000, 10000, 100000
    std::vector<double> method1Vec;
    method1Vec.reserve(iterationNum);
    std::vector<double> method2Vec;
    method2Vec.reserve(iterationNum);
    std::vector<double> method3Vec;
    method3Vec.reserve(iterationNum);
    std::vector<double> method4Vec;
    method4Vec.reserve(iterationNum);

    // for-for to access pixel of image
    {
        for (int iterIdx{0}; iterIdx < iterationNum; ++iterIdx)
        {
            auto startTime = std::chrono::high_resolution_clock::now();

            // 基于cv::Mat对象的随机像素访问API实现，通过行列索引方式遍历每个像素值
            if (3 == c)
            {
                for (int row = 0; row < h; row++)
                {
                    for (int col = 0; col < w; col++)
                    {
                        auto bgr                      = image.at<cv::Vec3b>(row, col);
                        bgr[0]                        = 255 - bgr[0];
                        bgr[1]                        = 255 - bgr[1];
                        bgr[2]                        = 255 - bgr[2];
                        image.at<cv::Vec3b>(row, col) = bgr;
                    }
                }
            }
            else if (1 == c)
            {
                for (int row = 0; row < h; row++)
                {
                    for (int col = 0; col < w; col++)
                    {
                        auto pixel                = image.at<uchar>(row, col);
                        pixel                     = 255 - pixel;
                        image.at<uchar>(row, col) = pixel;
                    }
                }
            }
            else
            {
                std::cout << "Not supported Channel for image\n";
            }

            auto endTime  = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            method1Vec.emplace_back(duration.count());
        }
        /*
        auto average = std::reduce(method1Vec.begin(), method1Vec.end(), 0.0) / iterationNum;
        std::cout << "The average time: " << average << " ms for " << iterationNum << " Iteration\n";
        std::cout << "The max time: " << *std::max_element(method1Vec.begin(), method1Vec.end()) << " ms for " << iterationNum << " Iteration\n";
        std::cout << "The min time: " << *std::max_element(method1Vec.begin(), method1Vec.end()) << " ms for " << iterationNum << " Iteration\n";

        double accum = 0.0;
        std::for_each(method1Vec.begin(), method1Vec.end(), [&](const double d) {
            accum += (d - average) * (d - average);
            });

        auto variance_ = accum / (iterationNum - 1);
        auto standard_deviation_ = std::sqrt(accum / (iterationNum - 1));
        std::cout << "The variance time: " << variance_ << " ms for " << iterationNum << " Iteration\n";
        std::cout << "The standard deviation time: " << standard_deviation_ << " ms for " << iterationNum << " Iteration\n";*/
        computeStatisticalInfo(method1Vec, iterationNum);
    }

    // for-for to access pixel of image
    {
        for (int iterIdx{0}; iterIdx < iterationNum; ++iterIdx)
        {
            auto startTime = std::chrono::high_resolution_clock::now();

            // 基于cv::Mat对象的行随机访问指针方式实现对每个像素的遍历
            if (3 == c)
            {
                for (int row = 0; row < h; row++)
                {
                    /* 行指针遍历方式, 常见的行指针
                    1. CV_8UC1: 灰度图像
                        uchar* ptr = image.ptr<uchar>(row_index);
                    2. CV_8UC3: 彩色图像
                        Vec3b* ptr = image.ptr<cv::Vec3b>(row_index);
                    3. CV_32FC1: 单通道浮点数图像
                        float* ptr = image.ptr<float>(row_index);
                    4. CV_32FC3: 三通道浮点数图像
                        Vec3f* ptr = image.ptr<cv::Vec3f>(row_index);
                    */
                    cv::Vec3b *pCurr = image.ptr<cv::Vec3b>(row);
                    for (int col = 0; col < w; col++)
                    {
                        cv::Vec3b bgr = pCurr[col];
                        bgr[0]        = 255 - bgr[0];
                        bgr[1]        = 255 - bgr[1];
                        bgr[2]        = 255 - bgr[2];
                    }
                }
            }
            else if (1 == c)
            {
                for (int row = 0; row < h; row++)
                {
                    uchar *pCurr = image.ptr<uchar>(row);
                    for (int col = 0; col < w; col++)
                    {
                        uchar pixel = pCurr[col];
                        pixel       = 255 - pixel;
                        pCurr[col]  = pixel;
                    }
                }
            }
            else
            {
                std::cout << "Not supported Channel for image\n";
            }

            auto endTime  = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            method2Vec.emplace_back(duration.count());
        }
        //std::cout << "The average time: " << std::reduce(method2Vec.begin(), method2Vec.end(), 0.0) / iterationNum << " ms for " << iterationNum << " Iteration\n";
        computeStatisticalInfo(method2Vec, iterationNum);
    }

    // for-for to access pixel of image
    {
        for (int iterIdx{0}; iterIdx < iterationNum; ++iterIdx)
        {
            auto startTime = std::chrono::high_resolution_clock::now();

            // 直接获取cv::Mat对象的像素块的数据指针，基于指针操作，实现快速像素方法
            if (3 == c)
            {
                for (int row = 0; row < h; row++)
                {
                    uchar *uc_pixel = image.data + row * image.step;
                    for (int col = 0; col < w; col++)
                    {
                        uc_pixel[0] = 255 - uc_pixel[0];
                        uc_pixel[1] = 255 - uc_pixel[1];
                        uc_pixel[2] = 255 - uc_pixel[2];
                        uc_pixel += 3;
                    }
                }
            }
            else if (1 == c)
            {
                for (int row = 0; row < h; row++)
                {
                    uchar *uc_pixel = image.data + row * image.step;
                    for (int col = 0; col < w; col++)
                    {
                        uchar pixel   = uc_pixel[col];
                        pixel         = 255 - pixel;
                        uc_pixel[col] = pixel;
                    }
                }
            }
            else
            {
                std::cout << "Not supported Channel for image\n";
            }

            auto endTime  = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            method3Vec.emplace_back(duration.count());
        }
        //std::cout << "The average time: " << std::reduce(method3Vec.begin(), method3Vec.end(), 0.0) / iterationNum << " ms for " << iterationNum << " Iteration\n";
        computeStatisticalInfo(method3Vec, iterationNum);
    }

    // for-for with OpenMP to access pixel of image
    /* /openmp /std:c++20
    * VS--->[项目]--->[属性]--->[C/C++]--->[语言]--->[OpenMP支持]--->是
    */
    {
        for (int iterIdx{0}; iterIdx < iterationNum; ++iterIdx)
        {
            auto startTime = std::chrono::high_resolution_clock::now();

            if (3 == c)
            {
#pragma omp parallel for
                for (int row = 0; row < h; row++)
                {
                    uchar *uc_pixel = image.data + row * image.step;
                    for (int col = 0; col < w; col++)
                    {
                        uc_pixel[0] = 255 - uc_pixel[0];
                        uc_pixel[1] = 255 - uc_pixel[1];
                        uc_pixel[2] = 255 - uc_pixel[2];
                        uc_pixel += 3;
                    }
                }
            }
            else if (1 == c)
            {
#pragma omp parallel for
                for (int row = 0; row < h; row++)
                {
                    uchar *uc_pixel = image.data + row * image.step;
                    for (int col = 0; col < w; col++)
                    {
                        uchar pixel   = uc_pixel[col];
                        pixel         = 255 - pixel;
                        uc_pixel[col] = pixel;
                    }
                }
            }
            else
            {
                std::cout << "Not supported Channel for image\n";
            }

            auto endTime  = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            method4Vec.emplace_back(duration.count());
        }
        //std::cout << "The average time: " << std::reduce(method4Vec.begin(), method4Vec.end(), 0.0) / iterationNum << " ms for " << iterationNum << " Iteration\n";
        computeStatisticalInfo(method4Vec, iterationNum);
    }

    return 0;
}
