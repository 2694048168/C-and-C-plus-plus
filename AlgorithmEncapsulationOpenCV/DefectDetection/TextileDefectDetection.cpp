/**
 * @file TextileDefectDetection.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-07
 * 
 * @copyright Copyright (c) 2025
 * 
 * g++ TextileDefectDetection.cpp -std=c++20
 * clang++ TextileDefectDetection.cpp -std=c++20
 * 
 */

#include "opencv2/opencv.hpp"

#include <cmath>
#include <iostream>
#include <vector>

#ifdef _WIN32
#    include <Windows.h>
#endif // _WIN32

/* 机器视觉应用场景中缺陷检测的应用是非常广泛的, 通常涉及各个行业、各种缺陷类型.
 * 纺织物的缺陷检测, 缺陷类型包含脏污、油渍、线条破损三种;
 * 这三种缺陷与LCD屏幕检测的缺陷很相似, 处理方法也可借鉴.
 * 
 */

// -------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif // _WIN32

    // # -------------------------------------------------------
    // 脏污缺陷图片, 肉眼可见明显的几处脏污
    cv::Mat img = cv::imread("./images/smudge.png", cv::IMREAD_UNCHANGED);
    // 【1】使用高斯滤波消除背景纹理的干扰, 将原图放大后会发现纺织物自带的纹理比较明显,
    // 这会影响后续处理结果, 所以先做滤波平滑
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(7, 7), 0);
    cv::imwrite("./images/smudge_blur.png", blur);
    // 【2】Canny边缘检测凸显缺陷,
    // Canny边缘检测对低对比度缺陷检测有很好的效果, 这里注意高低阈值的设置
    cv::Mat edged;
    cv::Canny(blur, edged, 10, 30);
    cv::imwrite("./images/smudge_edged.png", edged);
    // 【3】轮廓查找、筛选与结果标记,
    // 轮廓筛选可以根据面积、长度过滤掉部分干扰轮廓, 找到真正的缺陷
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i>              hierarchy;
    cv::findContours(edged, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    for (const auto &cnt : contours)
    {
        auto length = cv::arcLength(cnt, true);
        if (length > 1.0)
            cv::drawContours(img, contours, -1, (255, 0, 0), 2, 8, hierarchy);
    }
    cv::imwrite("./images/smudge_defect.png", img);

    return 0;
}
