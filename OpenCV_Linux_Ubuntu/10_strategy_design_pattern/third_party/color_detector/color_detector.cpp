/**
 * @File    : color_detector.cpp
 * @Brief   : 采用 策略设计模式 来实现算法的封装
 * @Author  : Wei Li
 * @Date    : 2021-07-27
*/

#include <vector>
#include "color_detector.hpp"

/*
// 图像处理，获取一个通道的二值化图像
cv::Mat ColorDetector::process(const cv::Mat &image)
{
    // 二值化图像
    result.create(image.size(), CV_8U);

    if (useLab)
    {
        cv::cvtColor(image, converted, cv::COLOR_BGR2Lab);
    }

    // 利用 迭代器 对像素进行处理
    cv::Mat_<cv::Vec3b>::const_iterator iterator_begin = image.begin<cv::Vec3b>();
    cv::Mat_<cv::Vec3b>::const_iterator iterator_end = image.end<cv::Vec3b>();
    if (useLab)
    {
        cv::Mat_<cv::Vec3b>::const_iterator iterator_begin = converted.begin<cv::Vec3b>();
        cv::Mat_<cv::Vec3b>::const_iterator iterator_end = converted.end<cv::Vec3b>();
    }

    cv::Mat_<uchar>::iterator iterator_output = result.begin<uchar>();

    // each pixle process 二值化处理
    for (; iterator_begin != iterator_end; ++iterator_begin, ++iterator_output)
    {
        if (getDistanceToTargetColor(*iterator_begin) < maxDist)
        {
            // 满足目标颜色的像素直接变为 白色
            *iterator_output = 255;
        }
        else
        {
            *iterator_output = 0;
        }
    }

    return result;
}
*/

/**图像处理，获取一个通道的二值化图像
 * 使用 OpenCV 函数进行实现
 * 一般来说，最好直接使用 OpenCV 函数。它可以快速建立复杂程序，减少潜在的错误，
 * 而且程序的运行效率通常也比较高（得益于 OpenCV 项目参与者做的优化工作）。
 * 不过这样会执行很多的中间步骤，消耗更多内存。
 */
cv::Mat ColorDetector::process(const cv::Mat &image)
{
    cv::Mat output;

    // 计算与目标颜色的距离的绝对值
    cv::absdiff(image, cv::Scalar(target), output);

    // 把通道分割进 3 幅图像
    std::vector<cv::Mat> images;
    cv::split(output, images);

    // 3 个通道相加（这里可能出现饱和的情况） OpenCV 使用饱和算法特性???
    output = images[0] + images[1] + images[2];
    // 应用阈值
    cv::threshold(output, // 相同的输入/输出图像
                  output,
                  maxDist,                // 阈值（必须<256）
                  255,                    // 最大值
                  cv::THRESH_BINARY_INV); // 阈值化模式
    return output;
}