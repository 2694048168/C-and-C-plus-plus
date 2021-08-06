/**
 * @File    : colorDetection.cpp
 * @Brief   : 策略设计模式把算法封装进类，通过创建类的实例来部署算法
 *          OpenCV 中定义了一个算法基类 cv::Algorithm，实现策略设计模式的概念
 * @Author  : Wei Li
 * @Date    : 2021-07-27
*/

#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include "color_detector.hpp"

// ----------------------------
int main(int argc, char **argv)
{
    cv::Mat image = cv::imread("./../images/boldt.jpg");
    if (image.empty())
    {
        std::cerr << "Error reading image file." << std::endl;
        return 1;
    }

    const cv::String win_name_orginal = "OriginalImage";
    cv::namedWindow(win_name_orginal);
    cv::imshow(win_name_orginal, image);

    ColorDetector color_detect;
    color_detect.setTargetColor(230, 190, 130); // here blue sky
    cv::Mat result = color_detect.process(image);

    const cv::String win_name_detect = "OriginalImageDetect";
    cv::namedWindow(win_name_detect);
    cv::imshow(win_name_detect, result);

    // 利用函数对象 () 运算符重载
    // here distance is measured with the Lab color space
    ColorDetector colordetector(230, 190, 130, // color
                                45, true);     // Lab threshold
    cv::namedWindow("result (functor)");
    result = colordetector(image);
    cv::imshow("result (functor)", result);


    // floodFill 函数 OpenCV 函数实现相同的功能
    // 很大的区别，那就是它在判断一个像素时，还要检查附近像素的状态，这是为了识别某种颜色的相关区域。
    // 用户只需指定一个起始位置和允许的误差，就可以找出颜色接近的连续区域。
    cv::floodFill(image,                      // input/ouput image
                  cv::Point(100, 50),         // seed point
                  cv::Scalar(255, 255, 255),  // repainted color
                  (cv::Rect *)0,              // bounding rectangle of the repainted pixel set
                  cv::Scalar(35, 35, 35),     // low and high difference threshold
                  cv::Scalar(35, 35, 35),     // most of the time will be identical
                  cv::FLOODFILL_FIXED_RANGE); // pixels are compared to seed color

    cv::namedWindow("Flood Fill result");
    result = colordetector(image);
    cv::imshow("Flood Fill result", image);

    // Creating artificial images to demonstrate color space properties
    cv::Mat colors(100, 300, CV_8UC3, cv::Scalar(100, 200, 150));
    cv::Mat range = colors.colRange(0, 100);
    range = range + cv::Scalar(10, 10, 10);
    range = colors.colRange(200, 300);
    range = range + cv::Scalar(-10, -10, 10);

    cv::namedWindow("3 colors");
    cv::imshow("3 colors", colors);

    cv::Mat labImage(100, 300, CV_8UC3, cv::Scalar(100, 200, 150));
    cv::cvtColor(labImage, labImage, cv::COLOR_BGR2Lab);
    range = colors.colRange(0, 100);
    range = range + cv::Scalar(10, 10, 10);
    range = colors.colRange(200, 300);
    range = range + cv::Scalar(-10, -10, 10);
    cv::cvtColor(labImage, labImage, cv::COLOR_Lab2BGR);

    cv::namedWindow("3 colors (Lab)");
    cv::imshow("3 colors (Lab)", colors);

    // brightness versus luminance
    cv::Mat grayLevels(100, 256, CV_8UC3);
    for (int i = 0; i < 256; i++)
    {
        grayLevels.col(i) = cv::Scalar(i, i, i);
    }

    range = grayLevels.rowRange(50, 100);
    cv::Mat channels[3];
    cv::split(range, channels);
    channels[1] = 128;
    channels[2] = 128;
    cv::merge(channels, 3, range);
    cv::cvtColor(range, range, cv::COLOR_Lab2BGR);

    cv::namedWindow("Luminance vs Brightness");
    cv::imshow("Luminance vs Brightness", grayLevels);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
