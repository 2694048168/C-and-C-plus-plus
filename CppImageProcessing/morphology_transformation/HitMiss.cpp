/**
 * @brief Goal: Hit-or-Miss theory
 * In this tutorial will learn how to find a given configuration or pattern
 * in a binary image by using the Hit-or-Miss transform
 * (also known as Hit-and-Miss transform). 
 * This transform is also the basis of more advanced morphological operations 
 * such as thinning or pruning.
 *
 * @file HitMiss.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-02
 * @version OpenCV 4.7 examples
 *
 */

#include <iostream>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

int main(int argc, char const **argv)
{
    // a binary image(size of 8x8) as following:
    cv::Mat input_img = (cv::Mat_<uchar>(8, 8) <<
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 255, 255, 255, 0, 0, 0, 255,
        0, 255, 255, 255, 0, 0, 0, 0,
        0, 255, 255, 255, 0, 255, 0, 0,
        0, 0, 255, 0, 0, 0, 0, 0,
        0, 0, 255, 0, 0, 255, 255, 0,
        0, 255, 0, 255, 0, 0, 255, 0,
        0, 255, 255, 255, 0, 0, 0, 0);

    // the kernel for filter or transform
    cv::Mat kernel = (cv::Mat_<int>(3, 3) <<
        0, 1, 0,
        1, -1, 1,
        0, 1, 0);        
    
    cv::Mat output_img;
    cv::morphologyEx(input_img, output_img, cv::MORPH_HITMISS, kernel);

    const int rate = 50;
    kernel = (kernel + 1) * 127;
    kernel.convertTo(kernel, CV_8U);

    cv::resize(kernel, kernel, cv::Size(), rate, rate, cv::INTER_NEAREST);
    cv::imshow("kernel", kernel);
    cv::moveWindow("kernel", 0, 0);

    cv::resize(input_img, input_img, cv::Size(), rate, rate, cv::INTER_NEAREST);
    cv::imshow("Original", input_img);
    cv::moveWindow("Original", 0, 200);

    cv::resize(output_img, output_img, cv::Size(), rate, rate, cv::INTER_NEAREST);
    cv::imshow("Hit or Miss", output_img);
    cv::moveWindow("Hit or Miss", 500, 200);

    cv::waitKey(0);

    return 0;
}
