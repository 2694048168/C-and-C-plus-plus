/**
 * @File    : mat_data.cpp
 * @Brief   : 深入了解 cv::Mat 数据结构的属性
 * @Author  : Wei Li
 * @Date    : 2021-07-25
*/

#include <iostream>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>


/** cv::Mat class
 * 1. cv::InputArray and cv::OutputArray 都是代理类
 * 2. cv::Matx class and subclass cv::Matx33d 处理小矩阵
 */


cv::Mat function_img()
{
    cv::Mat img(500, 500, CV_8U, 50);
    return img;
}

// -----------------------------
int main(int argc, char** argv)
{
    // [rows, columns, type_data, init_value]
    // type_data: CV_[The number of bits per item][Signed or Unsigned][Type Prefix]C[The channel number]
    cv::Mat image1(240, 320, CV_8U, 100);
    cv::imshow("Image", image1);
    cv::waitKey(0);

    // Ref: https://docs.opencv.org/4.5.3/d6/d6d/tutorial_mat_the_basic_image_container.html
    image1.create(200, 200, CV_8U);
    image1 = 200;
    cv::imshow("Image", image1);
    cv::waitKey(0);

    // channel order = [BGR]
    cv::Mat image2(240, 320, CV_8UC3, cv::Scalar(0, 0, 255));
    // cv::Mat image2(240, 320, CV_8UC3, cv::Scalar(0, 0, 255));
    // image2 = cv::Scalar(0, 0, 255);
    cv::imshow("ColorImage", image2);
    cv::waitKey(0);

    cv::Mat image3 = cv::imread("./../images/puppy.bmp");
    if (image3.empty())
    {
        std::cout << "Error reading image file." << std::endl;
        return 1;
    }
    // Mat: 包含 head 和 data
    // 指向同一个数据块 cv::Mat 实现计数引用和浅复制
    cv::Mat image4(image3);
    image1 = image3;

    // 深拷贝一个副本 
    image3.copyTo(image2);
    cv::Mat image5 = image3.clone();

    // in-place opeator
    cv::flip(image3, image3, 1);

    // 验证一下 Mat 数据结构
    cv::imshow("Image 3", image3);
    cv::imshow("Image 1", image1);
    cv::imshow("Image 2", image2);
    cv::imshow("Image 4", image4);
    cv::imshow("Image 5", image5);
    cv::waitKey(0);

    // cv::Mat 实现计数引用???和浅复制
    cv::Mat gray_img = function_img();
    cv::imshow("GrayImage", gray_img);
    cv::waitKey(0);

    image1 = cv::imread("./../images/puppy.bmp", cv::IMREAD_GRAYSCALE);
    image1.convertTo(image2, CV_32F, 1/255.0, 0.0);
    cv::imshow("ImageGray", image2);
    cv::waitKey(0);

    return 0;
}
