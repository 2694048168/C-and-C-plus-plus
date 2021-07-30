/**
 * @File    : image_add_operator.cpp
 * @Brief   : 实现简单的图像运算 四则运算
 * @Author  : Wei Li
 * @Date    : 2021-07-27
*/

#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

/**图像简单计算
 * 在所有场合都要使用 cv::saturate_cast 函数，以确保结果在预定的像素值范围之内（避免上溢或下溢）
 * https://docs.opencv.org/4.5.3/d1/dfb/intro.html
 * OPenCV 几个重要概念： 模块化结构； cv 命令空间； 自动内存管理； 输出数据自动分配； 
 *                    计算采用饱和算法； 固定像素类型；输入输出数组
 */
int main(int argc, char **argv)
{
    cv::Mat image1 = cv::imread("./../images/boldt.jpg");
    cv::Mat image2 = cv::imread("./../images/rain.jpg");
    if (!image1.data)
    {
        std::cerr << "Error read image1 file." << std::endl;
        return 1;
    }
    if (!image2.data)
    {
        std::cerr << "Error read image1 file." << std::endl;
        return 1;
    }

    // cv::String 对 C++ string class 进行重载?
    const cv::String win_name1 = "Image1";
    cv::namedWindow(win_name1);
    cv::imshow(win_name1, image1);

    const cv::String win_name2 = "Image2";
    cv::namedWindow(win_name2);
    cv::imshow(win_name2, image2);

    cv::Mat result;
    // 带权重逐像素相加 element-wise add
    cv::addWeighted(image1, 0.7, image2, 0.9, 0., result);
    const cv::String win_name3 = "ImageAdd";
    cv::namedWindow(win_name3);
    cv::imshow(win_name3, result);

    // using overloaded operator
    // 针对图像运算 cv::Mat class 对 C++ 运算符进行了重载，可以直接使用
    result = 0.7 * image1 + 0.9 * image2;
    cv::namedWindow("result with operators");
    cv::imshow("result with operators", result);

    // -------------------------------------------
    // 通道的切分和合并 cv::split() and cv::merge()
    // 把一张雨景图只加到蓝色通道中
    image2 = cv::imread("./../images/rain.jpg", 0);
    // create vector of 3 images
    std::vector<cv::Mat> planes;
    // split 1 3-channel image into 3 1-channel images [B-G-R]
    cv::split(image1, planes);
    // add to blue channel
    planes[0] += image2;
    // merge the 3 1-channel images into 1 3-channel image
    cv::merge(planes, result);

    cv::namedWindow("Result on blue channel");
    cv::imshow("Result on blue channel", result);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
