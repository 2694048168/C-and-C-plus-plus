// opencv_core 模块包含库的核心功能
#include <opencv2/core.hpp>
// opencv_imgproc 模块包含主要的图像处理函数
#include <opencv2/imgcodecs.hpp>
// opencv_highgui 模块提供了读写图像和视频的函数以及一些用户交互函数
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

// 建议初学者使用 cv:: 方式使用命名空间，便于知道哪些类和函数是OpenCV的
// using namespace cv;
// 建议初学者使用 std::
// using namespace std;

int main(int argc, char** argv)
{
    // 从控制台终端运行程序，则图像路径默认从可执行程序所在的目录开始 ./
    // 从IDE中运行程序，则图像路径默认从源文件所在目录开始 ./
    std::string imageName = "./../image/lena.jpg"; // by default
    if( argc > 1)
    {
        imageName = argv[1];
    }

    cv::Mat image;
    image = cv::imread(imageName, cv::IMREAD_COLOR); // Read the file
    if(image.empty())                      // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl;
        return -1;
    }

    std::cout << "the Original Image size: " << image.rows << " " << image.cols << std::endl;
    std::cout << "the Original Image channels: " << image.channels() << std::endl;

    cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Original Image", image);                // Show our image inside it.

    // flip function 如果不定义新的图像矩阵，则是就地修改原图像
    // OpenCV 中大部分函数都是直接修改原图像，所以经常需要创建一个副本进行图像处理
    cv::Mat flipImage;
    // 正数表示水平，0 表示垂直， 负数表示水平和垂直
    cv::flip(image, flipImage, 1);
    cv::namedWindow("Flip Image");
    std::cout << "the Original Image size: " << flipImage.rows << " " << flipImage.cols << std::endl;
    std::cout << "the Flip Image channels: " << flipImage.channels() << std::endl;
    cv::imshow("Flip Image", flipImage);
    cv::imwrite("./../image/flipImage.jpg", flipImage);

    // 读取图像并转换为灰度图
    cv::Mat grayImage = cv::imread("./../image/flipImage.jpg", cv::IMREAD_GRAYSCALE);
    std::cout << "the Original Image size: " << grayImage.rows << " " << grayImage.cols << std::endl;
    std::cout << "the Gray Image channels: " << grayImage.channels() << std::endl;
    cv::imshow("Gray Image", grayImage);
    cv::imwrite("./../iamge/grayImage.jpg", grayImage);

    cv::waitKey(0); // Wait for a keystroke in the window

    return 0;
}