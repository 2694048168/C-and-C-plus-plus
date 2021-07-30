/**
 * @File    : load_display_save_image.cpp
 * @Brief   : 加载、显示和保存图像
 * @Author  : Wei Li
 * @Date    : 2021-07-23
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


// -----------------------------
int main(int argc, char** argv)
{
    cv::Mat image;
    std::cout << "This image is " << image.rows << " x " << image.cols << std::endl;

    // -----------------------------------------------------------------
    // 读取图像同时进行色彩转换，灰度图提高运行速度同时减少内存使用 CV_8U
    // image = cv::imread("./../images/puppy.bmp", cv::IMREAD_GRAYSCALE);
    // 读取图像，转换为三通道彩色 CV_8UC3
    // image = cv::imread("./../images/puppy.bmp"， cv::IMREAD_COLOR);
    // 在读入图像时候需要采用文件本身的数据格式，将第二个参数设置为 负数即可 (-1)
    // image = cv::imread("./../images/puppy.bmp"， -1);
    // -----------------------------------------------------------------
    // imread 采用默认路径：终端运行程序，默认开始路径为当前终端所在目录；
    // IDE 运行程序，默认开始路径为项目程序文件所在路径；
    image = cv::imread("./../../images/puppy.bmp");
    if (image.empty())
    {
        std::cout << "NO image file." << std::endl;
    }

    std::cout << "The information of image : rows=" << image.rows 
              << ", columns=" << image.cols 
              << ", channels=" << image.channels() << std::endl;

    cv::namedWindow("Original Image");
    cv::imshow("Original Image", image);

    cv::Mat result;
    // 水平翻转， 0 表示垂直； 正数表示水平； 负数表示水平和垂直
    cv::flip(image, result, 1);

    cv::namedWindow("Flip Image");
    cv::imshow("Flip Image", result);

    // 0 表示一直等待用户按键处理； 正数表示等待的毫秒数
    cv::waitKey(0);

    // 支持 JPG; TIFF; PNG图像格式
    cv::imwrite("flip.bmp", result);
    
    return 0;
}
