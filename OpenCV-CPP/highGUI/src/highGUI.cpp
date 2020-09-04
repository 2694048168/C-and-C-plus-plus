/* Core functionality (core) 
** - a compact module defining basic data structures, 
** including the dense multi-dimensional array Mat 
** and basic functions used by all other modules.
**
** 核心功能（core）
** - 紧凑的模块，定义基本数据结构，
** 包括密集的多维数组 Mat
** 和所有其他模块使用的基本功能。
*/
#include "opencv2/core.hpp"

/* Image Processing (imgproc) 
** - an image processing module 
** that includes linear and non-linear image filtering, 
** geometrical image transformations 
** (resize, affine and perspective warping, generic table-based remapping), 
** color space conversion, histograms, and so on.
**
** 图像处理（imgproc）
** - 一种图像处理模块，
** 包括线性和非线性图像过滤，
** 几何图像转换
**（调整大小，仿射和透视变换，基于通用表的重新映射），
** 颜色空间转换，直方图等。
*/
#include "opencv2/imgproc.hpp"

/* High-level GUI (highgui) 
** - an easy-to-use interface to simple UI capabilities.
**
** 高级GUI（highgui）
** - 简单易用的 UI 功能界面。
*/
#include "opencv2/highgui.hpp"

/* Video I/O (videoio) 
** - an easy-to-use interface to video capturing and video codecs.
**
** 视频 I/O（videoio）
** -用于视频捕获和视频编解码器的易于使用的界面。
*/
#include "opencv2/videoio.hpp"

#include <iostream>

/* 命名空间 namespace
** case 1: cv::Mat; case 2: using namespace cv;
** 建议初学者使用 cv:: 方式使用命名空间，便于知道哪些类和函数是OpenCV的
** 同时由于一些函数或者类与 C++ 的 STL有冲突，建议使用 cv:: 方式
** Some of the current or future OpenCV external names may conflict with STL or other libraries. 
** In this case, use explicit namespace specifiers to resolve the name conflicts:
*/

// 建议初学者使用 std::
// using namespace std;

// 在图像 (20，50) 像素位置绘制 Hello OpenCV 文本
void drawText(cv::Mat & image);

int main(int argc, char ** argv)
{
    std::cout << "Built with OpenCV " << CV_VERSION << std::endl;
    cv::Mat image;
    cv::VideoCapture capture;
    // 打开默认摄像头
    capture.open(0);
    if(capture.isOpened())
    {
        std::cout << "Capture is opened" << std::endl;
        for(;;)
        {
            // 捕捉画面
            capture >> image;
            if(image.empty())
                break;
            // 在每一帧图像上绘制 Hello OpenCV 文字
            drawText(image);
            cv::imshow("Sample", image);
            // 每隔十毫秒，ESC键退出 = 27
            if(cv::waitKey(10) == 27)
                break;
        }
    }
    else
    {
        // 捕捉不到画面，就默认显示
        std::cout << "No capture" << std::endl;
        image = cv::Mat::zeros(480, 640, CV_8UC1);
        drawText(image);
        cv::imshow("Sample", image);
        cv::waitKey(0);
    }

    return 0;
}

/* 在图像 (20，50) 像素位置绘制 Hello OpenCV
** 字体、颜色、线条大小等等设置
*/
void drawText(cv::Mat & image)
{
    cv::putText(image, "Hello OpenCV",
                cv::Point(20, 50),
                cv::FONT_HERSHEY_COMPLEX, 1,  // font face and scale
                cv::Scalar(255, 255, 255),  // white
                1, cv::LINE_AA);          // line thickness and type
}
