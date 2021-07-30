/**
 * @File    : high_gui_mouse.cpp
 * @Brief   : highgui 对鼠标事件的响应
 * @Author  : Wei Li
 * @Date    : 2021-07-23
*/

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


/**1. 在图像上点击事件
 * 通过编程，你可以让鼠标在置于图像窗口上时运行特定的指令。
 * 要实现这个功能，需定义一个合适的回调函数。回调函数不会被显式地调用，
 * 但会在响应特定事件（这里是指有关鼠标与图像窗口交互的事件）的时候被程序调用。
 * 为了能被程序识别，回调函数需要具有特定的签名，并且必须注册。
 * 
 * 对于鼠标事件处理函数，回调函数必须具有这种签名：
 * void onMouse( int event, int x, int y, int flags, void* param);
 * 
 * 可用下面的方法在程序中注册回调函数：
 * cv::setMouseCallback("Original Image", onMouse, reinterpret_cast<void*>(&image));
 */
// 鼠标点击事件，显示灰度图像的对应的像素值
void onMouse(int event, int x, int y, int flags, void* param)
{
    cv::Mat *im = reinterpret_cast<cv::Mat*>(param);

    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN: // 鼠标左键按下事件
        std::cout << "at (" << x << "," << y << ") value is: " 
                  << static_cast<int>(im->at<u_char>(cv::Point(x,y))) << std::endl; 
        break;
    
    default:
        break;
    }
}

// ------------------------------
int main(int argc, char** argv)
{
    cv::Mat image = cv::imread("../images/puppy.bmp", cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        std::cout << "Error reading image file." << std::endl;
        return 1;
    }
    cv::namedWindow("DrawingImage");

    // set the mouse callback for this iamge.
    // 对鼠标事件的回调函数进行注册
    cv::setMouseCallback("DrawingImage", onMouse, reinterpret_cast<void*>(&image));

    /**2. 在图像上绘制图形
     * 在图像上绘制形状和写入文本的函数，基本形状 circle; ellipse; line; rectangle
     */

    // 目标图像, 中心点坐标, 半径, 颜色, 厚度
    cv::circle(image, cv::Point(155, 110), 65, 0, 3);

    // 目标图像, 文本, 文本位置, 字体类型, 字体大小, 字体颜色, 文本厚度
    cv::putText(image, "This is a dog.", cv::Point(40, 200), cv::FONT_HERSHEY_PLAIN, 2.0, 255, 2);

    cv::imshow("DrawingImage", image);
    cv::waitKey(0);
    
    return 0;
}
