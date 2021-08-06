/**
 * @File    : integral.hpp
 * @Brief   : 用积分图像统计像素
 * @Author  : Wei Li
 * @Date    : 2021-07-29
*/

/**积分图像 & 直方图
 * 直方图： 直方图的计算方法，即遍历图像的全部像素并累计每个强度值在图像中出现的次数
 * 有时只需要计算图像中某个特定区域的直方图。
 * 实际上，累计图像某个子区域内的像素总数是很多计算机视觉算法中的常见过程。
 * 假设需要对图像中的多个感兴趣区域计算几个此类直方图，这些计算过程马上都会变得非常耗时。
 * 这种情况下，有一个工具可以极大地提高统计图像子区域像素的效率，那就是积分图像。
 * 
 * 使用积分图像统计图像感兴趣区域的像素是一种高效的方法。
 * 它在程序中的应用非常广泛，例如用于计算基于不同大小的滑动窗口。
 * 积分图像背后的原理。只用三次算术运算，就能累加一个矩形区域的像素，有效使用积分图像的实例。
 * 
 * 积分图像：取图像左上方的全部像素计算累加和，并用这个累加和替换图像中的每一个像素，
 * 用这种方式得到的图像称为积分图像。计算积分图像时，只需对图像扫描一次。
 * 实际上，当前像素的积分值等于上方像素的积分值加上当前行的累计值。
 * 因此积分图像就是一个包含像素累加和的新图像。
 * 防止溢出，积分图像的值通常采用 int 类型（ CV_32S）或 float 类型（ CV_32F）。
 * 
 * 计算完积分图像后，只需要访问四个像素就可以得到任何矩形区域的像素累加和
 * 不管感兴趣区域的尺寸有多大，使用这种方法计算的复杂度是恒定不变的
 * 
 */

#ifndef IMAGE_INTEGRAL
#define IMAGE_INTEGRAL

#include <vector>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

template <typename T, int N>
class IntegralImage
{
private:
    cv::Mat integralImage;

public:
    // 构造函数
    IntegralImage(cv::Mat image)
    {
        // 计算积分图
        cv::integral(image, integralImage, cv::DataType<T>::type);
    }

    // 通过四个点的访问，计算任意大小的感兴趣子区域的直方图
    // 运算符重载，函数对象
    cv::Vec<T, N> operator()(int x0, int y0, int width, int height)
    {
        return (integralImage.at<cv::Vec<T, N>>(y0 + height, x0 + width) 
                - integralImage.at<cv::Vec<T, N>>(y0 + height, x0) 
                - integralImage.at<cv::Vec<T, N>>(y0, x0 + width) 
                + integralImage.at<cv::Vec<T, N>>(y0, x0));
    }
    // 重载版本
    cv::Vec<T, N> operator()(int x, int y, int radius)
    {
        // square window centered at (x,y) of size 2*radius+1
        return (integralImage.at<cv::Vec<T, N>>(y + radius + 1, x + radius + 1) 
                - integralImage.at<cv::Vec<T, N>>(y + radius + 1, x - radius) 
                - integralImage.at<cv::Vec<T, N>>(y - radius, x + radius + 1) 
                + integralImage.at<cv::Vec<T, N>>(y - radius, x - radius));
    }
};

// 由 0 和 1 组成的二值图像生成积分图像是一种特殊情况，
// 这时的积分累计值就是指定区域内值为 1 的像素总数
// convert to a multi-channel image made of binary planes
// nPlanes must be a power of 2
void convertToBinaryPlanes(const cv::Mat &input, cv::Mat &output, int nPlanes)
{
    // 需要屏蔽的位数
    int n = 8 - static_cast<int>(std::log(static_cast<double>(nPlanes)) / std::log(2.0));
    // 用来消除最低有效位的掩码
    uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0

    // 创建二值图像的向量
    std::vector<cv::Mat> planes;
    // 消除最低有效位，箱子数减为 nBins
    cv::Mat reduced = input & mask;
    // 计算每个二值图像平面
    for (int i = 0; i < nPlanes; i++)
    {
        // 将每个等于 i<<shift 的像素设为 1
        planes.push_back((reduced == (i << n)) & 0x1);
    }
    // 创建多通道图像
    cv::merge(planes, output);
}

#endif // IMAGE_INTEGRAL