/**
 * @File    : edge_detector.hpp
 * @Brief   : Sobel and Canny 算子检测图像轮廓
 * @Author  : Wei Li
 * @Date    : 2021-07-31
*/

#ifndef SOBELEDGES
#define SOBELEDGES

#define PI 3.1415926

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class EdgeDEtector
{
private:
    cv::Mat img;              // 原始图像
    cv::Mat soble;            // 16-bit signed int image
    int aperture;             // Aperture size of the Sobel kernel; Sobel 核的大小
    cv::Mat sobelMagnitude;   // Sobel magnitude 幅度
    cv::Mat sobelOrientation; // Sobel orientation 方向

public:
    // 无参默认构造函数
    EdgeDEtector() : aperture(3) {}

    // 类的 set 和 get 方法
    void setAperture(int kernel_size)
    {
        aperture = kernel_size;
    }
    int getAperture() const
    {
        return aperture;
    }

    cv::Mat getOrientation() const
    {
        return sobelOrientation;
    }

    cv::Mat getMagnitude() const
    {
        return sobelMagnitude;
    }

    // 计算 Sobel
    void computeSobel(const cv::Mat &image)
    {
        cv::Mat sobelX;
        cv::Mat sobelY;
        cv::Sobel(image, sobelX, CV_32F, 1, 0, aperture);
        cv::Sobel(image, sobelY, CV_32F, 0, 1, aperture);
        // 计算 Sobel 算子的 幅度大小和梯度方向
        cv::cartToPolar(sobelX, sobelY, sobelMagnitude, sobelOrientation);
    }

    // 函数重载
    void computeSobel(const cv::Mat &image, cv::Mat &sobelX, cv::Mat &sobelY)
    {
        cv::Sobel(image, sobelX, CV_32F, 1, 0, aperture);
        cv::Sobel(image, sobelY, CV_32F, 0, 1, aperture);
        cv::cartToPolar(sobelX, sobelY, sobelMagnitude, sobelOrientation);
    }

    // 设置阈值获取二值图像 Sobel 算子
    cv::Mat getBinaryMap(double threshold)
    {
        cv::Mat binary_img;
        cv::threshold(sobelMagnitude, binary_img, threshold, 255, cv::THRESH_BINARY_INV);
        return binary_img;
    }

    // 获取一个单通道的灰度图 Sobel 算子
    cv::Mat getSobelImage()
    {
        cv::Mat gray_img;
        double minVal, maxVal;
        cv::minMaxLoc(sobelMagnitude, &minVal, &maxVal);
        sobelMagnitude.convertTo(gray_img, CV_8U, 255 / maxVal);
        return gray_img;
    }

    // Get a CV_8U image of the Sobel orientation
    // 1 gray-level = 2 degrees
    cv::Mat getSobelOrientationImage()
    {
        cv::Mat bin;
        sobelOrientation.convertTo(bin, CV_8U, 90 / PI);
        return bin;
    }
};

#endif // SOBELEDGES