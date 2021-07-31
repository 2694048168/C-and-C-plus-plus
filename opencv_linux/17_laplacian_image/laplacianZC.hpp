/**
 * @File    : laplacianZC.hpp
 * @Brief   : 计算拉普拉斯算子
 *      拉普拉斯算子也是一种基于图像导数运算的高通线性滤波器，
 *     它通过计算二阶导数来度量图像函数的曲率
 * @Author  : Wei Li
 * @Date    : 2021-07-30
*/

#ifndef LAPLACIANZC
#define LAPLACIANZC

#include <opencv2/imgproc.hpp>

class LaplacianZC
{
private:
    // 拉普拉斯算子
    cv::Mat laplace;
    // 拉普拉斯内核的孔径大小
    int aperture;

public:
    // 默认无参构造函数
    LaplacianZC() : aperture(3) {}

    // 类的 set 和 get 方法
    void setAperture(int value)
    {
        aperture = value;
    }
    int getAperture() const
    {
        return aperture;
    }

    // 计算浮点数类型的拉普拉斯算子
    cv::Mat computeLaplacian(const cv::Mat &image)
    {
        // 计算拉普拉斯算子
        cv::Laplacian(image, laplace, CV_32F, aperture);
        return laplace;
    }

    /**拉普拉斯算子的计算在浮点数类型的图像上进行
     * 对结果做缩放处理才能使其正常显示
     * 缩放基于拉普拉斯算子的最大绝对值，其中数值 0 对应灰度级 128
     * 
     * 获得拉普拉斯结果，存在 8 位图像中
     * 0 表示灰度级 128
     * 如果不指定缩放比例，那么最大值会放大到 255
     * 在调用这个函数之前，必须先调用 computeLaplacian
     */
    cv::Mat getLaplacianImage(double scale = -1.0)
    {
        if (scale < 0)
        {
            double lapmin, lapmax;
            // 取得最小和最大拉普拉斯值
            cv::minMaxLoc(laplace, &lapmin, &lapmax);
            // 缩放拉普拉斯算子到 127
            scale = 127 / std::max(-lapmin, lapmax);
        }

        // 生成灰度图像
        cv::Mat laplaceImage;
        laplace.convertTo(laplaceImage, CV_8U, scale, 128);
        return laplaceImage;
    }

    // Get a binary image of the zero-crossings
    // laplacian image should be CV_32F
    cv::Mat getZeroCrossings(cv::Mat laplace)
    {
        // threshold at 0
        // negative values in black
        // positive values in white
        cv::Mat signImage;
        cv::threshold(laplace, signImage, 0, 255, cv::THRESH_BINARY);

        // convert the +/- image into CV_8U
        cv::Mat binary;
        signImage.convertTo(binary, CV_8U);

        // dilate the binary image of +/- regions
        cv::Mat dilated;
        cv::dilate(binary, dilated, cv::Mat());

        // return the zero-crossing contours
    return dilated - binary;
    }
};

#endif // LAPLACIANZC