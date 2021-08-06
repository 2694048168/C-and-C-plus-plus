/**
 * @File    : harrisDetector.hpp
 * @Brief   : 检测兴趣点——角点，特征，尺度不变和多尺度的特征
 * @Author  : Wei Li
 * @Date    : 2021-08-01
*/

#ifndef HARRIS_DETECTOR
#define HARRIS_DETECTOR

#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

class HarrisDetector
{
private:
    // 32 位浮点数型的角点强度图像 32-bit float image of corner strength
    cv::Mat cornerStrength;
    // 32 位浮点数型的阈值化角点图像 32-bit float image of thresholded corners
    cv::Mat cornerTh;
    // 局部最大值图像（内部） image of local maxima (internal)
    cv::Mat localMax;
    // 平滑导数的邻域尺寸 size of neighbourhood for derivatives smoothing
    int neighborhood;
    // 梯度计算的口径 aperture for gradient computation
    int aperture;
    // Harris 参数 Harris parameter
    double k;
    // 阈值计算的最大强度 maximum strength for threshold computation
    double maxStrength;
    // 计算得到的阈值（内部） calculated threshold (internal)
    double threshold;
    // 非最大值抑制的邻域尺寸 size of neighbourhood for non-max suppression
    int nonMaxSize;
    // 非最大值抑制的内核 kernel for non-max suppression
    cv::Mat kernel;

public:
    // 无参默认构造函数
    HarrisDetector() : neighborhood(3), aperture(3), k(0.1), maxStrength(0.0), threshold(0.01), nonMaxSize(3)
    {
        setLocalMaxWindowSize(nonMaxSize);
    }

    // 创建用于非最大值抑制的内核
    void setLocalMaxWindowSize(int size)
    {
        nonMaxSize = size;
        kernel.create(nonMaxSize, nonMaxSize, CV_8U);
    }

    /**检测 Harris 角点需要两个步骤
     * 1. 首先是计算每个像素的 Harris 值
     * 2. 然后，用指定的阈值获得特征点
     */
    // 1. Compute Harris corners
    void detect(const cv::Mat &image)
    {
        cv::cornerHarris(image, cornerStrength,
                         neighborhood, // 邻域尺寸
                         aperture,      // 口径尺寸
                         k);            // Harris 参数

        // 计算内部阈值
        cv::minMaxLoc(cornerStrength, 0, &maxStrength);

        // 检测局部最大值
        cv::Mat dilated; // temporary image
        cv::dilate(cornerStrength, dilated, cv::Mat());
        cv::compare(cornerStrength, dilated, localMax, cv::CMP_EQ);
    }

    // 2. 用 Harris 值得到角点分布图
    cv::Mat getCornerMap(double qualityLevel)
    {
        cv::Mat cornerMap;
        // 对角点强度阈值化
        threshold = qualityLevel * maxStrength;
        cv::threshold(cornerStrength, cornerTh, threshold, 255, cv::THRESH_BINARY);

        // 转换成 8 位图像
        cornerTh.convertTo(cornerMap, CV_8U);

        // 非最大值抑制
        cv::bitwise_and(cornerMap, localMax, cornerMap);

        return cornerMap;
    }

    // 用 Harris 值得到特征点
    void getCorners(std::vector<cv::Point> &points, double qualityLevel)
    {
        // 获得角点分布图
        cv::Mat cornerMap = getCornerMap(qualityLevel);
        // 获得角点
        getCorners(points, cornerMap);
    }

    // 用角点分布图得到特征点
    void getCorners(std::vector<cv::Point> &points, const cv::Mat &cornerMap)
    {
        // 迭代遍历像素，得到所有特征
        for (int y = 0; y < cornerMap.rows; ++y)
        {
            const uchar *rowPtr = cornerMap.ptr<uchar>(y);

            for (int x = 0; x < cornerMap.cols; ++x)
            {
                // 如果它是一个特征点
                if (rowPtr[x])
                {
                    points.push_back(cv::Point(x, y));
                }
            }
        }
    }

    // 用 cv::circle 函数画出检测到的特征点
    // 在特征点的位置画圆形
    void drawOnImage(cv::Mat &image,
                     const std::vector<cv::Point> &points,
                     cv::Scalar color = cv::Scalar(255, 255, 255),
                     int radius = 3,
                     int thickness = 1)
    {
        std::vector<cv::Point>::const_iterator it = points.begin();
        while (it != points.end())
        {
            cv::circle(image, *it, radius, color, thickness);
            ++it;
        }
    }
};

#endif // HARRIS_DETECTOR