/**
 * @File    : contentFinder.hpp
 * @Brief   : 根据图像直方图属性查找图像特定内容； 反向投影直方图检测特定图像内容
 * @Author  : Wei Li
 * @Date    : 2021-07-28
*/

#ifndef CONTENTFINDER
#define CONTENTFINDER

#include <opencv2/imgproc.hpp>

class ContentFinder
{
private:
    // 直方图参数
    float hranges[2];
    const float *ranges[3];
    int channels[3];

    float threshold;   // 决策阈值 [0,1]
    cv::Mat histogram; // 直方图可能是稀疏的
    cv::SparseMat sparse_histogram;
    bool isSparse;

public:
    // 无参默认构造函数
    ContentFinder() : threshold(0.1f), isSparse(false)
    {
        // 所有通道数值范围相同
        ranges[0] = hranges;
        ranges[1] = hranges;
        ranges[2] = hranges;
    }

    // 该类的 get 和 set 方法
    void setThreshold(float threshold)
    {
        threshold = threshold;
    }
    float getThreshold() const
    {
        return threshold;
    }

    void setHistogram(const cv::Mat &hist)
    {
        isSparse = false;
        // 直方图归一化
        cv::normalize(hist, histogram, 1.0);
    }
    void setHistogram(const cv::SparseMat &hist)
    {
        isSparse = true;
        // 归一化距离选择： cv::NORM_MINMAX; cv::NORM_INF; cv::NORM_L1; cv::NORM_L2
        cv::normalize(hist, sparse_histogram, 1.0, cv::NORM_L2);
    }

    // 利用 直方图方向投影 查找指定内容的像素(概率上)
    cv::Mat find(const cv::Mat &image)
    {
        cv::Mat result;

        hranges[0] = 0.0;
        hranges[1] = 256.0;
        channels[0] = 0;
        channels[1] = 1;
        channels[2] = 2;

        return find(image, hranges[0], hranges[1], channels);
    }
    // 单通道处理，重载版本，便于可以直接处理三个通道的
    // Finds the pixels belonging to the histogram
    cv::Mat find(const cv::Mat &image, float minValue, float maxValue, int *channels)
    {
        cv::Mat result;

        hranges[0] = minValue;
        hranges[1] = maxValue;

        // call the right function based on histogram type
        // 根据 稀疏 调用正确的版本
        if (isSparse)
        {
            for (int i = 0; i < sparse_histogram.dims(); ++i)
            {
                // 直方图的维度数与通道列表一致
                this->channels[i] = channels[i];
            }

            // 反向投影 OpenCV 函数
            cv::calcBackProject(&image,
                                1,                // 只使用一幅图像
                                channels,         // 通道
                                sparse_histogram, // 直方图
                                result,           // 反向投影的图像
                                ranges,           // 每个维度的值范围
                                255.0             // 选用的换算系数 把概率值从 1 映射到 255
            );
        }
        else
        {
            for (int i = 0; i < histogram.dims; i++)
            {
                this->channels[i] = channels[i];
            }

            cv::calcBackProject(&image,
                                1,         // we only use one image at a time
                                channels,  // vector specifying what histogram dimensions belong to what image channels
                                histogram, // the histogram we are using
                                result,    // the resulting back projection image
                                ranges,    // the range of values, for each dimension
                                255.0      // the scaling factor is chosen such that a histogram value of 1 maps to 255
            );
        }

        // 对反向投影结果做阈值化，得到二值图像
        if (threshold > 0.0)
        {
            cv::threshold(result, result, 255.0*threshold, 255.0, cv::THRESH_BINARY);
        }

        return result;
    }
};

#endif // CONTENTFINDER