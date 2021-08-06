/**
 * @File    : histogram.hpp
 * @Brief   : 计算灰度图的(单通道)的直方图
 * @Author  : Wei Li
 * @Date    : 2021-07-28
*/

#ifndef HISTOGRAM
#define HISTOGRAM

#include <opencv2/imgproc.hpp>

class Histogram1D
{
private:
    int histSize[1];        // 直方图中箱子的数量 num_bins
    float hranges[2];       // 数值范围
    const float *ranges[1]; // 数值范围的指针
    int channels[1];        // 要检查的通道数量

public:
    // 默认（无参）构造函数
    Histogram1D()
    {
        histSize[0] = 256; // 2^8 = 256
        hranges[0] = 0.0;
        hranges[1] = 256.0;
        ranges[0] = hranges; // 数组名即数组第一个元素的地址
        channels[0] = 0;     //只计算一个通道
    }

    // getter and setter methods
    void setChannel(int channel)
    {
        channels[0] = channel;
    }
    int getChannel() const
    {
        return channels[0];
    }

    void setRange(float minValue, float maxValue)
    {
        hranges[0] = minValue;
        hranges[1] = maxValue;
    }
    float getMinValue() const
    {
        return hranges[0];
    }
    float getMaxValue() const
    {
        return hranges[1];
    }

    void setNumBins(int num_bins)
    {
        histSize[0] = num_bins;
    }
    int getNumBins() const
    {
        return histSize[0];
    }

    // 计算单通道图像的直方图
    cv::Mat getHistogram(const cv::Mat &image)
    {
        cv::Mat hist;
        // 利用 OpenCV 函数直接计算
        cv::calcHist(&image, 1, // 仅为一幅图像的直方图
                     channels,  // 使用的通道
                     cv::Mat(), // 不使用掩码
                     hist,      // 作为结果的直方图
                     1,         // 这是一维的直方图
                     histSize,  // 箱子数量
                     ranges     // 像素值的范围
        );
        return hist;
    };

    // 图像直方图以可视化形式展示
    cv::Mat getHistogramImage(const cv::Mat &image, int zoom = 1)
    {
        cv::Mat hist = getHistogram(image);
        // 类的静态方法实现柱状图
        return Histogram1D::getImageOfHistogram(hist, zoom);
    }

    // Stretches the source image using min number of count in bins.
    // 对图像直方图进行变换拉伸，增加图像对比度
    cv::Mat stretch(const cv::Mat &image, int minValue = 0)
    {
        // 首先计算直方图
        cv::Mat hist = getHistogram(image);

        // 找到直方图中的最小值bins，即最左边
        int init_min_left = 0;
        for (; init_min_left < histSize[0]; ++init_min_left)
        {
            // ignore bins with less than minValue entries
            if (hist.at<float>(init_min_left) > minValue)
            {
                break;
            }
        }

        // 找到直方图最右边
        int init_max_right = histSize[0] - 1;
        for (; init_max_right >= 0; --init_max_right)
        {
            // ignore bins with less than minValue entries
            if (hist.at<float>(init_max_right) > minValue)
            {
                break;
            }
        }

        // 创建查找表
        int dims[1] = {256};
        cv::Mat lookup(1, dims, CV_8U);
        for (int i = 0; i < 256; ++i)
        {
            if (i < init_min_left)
            {
                lookup.at<uchar>(i) = 0;
            }
            else if (i > init_max_right)
            {
                lookup.at<uchar>(i) = 255;
            }
            else
            {
                lookup.at<uchar>(i) = cvRound(255.0 * (i - init_min_left) / (init_max_right - init_min_left));
            }
        }

        // 使用查找表 静态方法 applyLookUp
        cv::Mat result = applyLookUp(image, lookup);
        return result;
    }

    // Stretches the source image using percentile.
    /** 计算图像直方图
     * 图像由各种数值的像素构成。例如在单通道灰度图像中，每个像素都有一个 0（黑色） ~255（白色）的整数。
     * 对于每个灰度，都有不同数量的像素分布在图像内，具体取决于图片内容。
     * 直方图是一个简单的表格，表示一幅图像（有时是一组图像）中具有某个值的像素的数量。
     * 因此，灰度图像的直方图有 256 个项目，也叫箱子（ bin）。 
     * 0 号箱子提供值为 0 的像素的数量，1 号箱子提供值为 1 的像素的数量，以此类推。
     * 很明显，如果把直方图的所有箱子进行累加，得到的结果就是像素的总数。
     * 也可以把直方图归一化，即所有箱子的累加和等于 1，每个箱子的数值表示对应的像素数量占总数的百分比。
     * 
     * 通过伸展直方图，使它布满可用强度值的全部范围。
     * 这方法确实可以简单有效地提高图像质量，
     * 但很多时候，图像的视觉缺陷并不因为它使用的强度值范围太窄，
     * 而是因为部分强度值的使用频率远高于其他强度值
     */
    // 重载版本, 拉伸图像的直方图来增强图像对比度
    cv::Mat stretch(const cv::Mat &image, float percentile)
    {
        // number of pixels in percentile
        float number = image.total() * percentile;

        // Compute histogram first
        cv::Mat hist = getHistogram(image);

        // find left extremity of the histogram
        int imin = 0;
        for (float count = 0.0; imin < 256; imin++)
        {
            // number of pixel at imin and below must be > number
            if ((count += hist.at<float>(imin)) >= number)
                break;
        }

        // find right extremity of the histogram
        int imax = 255;
        for (float count = 0.0; imax >= 0; imax--)
        {
            // number of pixel at imax and below must be > number
            if ((count += hist.at<float>(imax)) >= number)
                break;
        }

        // Create lookup table
        int dims[1] = {256};
        cv::Mat lookup(1, dims, CV_8U);

        for (int i = 0; i < 256; i++)
        {

            if (i < imin)
                lookup.at<uchar>(i) = 0;
            else if (i > imax)
                lookup.at<uchar>(i) = 255;
            else
                lookup.at<uchar>(i) = cvRound(255.0 * (i - imin) / (imax - imin));
        }

        // Apply lookup table
        cv::Mat result;
        result = applyLookUp(image, lookup);

        return result;
    }

    // 静态方法
    // an image representing a histogram 柱状图
    static cv::Mat getImageOfHistogram(const cv::Mat &hist, int zoom)
    {
        // 取得箱子值的最大值和最小值
        double maxVal = 0;
        double minVal = 0;
        cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);

        // 取得直方图的大小
        int histSize = hist.rows;

        // 用于显示直方图的方形图像
        cv::Mat histImg(histSize * zoom, histSize * zoom, CV_8U, cv::Scalar(255));

        // 设置最高点为 90%（即图像高度）的箱子个数
        int hpt = static_cast<int>(0.9 * histSize);

        // 为每个箱子画垂直线
        for (int h = 0; h < histSize; ++h)
        {
            float binVal = hist.at<float>(h);
            if (binVal > 0)
            {
                int intensity = static_cast<int>(binVal * hpt / maxVal);
                cv::line(histImg, cv::Point(h * zoom, histSize * zoom),
                         cv::Point(h * zoom, (histSize - intensity) * zoom),
                         cv::Scalar(0), zoom);
            }
        }

        return histImg;
    }

    // 使用查找表 静态方法 applyLookUp
    // lookup is 1x256 matrix
    static cv::Mat applyLookUp(const cv::Mat &image, const cv::Mat &lookup)
    {
        cv::Mat result;
        // 直接调用 OpenCV 函数
        cv::LUT(image, lookup, result);
        return result;
    }
    // 利用 迭代器 进行实现
    static cv::Mat applyLookUpWithIterator(const cv::Mat &image, const cv::Mat &lookup)
    {
        cv::Mat result(image.rows, image.cols, CV_8U);
        cv::Mat_<uchar>::iterator iterator_begin_result = result.begin<uchar>();

        cv::Mat_<uchar>::const_iterator iterator_begin_input = image.begin<uchar>();
        cv::Mat_<uchar>::const_iterator iterator_end_input = image.end<uchar>();

        // 针对每一个像素进行应用查找表
        for (; iterator_begin_input != iterator_end_input; ++iterator_begin_input, ++iterator_begin_result)
        {
            *iterator_begin_result = lookup.at<uchar>(*iterator_begin_input);
        }

        return result;
    }

    /**直方图均衡化
     * 通过伸展直方图，使它布满可用强度值的全部范围。这方法确实可以简单有效地提高图像质量，
     * 但很多时候，图像的视觉缺陷并不因为它使用的强度值范围太窄，而是因为部分强度值的使用频率远高于其他强度值。
     * 
     * 如：中等灰度的强度值非常多，而较暗和较亮的像素值则非常稀少。
     * 均衡对所有像素强度值的使用频率可以作为提高图像质量的一种手段
     * 这正是直方图均衡化这一概念背后的思想，也就是让图像的直方图尽可能地平稳
     * 
     * 正确区分直方图均衡化和直方图归一化两个基本的概念
     * 图像的直方图计算； 直方图归一化； 直方图均衡化
     */
    // Equalizes the source image.
    static cv::Mat equalize(const cv::Mat &image)
    {
        cv::Mat result;
        cv::equalizeHist(image, result);

        return result;
    }
};

#endif // HISTOGRAM