/**
 * @File    : colorhistogram.hpp
 * @Brief   : 计算彩色图像的(三通道)的直方图
 * @Author  : Wei Li
 * @Date    : 2021-07-28
*/

#ifndef COLHISTOGRAM
#define COLHISTOGRAM

#include <opencv2/imgproc.hpp>

class ColorHistogram
{
private:
    int histSize[3];        // size of each dimension
    float hranges[2];       // range of values (same for the 3 dimensions)
    const float *ranges[3]; // array of ranges for each dimension
    int channels[3];        // channel to be considered

public:
    // 无参默认构造函数
    ColorHistogram()
    {
        // Prepare default arguments for a color histogram
        // each dimension has equal size and range
        histSize[0] = histSize[1] = histSize[2] = 256;
        hranges[0] = 0.0; // BRG range from 0 to 256
        hranges[1] = 256.0;
        ranges[0] = hranges; // in this class,
        ranges[1] = hranges; // all channels have the same range
        ranges[2] = hranges;
        channels[0] = 0; // the three channels: B
        channels[1] = 1; // G
        channels[2] = 2; // R
    }

    // 类的 set and get 方法, 实现对数据的封装
    // set histogram size for each dimension
    void setSize(int size)
    {
        // each dimension has equal size
        histSize[0] = histSize[1] = histSize[2] = size;
    }

    // Computes the histogram.
    cv::Mat getHistogram(const cv::Mat &image)
    {
        cv::Mat hist;

        // BGR color histogram
        hranges[0] = 0.0; // BRG range
        hranges[1] = 256.0;
        channels[0] = 0; // the three channels
        channels[1] = 1;
        channels[2] = 2;

        // Compute histogram
        cv::calcHist(&image,
                     1,         // histogram of 1 image only
                     channels,  // the channel used
                     cv::Mat(), // no mask is used
                     hist,      // the resulting histogram
                     3,         // it is a 3D histogram
                     histSize,  // number of bins
                     ranges     // pixel value range
        );

        return hist;
    }

    // Computes the histogram.
    cv::SparseMat getSparseHistogram(const cv::Mat &image)
    {
        cv::SparseMat hist(3,        // number of dimensions
                           histSize, // size of each dimension
                           CV_32F);

        // BGR color histogram
        hranges[0] = 0.0; // BRG range
        hranges[1] = 256.0;
        channels[0] = 0; // the three channels
        channels[1] = 1;
        channels[2] = 2;

        // Compute histogram
        cv::calcHist(&image,
                     1,         // histogram of 1 image only
                     channels,  // the channel used
                     cv::Mat(), // no mask is used
                     hist,      // the resulting histogram
                     3,         // it is a 3D histogram
                     histSize,  // number of bins
                     ranges     // pixel value range
        );

        return hist;
    }

    // Computes the 1D Hue histogram.
    // BGR source image is converted to HSV
    // Pixels with low saturation are ignored
    cv::Mat getHueHistogram(const cv::Mat &image, int minSaturation = 0)
    {
        cv::Mat hist;

        // Convert to HSV colour space
        cv::Mat hsv;
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

        // Mask to be used (or not)
        cv::Mat mask;
        // creating the mask if required
        if (minSaturation > 0)
        {

            // Spliting the 3 channels into 3 images
            std::vector<cv::Mat> v;
            cv::split(hsv, v);

            // Mask out the low saturated pixels
            cv::threshold(v[1], mask, minSaturation, 255, cv::THRESH_BINARY);
        }

        // Prepare arguments for a 1D hue histogram
        hranges[0] = 0.0; // range is from 0 to 180
        hranges[1] = 180.0;
        channels[0] = 0; // the hue channel

        // Compute histogram
        cv::calcHist(&hsv,
                     1,        // histogram of 1 image only
                     channels, // the channel used
                     mask,     // binary mask
                     hist,     // the resulting histogram
                     1,        // it is a 1D histogram
                     histSize, // number of bins
                     ranges    // pixel value range
        );

        return hist;
    }

    // Computes the 2D ab histogram.
    // BGR source image is converted to Lab
    cv::Mat getabHistogram(const cv::Mat &image)
    {

        cv::Mat hist;

        // Convert to Lab color space
        cv::Mat lab;
        cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);

        // Prepare arguments for a 2D color histogram
        hranges[0] = 0;
        hranges[1] = 256.0;
        channels[0] = 1; // the two channels used are ab
        channels[1] = 2;

        // Compute histogram
        cv::calcHist(&lab,
                     1,         // histogram of 1 image only
                     channels,  // the channel used
                     cv::Mat(), // no mask is used
                     hist,      // the resulting histogram
                     2,         // it is a 2D histogram
                     histSize,  // number of bins
                     ranges     // pixel value range
        );

        return hist;
    }
};

#endif