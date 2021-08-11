/**
 * @File    : BGFGSegmentor.hpp
 * @Brief   : 提取视频中的前景物体
 * @Author  : Wei Li
 * @Date    : 2021-08-11
*/

#ifndef BGFGSEG_H
#define BGFGSEG_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "videoprocessor.hpp"

class BGFGSegmentor : public FrameProcessor
{
private:
    cv::Mat gray;        // current gray-level image
    cv::Mat background;  // accumulated background
    cv::Mat backImage;   // current background image
    cv::Mat foreground;  // foreground image
    double learningRate; // learning rate in background accumulation
    int threshold;       // threshold for foreground extraction

public:
    BGFGSegmentor() : threshold(10), learningRate(0.01) {}

    // Set the threshold used to declare a foreground
    void setThreshold(int t)
    {

        threshold = t;
    }

    // Set the learning rate
    void setLearningRate(double r)
    {

        learningRate = r;
    }

    // processing method
    void process(cv::Mat &frame, cv::Mat &output)
    {

        // convert to gray-level image
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // initialize background to 1st frame
        if (background.empty())
            gray.convertTo(background, CV_32F);

        // convert background to 8U
        background.convertTo(backImage, CV_8U);

        // compute difference between current image and background
        cv::absdiff(backImage, gray, foreground);

        // apply threshold to foreground image
        cv::threshold(foreground, output, threshold, 255, cv::THRESH_BINARY_INV);

        // accumulate background
        cv::accumulateWeighted(gray, background,
                               // alpha*gray + (1-alpha)*background
                               learningRate, // alpha
                               output);      // mask
    }
};

#endif // BGFGSEG_H
