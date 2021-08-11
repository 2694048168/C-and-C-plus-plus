/**
 * @File    : flow.cpp
 * @Brief   : 估算光流
 * @Author  : Wei Li
 * @Date    : 2021-08-11
*/

#include <iostream>
#include <string>
#include <sstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/optflow.hpp>

#include "videoprocessor.hpp"

// Drawing optical flow vectors on an image
void drawOpticalFlow(const cv::Mat &oflow,    // the optical flow
                     cv::Mat &flowImage,      // the produced image
                     int stride,              // the stride for displaying the vectors
                     float scale,             // multiplying factor for the vectors
                     const cv::Scalar &color) // the color of the vectors
{
    // create the image if required
    if (flowImage.size() != oflow.size())
    {
        flowImage.create(oflow.size(), CV_8UC3);
        flowImage = cv::Vec3i(255, 255, 255);
    }

    // for all vectors using stride as a step
    for (int y = 0; y < oflow.rows; y += stride)
        for (int x = 0; x < oflow.cols; x += stride)
        {
            // gets the vector
            cv::Point2f vector = oflow.at<cv::Point2f>(y, x);
            // draw the line
            cv::line(flowImage, cv::Point(x, y),
                     cv::Point(static_cast<int>(x + scale * vector.x + 0.5),
                               static_cast<int>(y + scale * vector.y + 0.5)),
                     color);
            // draw the arrow tip
            cv::circle(flowImage, cv::Point(static_cast<int>(x + scale * vector.x + 0.5), static_cast<int>(y + scale * vector.y + 0.5)), 1, color, -1);
        }
}

// -------------------------------
int main(int argc, char **argv)
{
    // pick 2 frames of the sequence
    cv::Mat frame1 = cv::imread("./../images/goose/goose230.bmp", 0);
    cv::Mat frame2 = cv::imread("./../images/goose/goose237.bmp", 0);
    if (!frame1.data || !frame2.data)
    {
        std::cerr << "--Error reading image file unsuccessfully." << std::endl;
        return 1;
    }

    // Combined display
    cv::Mat combined(frame1.rows, frame1.cols + frame2.cols, CV_8U);
    frame1.copyTo(combined.colRange(0, frame1.cols));
    frame2.copyTo(combined.colRange(frame1.cols, frame1.cols + frame2.cols));
    cv::imshow("Frames", combined);

    // Create the optical flow algorithm
    cv::Ptr<cv::optflow::DualTVL1OpticalFlow> tvl1 = cv::optflow::createOptFlow_DualTVL1();

    std::cout << "regularization coeeficient: " << tvl1->getLambda() << std::endl;                                  // the smaller the soomther
    std::cout << "Number of scales: " << tvl1->getScalesNumber() << std::endl;                                      // number of scales
    std::cout << "Scale step: " << tvl1->getScaleStep() << std::endl;                                               // size between scales
    std::cout << "Number of warpings: " << tvl1->getWarpingsNumber() << std::endl;                                  // size between scales
    std::cout << "Stopping criteria: " << tvl1->getEpsilon() << " and " << tvl1->getOuterIterations() << std::endl; // size between scales
                                                                                                                    // compute the optical flow between 2 frames
    cv::Mat oflow;                                                                                                  // image of 2D flow vectors
    // compute optical flow between frame1 and frame2
    tvl1->calc(frame1, frame2, oflow);

    // Draw the optical flow image
    cv::Mat flowImage;
    drawOpticalFlow(oflow,                // input flow vectors
                    flowImage,            // image to be generated
                    8,                    // display vectors every 8 pixels
                    2,                    // multiply size of vectors by 2
                    cv::Scalar(0, 0, 0)); // vector color

    cv::imshow("Optical Flow", flowImage);

    // compute a smoother optical flow between 2 frames
    tvl1->setLambda(0.075);
    tvl1->calc(frame1, frame2, oflow);

    // Draw the optical flow image
    cv::Mat flowImage2;
    drawOpticalFlow(oflow,                // input flow vectors
                    flowImage2,           // image to be generated
                    8,                    // display vectors every 8 pixels
                    2,                    // multiply size of vectors by 2
                    cv::Scalar(0, 0, 0)); // vector color

    cv::imshow("Smoother Optical Flow", flowImage2);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
