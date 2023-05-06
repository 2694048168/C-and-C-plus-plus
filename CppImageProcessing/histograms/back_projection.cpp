/**
 * @brief Goal
 * In this tutorial will learn how to:
 * ---- What is Back Projection and why it is useful
 * ---- How to use the OpenCV function cv::calcBackProject
 *      to calculate Back Projection
 * ---- How to mix different channels of an image by using the
 *      OpenCV function cv::mixChannels
 *
 * @file back_projection.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-04
 * @version OpenCV 4.7 examples
 *
 */

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

/**
 * @brief main function
 *
 * 1. Loads an image
 * 2. Convert the original to HSV format and separate only Hue channel to be
 *    used for the Histogram (using the OpenCV function cv::mixChannels )
 * 3. Let the user to enter the number of bins to be used
 *    in the calculation of the histogram.
 * 4. Calculate the histogram (and update it if the bins change)
 *    and the backprojection of the same image.
 * 5. Display the backprojection and the histogram in windows.
 *
 */

// Global Variables
cv::Mat src, hsv, mask;
int low = 20, up = 20;
const char *window_image = "Source image";

// Function Headers
void Hist_and_Backproj();
void pickPoint(int event, int x, int y, int, void *);

int main(int argc, const char **argv)
{
    const char *filename = argc >= 2 ? argv[1] : "Back_Projection_Theory0.jpg";
    src = cv::imread(filename);
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    cv::namedWindow(window_image);
    cv::imshow(window_image, src);

    // Set Trackbars for floodfill thresholds
    cv::createTrackbar("Low thresh", window_image, &low, 255, 0);
    cv::createTrackbar("High thresh", window_image, &up, 255, 0);

    // Set a Mouse Callback
    cv::setMouseCallback(window_image, pickPoint, 0);

    cv::waitKey();

    return 0;
}

/**
 * @function pickPoint
 */
void pickPoint(int event, int x, int y, int, void *)
{
    if (event != cv::EVENT_LBUTTONDOWN)
    {
        return;
    }

    // Fill and get the mask
    cv::Point seed = cv::Point(x, y);

    int newMaskVal = 255;
    cv::Scalar newVal = cv::Scalar(120, 120, 120);

    int connectivity = 8;
    int flags = connectivity + (newMaskVal << 8) +
                cv::FLOODFILL_FIXED_RANGE + cv::FLOODFILL_MASK_ONLY;

    cv::Mat mask2 = cv::Mat::zeros(src.rows + 2, src.cols + 2, CV_8U);

    cv::floodFill(src, mask2, seed, newVal, 0,
                  cv::Scalar(low, low, low), cv::Scalar(up, up, up), flags);

    mask = mask2(cv::Range(1, mask2.rows - 1), cv::Range(1, mask2.cols - 1));

    cv::imshow("Mask", mask);

    Hist_and_Backproj();
}

/**
 * @function Hist_and_Backproj
 */
void Hist_and_Backproj()
{
    cv::Mat hist;
    int h_bins = 30;
    int s_bins = 32;
    int histSize[] = {h_bins, s_bins};

    float h_range[] = {0, 180};
    float s_range[] = {0, 256};
    const float *ranges[] = {h_range, s_range};

    int channels[] = {0, 1};

    // Get the Histogram and normalize it
    cv::calcHist(&hsv, 1, channels, mask, hist, 2, histSize, ranges, true, false);

    cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());

    // Get Backprojection
    cv::Mat backproj;
    cv::calcBackProject(&hsv, 1, channels, hist, backproj, ranges, 1, true);

    // Draw the backproj
    cv::imshow("BackProj", backproj);
}
