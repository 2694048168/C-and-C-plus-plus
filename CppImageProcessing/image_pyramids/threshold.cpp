/**
 * @brief Goal
 * In this tutorial will learn how to:
 * Perform basic thresholding operations using OpenCV function
 *    cv::threshold provoide 5 types of Thresholding operations.
 *
 * @file threshold.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-03
 * @version OpenCV 4.7 examples
 *
 */

#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// global variable
int threshold_value = 0;
int threshold_type = 3;
const int max_value = 255; /* pixle value for 8bit image */
const int max_type = 4;   /* 5 types of Thresholding operations */
const int max_binary_value = 255;

cv::Mat src_img;
cv::Mat gray_img;
cv::Mat dst_img;
const char *window_name = "Image Threshold";

const char *tracker_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
const char *tracker_value = "Value";

// callback function signature for Tracker in OpenCV.
static void imageThreshold(int, void *);

/**
 * @brief main function
 */
int main(int argc, char const *argv[])
{
    const char *filename = argc >= 2 ? argv[1] : "stuff.jpg";
    cv::Mat src_img = cv::imread(cv::samples::findFile(filename),
                                 cv::IMREAD_COLOR);
    if (src_img.empty())
    {
        std::cout << "Could not open or find the image!\n";
        std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;

        return EXIT_FAILURE;
    }

    cv::cvtColor(src_img, gray_img, cv::COLOR_BGR2GRAY); /* grayscale image */
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    cv::createTrackbar(tracker_type,
                       window_name,
                       &threshold_type,
                       max_type,
                       imageThreshold);

    cv::createTrackbar(tracker_value,
                       window_name,
                       &threshold_value,
                       max_value,
                       imageThreshold);
                       

    imageThreshold(0, 0); /* Call the function to initialize */

    cv::waitKey(0);

    return EXIT_SUCCESS;
}

static void imageThreshold(int, void *)
{
    /* 5 types of Thresholding operations by cv::threshold function
        0: Binary
        1: Binary Inverted
        2: Threshold Truncated
        3: Threshold to Zero
        4: Threshold to Zero Inverted
    ------------------------------------ */
    cv::threshold(gray_img, dst_img,
                  threshold_value,
                  max_binary_value,
                  threshold_type);
                  
    cv::imshow(window_name, dst_img);
}
