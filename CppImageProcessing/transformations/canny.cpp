/**
 * @brief Goal
 * In this tutorial will learn how to:
 * ---- Use the OpenCV function cv::Canny to implement the Canny Edge Detector.
 *
 *  Also known to many as the optimal detector, the Canny algorithm
 *  aims to satisfy three main criteria:
 *    1. Low error rate: Meaning a good detection of only existent edges.
 *    2. Good localization: The distance between edge pixels detected
 *       and real edge pixels have to be minimized.
 *    3. Minimal response: Only one detector response per edge.
 *
 * @file laplacian.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-04
 * @version OpenCV 4.7 examples
 *
 */

#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

// global variable
cv::Mat src, src_gray;
cv::Mat dst, detected_edges;
int lowThreshold = 0;
const int max_lowThreshold = 100;
const int ratio = 3;
const int kernel_size = 3;
const char *window_name = "Edge Map";

static void CannyThreshold(int, void *);

/**
 * @brief main function
 */
int main(int argc, char const *argv[])
{
    cv::utils::logging::setLogLevel(
        cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
    /* ------------------------------------------
        Brief how-to for this program
    ------------------------------------------ */
    cv::CommandLineParser parser(argc, argv,
                                 "{@input | fruits.jpg | input image}");
    src = cv::imread(cv::samples::findFile(parser.get<cv::String>("@input")),
                     cv::IMREAD_COLOR);
    if (src.empty())
    {
        std::cout << "Could not open or find the image!\n"
                  << std::endl;
        std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;

        return EXIT_FAILURE;
    }

    dst.create(src.size(), src.type());
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    /* The variable to be controlled by the Trackbar is lowThreshold
     with a limit of max_lowThreshold (which we set to 100 previously);
     Each time the Trackbar registers an action,
     the callback function CannyThreshold will be invoked.
    -------------------------------------------------------- */
    cv::createTrackbar("Min threshold:",
                       window_name,
                       &lowThreshold,
                       max_lowThreshold,
                       CannyThreshold);

    CannyThreshold(0, 0);

    cv::waitKey(0);

    return EXIT_SUCCESS;
}

/**
 * @brief callback function for Canny algorithm via updating the threshold.
 * this function signature is required by cv::createTrackbar.
 */
static void CannyThreshold(int, void *)
{
    cv::blur(src_gray, detected_edges, cv::Size(3, 3));
    cv::Canny(detected_edges, detected_edges,
              lowThreshold, lowThreshold * ratio, kernel_size);

    dst = cv::Scalar::all(0);
    src.copyTo(dst, detected_edges);

    cv::imshow(window_name, dst);
}