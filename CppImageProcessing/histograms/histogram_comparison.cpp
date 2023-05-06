/**
 * @brief Goal
 * In this tutorial will learn how to:
 * ---- Use the function cv::compareHist to get a numerical parameter
 *      that express how well two histograms match with each other.
 *
 * ---- Use different metrics to compare histograms:
 *     1. Correlation ( CV_COMP_CORREL )
 *     2. Chi-Square ( CV_COMP_CHISQR )
 *     3. Intersection ( method=CV_COMP_INTERSECT )
 *     4. Bhattacharyya distance ( CV_COMP_BHATTACHARYYA ）
 *
 * @file histogram_comparison.cpp
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
 * 1. Loads a base image and 2 test images to be compared with it.
 * 2. Generate 1 image that is the lower half of the base image
 * 3. Convert the images to HSV format
 * 4. Calculate the H-S histogram for all the images and normalize them
 *    in order to compare them.
 * 5. Compare the histogram of the base image with respect to the 2 test
 *    histograms, the histogram of the lower half base image
 *    and with the same base image histogram.
 * 6. Display the numerical matching parameters obtained.
 *
 */
int main(int argc, char const **argv)
{
    cv::utils::logging::setLogLevel(
        cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
    /* ------------------------------------------
        Brief how-to for this program
    ------------------------------------------ */
    const char *keys =
        "{ help  h| | Print help message. }"
        "{ @input1 |Histogram_Comparison_0.jpg | Path to input image 1. }"
        "{ @input2 |Histogram_Comparison_1.jpg | Path to input image 2. }"
        "{ @input3 |Histogram_Comparison_2.jpg | Path to input image 3. }";
    cv::CommandLineParser parser(argc, argv, keys);
    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }

    cv::Mat src_base = cv::imread(parser.get<cv::String>("@input1"));
    cv::Mat src_test1 = cv::imread(parser.get<cv::String>("@input2"));
    cv::Mat src_test2 = cv::imread(parser.get<cv::String>("@input3"));
    if (src_base.empty() || src_test1.empty() || src_test2.empty())
    {
        std::cout << "Could not open or find the images!\n"
                  << std::endl;
        parser.printMessage();
        return -1;
    }

    cv::Mat hsv_base, hsv_test1, hsv_test2;
    cv::cvtColor(src_base, hsv_base, cv::COLOR_BGR2HSV);
    cv::cvtColor(src_test1, hsv_test1, cv::COLOR_BGR2HSV);
    cv::cvtColor(src_test2, hsv_test2, cv::COLOR_BGR2HSV);

    /* ------------------------------------------------------- */
    cv::Mat hsv_half_down = hsv_base(cv::Range(hsv_base.rows / 2, hsv_base.rows),
                                     cv::Range(0, hsv_base.cols));

    int h_bins = 50, s_bins = 60;
    int histSize[] = {h_bins, s_bins};

    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = {0, 180};
    float s_ranges[] = {0, 256};

    const float *ranges[] = {h_ranges, s_ranges};

    // Use the 0-th and 1-st channels
    int channels[] = {0, 1};
    /* ------------------------------------------------------- */
    cv::Mat hist_base, hist_half_down, hist_test1, hist_test2;

    cv::calcHist(&hsv_base, 1, channels, cv::Mat(), hist_base, 2,
                 histSize, ranges, true, false);
    cv::normalize(hist_base, hist_base, 0, 1,
                  cv::NORM_MINMAX, -1, cv::Mat());

    cv::calcHist(&hsv_half_down, 1, channels, cv::Mat(), hist_half_down, 2,
                 histSize, ranges, true, false);
    cv::normalize(hist_half_down, hist_half_down, 0, 1,
                  cv::NORM_MINMAX, -1, cv::Mat());

    cv::calcHist(&hsv_test1, 1, channels, cv::Mat(), hist_test1, 2,
                 histSize, ranges, true, false);
    cv::normalize(hist_test1, hist_test1, 0, 1,
                  cv::NORM_MINMAX, -1, cv::Mat());

    calcHist(&hsv_test2, 1, channels, cv::Mat(), hist_test2, 2,
             histSize, ranges, true, false);
    normalize(hist_test2, hist_test2, 0, 1,
              cv::NORM_MINMAX, -1, cv::Mat());

    /* -------------------------------------------------------------- */
    for (int compare_method = 0; compare_method < 4; compare_method++)
    {
        double base_base = cv::compareHist(hist_base, hist_base, compare_method);
        double base_half = cv::compareHist(hist_base, hist_half_down, compare_method);
        double base_test1 = cv::compareHist(hist_base, hist_test1, compare_method);
        double base_test2 = cv::compareHist(hist_base, hist_test2, compare_method);

        std::cout << "Method " << compare_method
                  << " Perfect, Base-Half, Base-Test(1), Base-Test(2) : "
                  << base_base << " / "
                  << base_half << " / "
                  << base_test1 << " / "
                  << base_test2 << std::endl;
    }
    std::cout << "Done \n";

    return EXIT_SUCCESS;
}
