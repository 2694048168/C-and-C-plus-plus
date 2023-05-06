/**
 * @brief Goal
 * In this tutorial will learn how to:
 * ---- Use the OpenCV function cv::split to divide
 *      an image into its correspondent planes.
 *
 * ---- To calculate histograms of arrays of images by using
 *      the OpenCV function cv::calcHist
 *
 * ---- To normalize an array by using the function cv::normalize
 *
 * @file histogram_calculation.cpp
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
 * 2. Splits the image into its R, G and B planes using the function cv::split
 * 3. Calculate the Histogram of each 1-channel plane
 *    by calling the function cv::calcHist
 * 4. Plot the three histograms in a window
 */
int main(int argc, char const **argv)
{
    cv::utils::logging::setLogLevel(
        cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
    /* ------------------------------------------
        Brief how-to for this program
    ------------------------------------------ */
    cv::CommandLineParser parser(argc, argv,
                                 "{@image | lena.jpg | input image name}");
    std::string filename = parser.get<std::string>(0);
    cv::Mat src = cv::imread(cv::samples::findFile(filename),
                             cv::IMREAD_COLOR);
    if (src.empty())
    {
        std::cout << "Cannot read image: " << filename << std::endl;
        std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
        return -1;
    }

    std::vector<cv::Mat> bgr_planes;
    cv::split(src, bgr_planes);

    /* ------------------------ */
    int histSize = 256;
    float range[] = {0, 256}; /* the upper boundary is exclusive */
    const float *histRange[] = {range};

    bool uniform = true;
    bool accumulate = false;

    cv::Mat b_hist, g_hist, r_hist;
    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1,
                 &histSize, histRange, uniform, accumulate);
    cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1,
                 &histSize, histRange, uniform, accumulate);
    cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1,
                 &histSize, histRange, uniform, accumulate);

    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::normalize(b_hist, b_hist, 0, histImage.rows,
                  cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(g_hist, g_hist, 0, histImage.rows,
                  cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(r_hist, r_hist, 0, histImage.rows,
                  cv::NORM_MINMAX, -1, cv::Mat());

    for (int i = 1; i < histSize; ++i)
    {
        cv::line(histImage,
                 cv::Point(bin_w * (i - 1),
                           hist_h - cvRound(b_hist.at<float>(i - 1))),
                 cv::Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
                 cv::Scalar(255, 0, 0), 2, 8, 0);

        cv::line(histImage,
                 cv::Point(bin_w * (i - 1),
                           hist_h - cvRound(g_hist.at<float>(i - 1))),
                 cv::Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
                 cv::Scalar(0, 255, 0), 2, 8, 0);

        cv::line(histImage,
                 cv::Point(bin_w * (i - 1),
                           hist_h - cvRound(r_hist.at<float>(i - 1))),
                 cv::Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
                 cv::Scalar(0, 0, 255), 2, 8, 0);
    }

    int const offset = 200;
    const char *window_src = "source image";
    const char *window_dst = "calcHist for Image";
    cv::imshow(window_src, src);
    cv::moveWindow(window_src, offset, offset);
    cv::imshow(window_dst, histImage );
    cv::moveWindow(window_dst, src.cols + offset, offset);

    cv::waitKey();
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}
