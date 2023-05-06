/**
 * @brief Goal
 * 1. use the OpenCV function cv::moments
 * 2. use the OpenCV function cv::contourArea
 * 3. use the OpenCV function cv::arcLength
 *
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-05
 * @version Samples of OpenCV 4.7
 *
 */

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <iostream>
#include <vector>
#include <iomanip>

// global variables
cv::Mat src_gray_img;
int threshold_value = 100;
cv::RNG rng(42);

void threshold_callback(int, void *);

/**
 * @brief main function and the programm entry.
 */
int main(int argc, char const *argv[])
{
    cv::utils::logging::setLogLevel(
        cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
    /* -------------------------------------------------- */
    const char *keys = "{@input | Moments_Source_Image.jpg | input image}";
    cv::CommandLineParser parser(argc, argv, keys);
    cv::Mat src_img = cv::imread(cv::samples::findFile(
        parser.get<cv::String>("@input")));
    if (src_img.empty())
    {
        std::cout << "[Error] Could not open or find the image.\n"
                  << std::endl;
        std::cout << "Usage: " << argv[0] << " <input image>" << std::endl;
        return EXIT_FAILURE;
    }

    cv::cvtColor(src_img, src_gray_img, cv::COLOR_BGR2GRAY);
    cv::blur(src_gray_img, src_gray_img, cv::Size(3, 3));

    const char *source_window = "Source";
    cv::namedWindow(source_window);
    cv::imshow(source_window, src_img);

    const int max_threshold = 255;
    cv::createTrackbar("Canny threshold:", source_window,
                       &threshold_value, max_threshold, threshold_callback);

    threshold_callback(0, 0);

    cv::waitKey(0);

    return 0;
}

void threshold_callback(int, void *)
{
    cv::Mat canny_output;
    cv::Canny(src_gray_img, canny_output, threshold_value, threshold_value * 2);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(canny_output, contours,
                     cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE,
                     cv::Point(0, 0));

    std::vector<cv::Moments> mu_img(contours.size());
    for (size_t i = 0; i < contours.size(); ++i)
    {
        mu_img[i] = cv::moments(contours[i]);
    }

    std::vector<cv::Point2f> mc_img(contours.size());
    for (size_t i = 0; i < contours.size(); ++i)
    {
        // add 1e-5 to avoid division by zero
        mc_img[i] = cv::Point2f(static_cast<float>(mu_img[i].m10 /
                                                   (mu_img[i].m00 + 1e-5)),
                                static_cast<float>(mu_img[i].m01 /
                                                   (mu_img[i].m00 + 1e-5)));

        std::cout << "mc_img[" << i << "] = " << mc_img[i] << "\n";
    }

    cv::Mat drawing = cv::Mat::zeros(canny_output.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++)
    {
        cv::Scalar color = cv::Scalar(rng.uniform(0, 256),
                                      rng.uniform(0, 256),
                                      rng.uniform(0, 256));

        cv::drawContours(drawing, contours, (int)i, color, 2);

        cv::circle(drawing, mc_img[i], 4, color, -1);
    }
    cv::imshow("Contours", drawing);

    std::cout << "\t Info: Area and Contour Length \n";
    for (size_t i = 0; i < contours.size(); i++)
    {
        std::cout << " * Contour[" << i << "] - Area (M_00) = "
                  << std::fixed << std::setprecision(2) << mu_img[i].m00
                  << " - Area OpenCV: " << contourArea(contours[i])
                  << " - Length: " << arcLength(contours[i], true)
                  << "\n";
    }
}
