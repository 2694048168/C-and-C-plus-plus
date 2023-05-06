/**
 * @brief Goal
 * 1. use the OpenCV function cv::minAreaRect
 * 2. use the OpenCV function cv::fitEllipse
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
    const char *keys = "{@input | stuff.jpg | input image}";
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

    std::vector<cv::RotatedRect> minRect(contours.size());
    std::vector<cv::RotatedRect> minEllipse(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {
        minRect[i] = cv::minAreaRect(contours[i]);
        if (contours[i].size() > 5)
        {
            minEllipse[i] = cv::fitEllipse(contours[i]);
        }
    }

    cv::Mat drawing = cv::Mat::zeros(canny_output.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++)
    {
        cv::Scalar color = cv::Scalar(rng.uniform(0, 256),
                                      rng.uniform(0, 256),
                                      rng.uniform(0, 256));

        // contour
        cv::drawContours(drawing, contours, (int)i, color);

        // ellipse
        cv::ellipse(drawing, minEllipse[i], color, 2);

        // rotated rectangle
        cv::Point2f rect_points[4];
        minRect[i].points(rect_points);
        for (int j = 0; j < 4; j++)
        {
            cv::line(drawing, rect_points[j], rect_points[(j + 1) % 4], color);
        }
    }

    cv::imshow("Contours", drawing);
}
