/**
 * @brief Goal
 * In this tutorial will learn how to:
 * ---- Use the OpenCV function HoughCircles() to detect circles in an image.
 *
 * @file hough_circle.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-04
 * @version OpenCV 4.7 examples
 *
 */

#include <iostream>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

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
    const char *filename = argc >= 2 ? argv[1] : "smarties.png";
    cv::Mat src = cv::imread(cv::samples::findFile(filename), cv::IMREAD_COLOR);
    if (src.empty())
    {
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default %s] \n", filename);

        return EXIT_FAILURE;
    }

    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    cv::medianBlur(gray, gray, 5);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1,
                     gray.rows / 16, /* change this value to detect circles with different distances to each other */
                     100, 30, 1, 30  /* change the last two parameters */
                                     /* (min_radius & max_radius) to detect larger circles */
    );
    for (size_t i = 0; i < circles.size(); ++i)
    {
        cv::Vec3i c = circles[i];
        cv::Point center = cv::Point(c[0], c[1]);
        // circle center
        cv::circle(src, center, 1, cv::Scalar(0, 100, 100), 3, cv::LINE_AA);
        // circle outline
        int radius = c[2];
        cv::circle(src, center, radius, cv::Scalar(255, 0, 255), 3, cv::LINE_AA);
    }

    cv::imshow("detected circles", src);

    cv::waitKey(0);

    return EXIT_SUCCESS;
}
