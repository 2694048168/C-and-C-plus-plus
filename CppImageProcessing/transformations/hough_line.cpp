/**
 * @brief Goal
 * In this tutorial will learn how to:
 * ---- Use the OpenCV functions HoughLines() and HoughLinesP()
 *      to detect lines in an image.
 *
 * OpenCV implements two kind of Hough Line Transforms:
 * a. The Standard Hough Transform[function cv::HoughLines()]
 *    It consists in pretty much what we just explained.
 *    It gives you as result a vector of couples (θ,rθ)
 *
 * b. The Probabilistic Hough Line Transform[function cv::HoughLinesP()]
 *    A more efficient implementation of the Hough Line Transform.
 *    It gives as output the extremes of the detected lines (x0,y0,x1,y1)
 *
 * @file hough_line.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-04
 * @version OpenCV 4.7 examples
 *
 */

#include <iostream>
#include <vector>

#include <opencv2/core.hpp> /* CV_PI */
#include <opencv2/core/fast_math.hpp> /* cvRound */
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
    const char *default_file = "sudoku.png";
    const char *filename = argc >= 2 ? argv[1] : default_file;
    cv::Mat src = cv::imread(cv::samples::findFile(filename),
                             cv::IMREAD_GRAYSCALE);
    if (src.empty())
    {
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default %s] \n", default_file);

        return EXIT_FAILURE;
    }

    // Edge detection
    cv::Mat dst;
    cv::Canny(src, dst, 50, 200, 3);
    // Copy edges to the images that will display the results in BGR
    cv::Mat cdst;
    cv::cvtColor(dst, cdst, cv::COLOR_GRAY2BGR);
    cv::Mat cdstP;
    cdstP = cdst.clone();

    // Standard Hough Line Transform
    std::vector<cv::Vec2f> lines; /* will hold the results of the detection */
    cv::HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0);
    // Draw the lines
    for (size_t i = 0; i < lines.size(); ++i)
    {
        float rho = lines[i][0], theta = lines[i][1];
        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));

        cv::line(cdst, pt1, pt2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    }

    // Probabilistic Line Transform
    std::vector<cv::Vec4i> linesP; /* will hold the results of the detection */
    HoughLinesP(dst, linesP, 1, CV_PI / 180, 50, 50, 10);
    // Draw the lines
    for (size_t i = 0; i < linesP.size(); i++)
    {
        cv::Vec4i l = linesP[i];
        cv::line(cdstP, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]),
                 cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    }

    cv::imshow("Source", src);
    cv::imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
    cv::imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);

    cv::waitKey(0);

    return EXIT_SUCCESS;
}
