/**
 * @brief Goal
 * In this tutorial will learn how to:
 * ---- Use the OpenCV function cv::Laplacian() to implement
 *      a discrete analog of the Laplacian operator.
 *     In fact, since the Laplacian uses the gradient of images,
 *     it calls internally the Sobel operator to perform its computation.
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
    // Declare the variables we are going to use
    cv::Mat src, src_gray, dst;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    const char *window_name = "Image Laplace";

    const char *imageName = argc >= 2 ? argv[1] : "lena.jpg";
    src = cv::imread(cv::samples::findFile(imageName), cv::IMREAD_COLOR);
    if (src.empty())
    {
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default lena.jpg] \n");

        return EXIT_FAILURE;
    }

    // Reduce noise by blurring with a Gaussian filter ( kernel size = 3 )
    cv::GaussianBlur(src, src, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

    cv::Mat abs_dst;
    cv::Laplacian(src_gray, dst, ddepth,
                  kernel_size, scale, delta, cv::BORDER_DEFAULT);
    // converting back to CV_8U
    cv::convertScaleAbs(dst, abs_dst);

    cv::imshow(window_name, abs_dst);
    cv::waitKey(0);

    return EXIT_SUCCESS;
}
