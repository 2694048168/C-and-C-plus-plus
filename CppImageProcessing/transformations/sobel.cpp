/**
 * @brief Goal
 * In this tutorial will learn how to:
 * ---- Use the OpenCV function cv::Sobel() to calculate the derivatives
 *      from an image. Use the OpenCV function cv::Scharr() to calculate
 *      a more accurate derivative for a kernel of size 3x3.
 *
 * @file sobel.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-03
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
    std::cout << "The sample uses Sobel or Scharr OpenCV functions for edge detection\n\n";

    cv::CommandLineParser parser(argc, argv,
                                 "{@input   |lena.jpg|input image}"
                                 "{ksize   k|1|ksize (hit 'K' to increase its value at run time)}"
                                 "{scale   s|1|scale (hit 'S' to increase its value at run time)}"
                                 "{delta   d|0|delta (hit 'D' to increase its value at run time)}"
                                 "{help    h|false|show help message}");
    parser.printMessage();

    std::cout << "\nPress 'ESC' to exit program.\nPress 'R' to reset values ( ksize will be -1 equal to Scharr function )";

    cv::String imageName = parser.get<cv::String>("@input");
    cv::Mat image = cv::imread(cv::samples::findFile(imageName),
                               cv::IMREAD_COLOR);
    if (image.empty())
    {
        printf("Error opening image: %s\n", imageName.c_str());
        return EXIT_FAILURE;
    }

    // First we declare the variables we are going to use
    cv::Mat src, src_gray, grad;
    const char *window_name = "Sobel - Simple Edge Detector";
    int ksize = parser.get<int>("ksize");
    int scale = parser.get<int>("scale");
    int delta = parser.get<int>("delta");
    int ddepth = CV_16S;

    for (;;)
    {
        // Remove noise by blurring with a Gaussian filter(kernel size = 3)
        cv::GaussianBlur(image, src, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
        cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
        cv::Mat grad_x, grad_y;
        cv::Mat abs_grad_x, abs_grad_y;
        
        cv::Sobel(src_gray, grad_x, ddepth,
                  1, 0, ksize, scale, delta, cv::BORDER_DEFAULT);
        cv::Sobel(src_gray, grad_y, ddepth,
                  0, 1, ksize, scale, delta, cv::BORDER_DEFAULT);

        // converting back to CV_8U
        cv::convertScaleAbs(grad_x, abs_grad_x);
        cv::convertScaleAbs(grad_y, abs_grad_y);
        // Gradient for Sobel operator.
        cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

        cv::imshow(window_name, grad);
        char key = (char)cv::waitKey(0);
        if (key == 27)
        {
            return EXIT_SUCCESS;
        }
        if (key == 'k' || key == 'K')
        {
            ksize = ksize < 30 ? ksize + 2 : -1;
        }
        if (key == 's' || key == 'S')
        {
            scale++;
        }
        if (key == 'd' || key == 'D')
        {
            delta++;
        }
        if (key == 'r' || key == 'R')
        {
            scale = 1;
            ksize = -1;
            delta = 0;
        }
    }

    return EXIT_SUCCESS;
}
