/**
 * @brief Goal
 * In this tutorial will learn how to:
 * ---- What an image histogram is and why it is useful;
 * To equalize histograms of images by using the OpenCV function
 *  cv::equalizeHist to implement simple remapping routines.
 *
 * @file histogram_equalization.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-04
 * @version OpenCV 4.7 examples
 *
 */

#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

/**
 * @brief main function
 *
 * 1. Loads an image
 * 2. Convert the original image to grayscale
 * 3. Equalize the Histogram by using the OpenCV function cv::equalizeHist
 * 4. Display the source and equalized images in a window.
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
    cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);

    cv::Mat dst;
    cv::equalizeHist(src, dst);

    int const offset = 200;
    const char *window_src = "source image";
    const char *window_dst = "Equalized Image";
    cv::imshow(window_src, src);
    cv::moveWindow(window_src, offset, offset);
    cv::imshow(window_dst, dst);
    cv::moveWindow(window_dst, src.cols + offset, offset);

    cv::waitKey();
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}
