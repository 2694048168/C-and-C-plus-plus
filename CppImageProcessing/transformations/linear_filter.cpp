/**
 * @brief Goal
 * In this tutorial will learn how to:
 * 1. Use the OpenCV function filter2D() to create your own linear filters.
 *
 * 2. Use the OpenCV function copyMakeBorder() to set the borders
 *    (extra padding to your image).
 * cv::BORDER_CONSTANT: Pad the image with a constant value
 *    (i.e. black or 0
 * cv::BORDER_REPLICATE: The row or column at the very edge of the original
 *    is replicated to the extra border.
 *
 * @file linear_filter.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-03
 * @version OpenCV 4.7 examples
 *
 */

#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

/**
 * @brief main function
 */
int main(int argc, char const *argv[])
{
    const char *filename = argc >= 2 ? argv[1] : "cat.jpg";
    cv::Mat src_img = cv::imread(cv::samples::findFile(filename),
                                 cv::IMREAD_COLOR);
    if (src_img.empty())
    {
        std::cout << "Could not open or find the image!\n";
        std::cout << "Usage: " << argv[0]
                  << " image_path -- default lena.jpg" << std::endl;

        return EXIT_FAILURE;
    }
    cv::imshow("Original Image", src_img);

    cv::Mat kernel;
    unsigned int kernel_size = 3;
    cv::Point anchor = cv::Point(-1, -1); /* -1 means anchor in center */
    double delta = 0.0;
    /* −1 indicates that dst. image depth(channels) is same as src. image. */
    int ddepth = -1;
    const char *window_name = "Image Filter";
    cv::Mat dst_img;

    unsigned int ind = 0;
    while (true)
    {
        // Update kernel size for a normalized box filter.
        kernel_size = 3 + 2 * (ind % 5);
        kernel = cv::Mat::ones(cv::Size(kernel_size, kernel_size), CV_32F) / (float)(kernel_size * kernel_size);

        // Apply the linear filter to image.
        cv::filter2D(src_img, dst_img,
                     ddepth,
                     kernel,
                     anchor,
                     delta,
                     cv::BORDER_DEFAULT);
        cv::imshow(window_name, dst_img);

        char c = (char)cv::waitKey(500);
        // Press 'ESC' to exit the program.
        if (c == 27)
        {
            break;
        }
        ind++;
    }

    cv::waitKey(0);

    /* ------------------------------------------
        Brief how-to for this program
    ------------------------------------------ */
    printf("\n \t copyMakeBorder Demo: \n");
    printf("\t -------------------- \n");
    printf("[====] Press 'c' to set the border to a random constant value \n");
    printf("[====] Press 'r' to set the border to be replicated \n");
    printf("[====] Press 'ESC' to exit the program \n");

    const char *window_title = "copyMakeBorder";
    cv::RNG rng(42);

    const char *imageName = argc >= 2 ? argv[1] : "lena.jpg";
    cv::Mat src = cv::imread(cv::samples::findFile(imageName), cv::IMREAD_COLOR);
    if (src.empty())
    {
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default lena.jpg] \n");
        return -1;
    }

    cv::namedWindow(window_title, cv::WINDOW_AUTOSIZE);
    // Initialize arguments for the filter
    int top = (int)(0.05 * src.rows);
    int bottom = top;
    int left = (int)(0.05 * src.cols);
    int right = left;
    cv::Mat dst;
    int borderType = cv::BORDER_CONSTANT;
    for (;;)
    {
        cv::Scalar value(rng.uniform(0, 255),
                         rng.uniform(0, 255),
                         rng.uniform(0, 255));

        cv::copyMakeBorder(src, dst, top, bottom, left, right, borderType, value);
        cv::imshow(window_title, dst);
        char ch = (char)cv::waitKey(500); /* 0.5 seconds */
        if (ch == 27)
        {
            break;
        }
        else if (ch == 'c')
        {
            borderType = cv::BORDER_CONSTANT;
        }
        else if (ch == 'r')
        {
            borderType = cv::BORDER_REPLICATE;
        }
    }

    return EXIT_SUCCESS;
}
