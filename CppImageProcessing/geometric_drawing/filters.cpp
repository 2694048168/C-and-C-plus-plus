/**
 * Goal
 * In this tutorial will learn how to apply diverse linear filters
 *  to smooth images using OpenCV functions such as:
 * cv::blur()
 * cv::GaussianBlur()
 * cv::medianBlur()
 * cv::bilateralFilter()
 *
 */

#include <iostream>
#include <cstdlib>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Global Variables
const int DELAY_CAPTION = 1500;
const int DELAY_BLUR = 100;
const int MAX_KERNEL_LENGTH = 31;

cv::Mat src_img;
cv::Mat dst_img;
const char *window_name = "Image Smoothing";

// Function headers
int display_caption(const char *caption);
int display_dst(const int delay);

/**
 * function main
 */
int main(int argc, char **argv)
{
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    const char *filename = argc >= 2 ? argv[1] : "lena.jpg";

    src_img = cv::imread(cv::samples::findFile(filename), cv::IMREAD_COLOR);
    if (src_img.empty())
    {
        std::cout << "[Error] opening the image file\n";
        std::cout << "Usage:\n " << argv[0]
                  << "[image_name -- default lena.jpg]" << std::endl;
        return EXIT_FAILURE;
    }

    if (display_caption("Original Image") != 0)
    {
        return 0;
    }

    // cv::Mat.clone to fully underlying copy!
    dst_img = src_img.clone();
    if (display_dst(DELAY_CAPTION) != 0)
    {
        return 0;
    }

    // Applying Homogeneous blur
    if (display_caption("Homogeneous Blur") != 0)
    {
        return 0;
    }
    for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
    {
        /* anchor point; default value cv::Point(-1,-1) means that
        the anchor is at the kernel[ cv::Size(w, h) ] center. */
        cv::blur(src_img, dst_img, cv::Size(i, i), cv::Point(-1, -1));
        if (display_dst(DELAY_BLUR) != 0)
        {
            return 0;
        }
    }

    // Applying Gaussian blur
    if (display_caption("Gaussian Blur") != 0)
    {
        return 0;
    }
    for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
    {
        /* kernel size 'w' and 'h' have to be odd and positive numbers
        otherwise the size will be calculated using the σx and σy arguments.
        σx(sigmaX): The standard deviation in x axes,
            Writing 0 implies that σx is calculated using kernel size;
        σy(sigmaY): The standard deviation in y axes,
            Writing 0 implies that σy is calculated using kernel size.
        --------------------------------------------------------------- */
        cv::GaussianBlur(src_img, dst_img, cv::Size(i, i), 0, 0);
        if (display_dst(DELAY_BLUR) != 0)
        {
            return 0;
        }
    }

    // Applying Median blur
    if (display_caption("Median Blur") != 0)
    {
        return 0;
    }
    for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
    {
        /* Size of the kernel must be odd,
        (only one because we use a square window). */
        cv::medianBlur(src_img, dst_img, i);
        if (display_dst(DELAY_BLUR) != 0)
        {
            return 0;
        }
    }

    // Applying Bilateral Filter
    if (display_caption("Bilateral Blur") != 0)
    {
        return 0;
    }
    for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
    {
        /* d: The diameter of each pixel neighborhood;
        sigmaColor: Standard deviation in the color space;
        sigmaSpace: Standard deviation in the coordinate space(in pixel terms)
        ------------------------------------------------------------------- */
        cv::bilateralFilter(src_img, dst_img, i, i * 2, i / 2);
        if (display_dst(DELAY_BLUR) != 0)
        {
            return 0;
        }
    }

    display_caption("Done!");

    return EXIT_SUCCESS;
}

/**
 * @function display_caption
 */
int display_caption(const char *caption)
{
    dst_img = cv::Mat::zeros(src_img.size(), src_img.type());
    putText(dst_img, caption,
            cv::Point(src_img.cols / 16, src_img.rows / 2),
            cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 255, 255));

    return display_dst(DELAY_CAPTION);
}

/**
 * @function display_dst
 */
int display_dst(const int delay)
{
    cv::imshow(window_name, dst_img);

    int c = cv::waitKey(delay);
    if (c >= 0)
    {
        return -1;
    }

    return 0;
}
