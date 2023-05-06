/**
 * @brief Goal
 * In this tutorial will learn how to:
 * Apply two very common morphological operators: Erosion and Dilation.
 * For this purpose, will use the following OpenCV functions:
 * cv::erode and cv::dilate
 *
 * @file erosion_dilation.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-01
 * @version OpenCV 4.7 examples
 *
 */

#include <iostream>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

// global variables
cv::Mat src_img;
cv::Mat erosion_img;
cv::Mat dilation_img;

const char* erosion_title = "Image Erosion";
const char* dilation_title = "Image Dilation";

int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;

// function define
void erosionFunc(int, void *);
void dilationFunc(int, void *);

/**
 * @brief main function
 */
int main(int argc, const char **argv)
{
    cv::CommandLineParser parser( argc, argv,
                                "{@input | LinuxLogo.jpg | input image}" );
    src_img = cv::imread(cv::samples::findFile(parser.get<cv::String>("@input")),
                         cv::IMREAD_UNCHANGED);
    if (src_img.empty())
    {
        std::cout << "[Error] Could not open or find the image file\n\n";
        std::cout << "Usage: " << argv[0] << " <input image path>" << std::endl;
        return -1;
    }

    cv::namedWindow(erosion_title, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(dilation_title, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(dilation_title, src_img.cols, 0);

    /* Create a set of two Trackbars for each operation:
    The first trackbar "Element" returns either erosion_elem or dilation_elem;
    The second trackbar "Kernel size" return erosion_size or dilation_size
    for the corresponding operation. 
    -----------------------------------
    Every time we move any slider, the user's function Erosion or Dilation
    will be called and it will update the output image based on 
    the current trackbar values.
    ----------------------------------- */
    cv::createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse \n",
                       erosion_title,
                       &erosion_elem,
                       max_elem,
                       erosionFunc);
    cv::createTrackbar("Kernel size:\n 2n +1 \n",
                       erosion_title,
                       &erosion_size,
                       max_kernel_size,
                       erosionFunc);
    cv::createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse \n",
                       dilation_title,
                       &dilation_elem,
                       max_elem,
                       dilationFunc);
    cv::createTrackbar("Kernel size:\n 2n +1 \n",
                       dilation_title,
                       &dilation_size,
                       max_kernel_size,
                       dilationFunc);

    erosionFunc(0, 0);
    dilationFunc(0, 0);

    cv::waitKey(0);

    return 0;
}

/**
 * @brief erosoin callback function required by "cv::createTrackbar" function
 *     with such 'void Func(int, void*)' signature.
 * @param the global variable 'erosion_elem' 
 *     and mapping erosion_type with callback function to update value.
 * @param the global variable 'erosion_size' with callback function
 *     to update the value.
 */
void erosionFunc(int, void *)
{
    int erosion_type = 0;
    if (erosion_elem == 0)
    {
        erosion_type = cv::MORPH_RECT;
    }
    else if (erosion_elem == 1)
    {
        erosion_type = cv::MORPH_CROSS;
    }
    else if (erosion_elem == 2)
    {
        erosion_type = cv::MORPH_ELLIPSE;
    }

    cv::Mat element = cv::getStructuringElement(erosion_type,
                    cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                    cv::Point(erosion_size, erosion_size));

    cv::erode(src_img, erosion_img, element);
    cv::imshow(erosion_title, erosion_img);
}

/**
 * @brief dilation callback function required by "cv::createTrackbar" function
 *     with such 'void Func(int, void*)' signature.
 * @param the global variable 'dilation_elem' 
 *     and mapping erosion_type with callback function to update value.
 * @param the global variable 'dilation_size' with callback function
 *     to update the value.
 */
void dilationFunc(int, void *)
{
    int dilation_type = 0;
    if (dilation_elem == 0)
    {
        dilation_type = cv::MORPH_RECT;
    }
    else if (dilation_elem == 1)
    {
        dilation_type = cv::MORPH_CROSS;
    }
    else if (dilation_elem == 2)
    {
        dilation_type = cv::MORPH_ELLIPSE;
    }
    cv::Mat element = cv::getStructuringElement(dilation_type,
                    cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                    cv::Point(dilation_size, dilation_size));
    cv::dilate(src_img, dilation_img, element);
    cv::imshow(dilation_title, dilation_img);
}
