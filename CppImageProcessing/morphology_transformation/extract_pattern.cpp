/**
 * @brief Goal
 * In this tutorial will learn how to:
 * Apply two very common morphology operators (i.e. Dilation and Erosion),
 * with the creation of custom kernels, in order to extract straight lines
 * on the horizontal and vertical axes.
 * For this purpose, will use the following OpenCV functions:
 * cv::erode() | cv::dilate() | cv::getStructuringElement(),
 *
 * An example where your goal will be to extract
 *  the music notes from a music sheet.
 *
 * @file extract_pattern.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-02
 * @version OpenCV 4.7 examples
 *
 */

#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void show_wait_destroy(const char *winname, cv::Mat &img);

int main(int argc, char const *argv[])
{
    cv::CommandLineParser parser(argc, argv, "{@input | notes.png | input image}");
    cv::Mat src = cv::imread(cv::samples::findFile(parser.get<cv::String>("@input")), cv::IMREAD_COLOR);
    if (src.empty())
    {
        std::cout << "Could not open or find the image!\n";
        std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;

        return -1;
    }
    // Show source image
    cv::imshow("src", src);
    
    // Transform source image to gray if it is not already
    cv::Mat gray;
    if (src.channels() == 3)
    {
        cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = src;
    }
    // Show gray image
    show_wait_destroy("gray", gray);

    // Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    cv::Mat bw;
    cv::adaptiveThreshold(~gray, bw, 255,
                          cv::ADAPTIVE_THRESH_MEAN_C,
                          cv::THRESH_BINARY,
                          15,
                          -2);
    // Show binary image
    show_wait_destroy("binary", bw);
    
    /* Create the images that will use to extract
     the horizontal and vertical lines */
    cv::Mat horizontal = bw.clone();
    cv::Mat vertical = bw.clone();

    /* 1. Specify size on horizontal axis;
       2. Create structure element for extracting horizontal lines
          through morphology operations;
       3. Apply morphology operations;
       4. Show extracted horizontal lines
    ----------------------------------------- */
    int horizontal_size = horizontal.cols / 30;
    cv::Mat horizontalStructure = cv::getStructuringElement(cv::MORPH_RECT,
                                        cv::Size(horizontal_size, 1));
    cv::erode(horizontal, horizontal, horizontalStructure, cv::Point(-1, -1));
    cv::dilate(horizontal, horizontal, horizontalStructure, cv::Point(-1, -1));
    show_wait_destroy("horizontal", horizontal);

    /* 1. Specify size on vertical axis;
       2. Create structure element for extracting vertical lines
          through morphology operations;
       3. Apply morphology operations;
       4. Show extracted vertical lines;
       5. Inverse vertical image;
    ---------------------------------------- */
    int vertical_size = vertical.rows / 30;
    cv::Mat verticalStructure = cv::getStructuringElement(cv::MORPH_RECT,
                                        cv::Size(1, vertical_size));
    cv::erode(vertical, vertical, verticalStructure, cv::Point(-1, -1));
    cv::dilate(vertical, vertical, verticalStructure, cv::Point(-1, -1));
    show_wait_destroy("vertical", vertical);
    // Inverse vertical image
    bitwise_not(vertical, vertical);
    show_wait_destroy("vertical_bit", vertical);

    /* Extract edges and smooth image according to the logic
        1. extract edges
        2. dilate(edges)
        3. src.copyTo(smooth)
        4. blur smooth img
        5. smooth.copyTo(src, edges)
    ---------------------------------- */
    cv::Mat edges;
    cv::adaptiveThreshold(vertical, edges, 255,
                          cv::ADAPTIVE_THRESH_MEAN_C,
                          cv::THRESH_BINARY,
                          3,
                          -2);
    show_wait_destroy("edges", edges);
    cv::Mat kernel = cv::Mat::ones(2, 2, CV_8UC1);
    cv::dilate(edges, edges, kernel);
    show_wait_destroy("dilate", edges);
    cv::Mat smooth;
    vertical.copyTo(smooth);
    cv::blur(smooth, smooth, cv::Size(2, 2));
    smooth.copyTo(vertical, edges);
    show_wait_destroy("smooth - final", vertical);

    cv::destroyAllWindows();

    return 0;
}

void show_wait_destroy(const char *winname, cv::Mat &img)
{
    cv::imshow(winname, img);
    cv::moveWindow(winname, 300, 0);
    cv::waitKey(0);
}
