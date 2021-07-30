/**
 * @File    : display_image.cpp
 * @Brief   : 显示图像
 * @Author  : Wei Li
 * @Date    : 2021-07-20
*/

#include <iostream>
#include <opencv2/opencv.hpp>


int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: Display Image with image path." << std::endl;
        return -1;
    }

    cv::Mat image;
    // argv[1] = "path2image"
    image = cv::imread(argv[1], 1);

    if (!image.data)
    {
        std::cout << "NO image data." << std::endl;
        return -1;
    }
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", image);

    cv::waitKey(0);
    
    return 0;
}
