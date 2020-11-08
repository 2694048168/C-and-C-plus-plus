/*
** showing how to detect edges using the Laplace operator
** 使用 拉普拉斯算子进行边缘检测
*/

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

int main(int argc, char **argv)
{
    //![variables]
    // Declare the variables we are going to use
    cv::Mat src, src_gray, dst;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    const char *window_name = "Laplace Demo";
    //![variables]

    //![load]
    const char *imageName = argc >= 2 ? argv[1] : "./../../image/lena.jpg";

    src = cv::imread(imageName, cv::IMREAD_COLOR); // Load an image

    // Check if image is loaded fine
    if (src.empty())
    {
        std::cout << " Error opening image" << std::endl;
        std::cout << " Program Arguments: [image_name -- default lena.jpg] " << std::endl;
        return -1;
    }
    //![load]

    //![reduce_noise]
    // Reduce noise by blurring with a Gaussian filter ( kernel size = 3 )
    cv::GaussianBlur(src, src, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
    //![reduce_noise]

    //![convert_to_gray]
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY); // Convert the image to grayscale
    //![convert_to_gray]

    /// Apply Laplace function
    cv::Mat abs_dst;
    //![laplacian]
    cv::Laplacian(src_gray, dst, ddepth, kernel_size, scale, delta, cv::BORDER_DEFAULT);
    //![laplacian]

    //![convert]
    // converting back to CV_8U
    cv::convertScaleAbs(dst, abs_dst);
    //![convert]

    //![display]
    cv::imshow(window_name, abs_dst);
    cv::waitKey(0);
    //![display]

    return 0;
}
