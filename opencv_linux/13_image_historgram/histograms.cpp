/**
 * @File    : histograms.cpp
 * @Brief   : 图像直方图计算； 直方图归一化； 直方图均衡化
 * @Author  : Wei Li
 * @Date    : 2021-07-28
*/

// 首先引入 C++ 标准库头文件
#include <iostream>
// 其次引入第三方头文件
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
// 最后引入自己编写的项目头文件
#include "histogram.hpp"

// -------------------------------
int main(int argc, char **argv)
{
    cv::Mat image = cv::imread("./../images/group.jpg", 0);
    if (!image.data)
    {
        std::cerr << "Error reading image file." << std::endl;
        return 1;
    }

    cv::imwrite("groupBW.jpg", image);
    cv::namedWindow("GrayImage");
    cv::imshow("GrayImage", image);

    // 实例化对象，计算图像直方图
    Histogram1D h;
    cv::Mat hist0 = h.getHistogram(image);
    // loop over each bin
    for (int i = 0; i < 256; ++i)
    {
        std::cout << "Gray value " << i << " = " << hist0.at<float>(i) << std::endl;
    }
    // 可视化直方图的计算
    cv::namedWindow("Histogram");
    cv::imshow("Histogram", h.getHistogramImage(image));

    // 利用阈值进行选择
    // re-display the histagram with chosen threshold indicated
    cv::Mat hi = h.getHistogramImage(image);
    cv::line(hi, cv::Point(70, 0), cv::Point(70, 255), cv::Scalar(128));
    cv::namedWindow("Histogram with threshold value");
    cv::imshow("Histogram with threshold value", hi);

    // creating a binary image by thresholding at the valley
    cv::Mat thresholded; // output binary image
    cv::threshold(image, thresholded,
                  70,                 // threshold value
                  255,                // value assigned to pixels over threshold value
                  cv::THRESH_BINARY); // thresholding type

    // Display the thresholded image
    cv::namedWindow("Binary Image");
    cv::imshow("Binary Image", thresholded);
    thresholded = 255 - thresholded;
    cv::imwrite("binary.bmp", thresholded);

    // 直方图均衡化
    cv::Mat eq = h.equalize(image);
    cv::namedWindow("Equalized Image");
    cv::imshow("Equalized Image", eq);

    // Show the new histogram
    cv::namedWindow("Equalized H");
    cv::imshow("Equalized H", h.getHistogramImage(eq));

    // Stretch the image, setting the 1% of pixels at black and 1% at white
    cv::Mat str = h.stretch(image, 0.01f);

    // Show the result
    cv::namedWindow("Stretched Image");
    cv::imshow("Stretched Image", str);

    // Show the new histogram
    cv::namedWindow("Stretched H");
    cv::imshow("Stretched H", h.getHistogramImage(str));

    // Create an image inversion table
    cv::Mat lut(1, 256, CV_8U); // 1x256 matrix

    // Or:
    // int dim(256);
    // cv::Mat lut(1,  // 1 dimension
    // 	&dim,          // 256 entries
    //	CV_8U);        // uchar

    for (int i = 0; i < 256; i++)
    {
        // 0 becomes 255, 1 becomes 254, etc.
        lut.at<uchar>(i) = 255 - i;
    }

    // Apply lookup and display negative image
    cv::namedWindow("Negative image");
    cv::imshow("Negative image", h.applyLookUp(image, lut));

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
