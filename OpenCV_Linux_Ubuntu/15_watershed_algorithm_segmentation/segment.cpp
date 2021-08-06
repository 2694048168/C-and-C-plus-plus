/**
 * @File    : segment.cpp
 * @Brief   : 用分水岭算法实现图像分割
 * @Author  : Wei Li
 * @Date    : 2021-07-29
*/

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "watershedSegmentation.hpp"

// -----------------------------
int main(int argc, char **argv)
{
    cv::Mat image = cv::imread("./../images/group.jpg");
    if (!image.data)
    {
        std::cerr << "--Error reading image file." << std::endl;
        return 1;
    }
    cv::namedWindow("Original Image");
    cv::imshow("Original Image", image);

    // Get the binary map
    cv::Mat binary;
    binary = cv::imread("./../images/binary.bmp", 0);
    if (!binary.data)
    {
        std::cerr << "--Error reading binary image file." << std::endl;
        return 1;
    }
    cv::namedWindow("Binary Image");
    cv::imshow("Binary Image", binary);

    // 消除噪声和很小的目标物体
    cv::Mat fg;
    cv::erode(binary, fg, cv::Mat(), cv::Point(-1, -1), 4);

    // Display the foreground image
    cv::namedWindow("Foreground Image");
    cv::imshow("Foreground Image", fg);

    // Identify image pixels without objects
    // 标识不含物体的图像像素
    cv::Mat bg;
    cv::dilate(binary, bg, cv::Mat(), cv::Point(-1, -1), 4);
    cv::threshold(bg, bg, 1, 128, cv::THRESH_BINARY_INV);

    // Display the background image
    cv::namedWindow("Background Image");
    cv::imshow("Background Image", bg);

    // 合并这两幅图像，得到标记图像
    // Show markers image
    cv::Mat markers(binary.size(), CV_8U, cv::Scalar(0));
    markers = fg + bg;
    cv::namedWindow("Markers");
    cv::imshow("Markers", markers);

    // 实例化对象 —— 利用分水岭算法分割图像
    WatershedSegmenter segmenter;
    segmenter.setMarkers(markers);
    segmenter.process(image);
    cv::namedWindow("Segmentation");
    cv::imshow("Segmentation", segmenter.getSegmentation());
    // Display watersheds
    cv::namedWindow("Watersheds");
    cv::imshow("Watersheds", segmenter.getWatersheds());

    // 换一副图像重新进行 分水岭算法进行分割图像
    image = cv::imread("./../images/tower.jpg");
    if (image.empty())
    {
        std::cerr << "--Error reading tower image file." << std::endl;
        return 1;
    }
    cv::namedWindow("Original Tower Image");
    cv::imshow("Original Tower Image", image);

    // 标记图像的前景和背景
    // Identify background pixels
    cv::Mat imageMask(image.size(), CV_8U, cv::Scalar(0));
    cv::rectangle(imageMask, cv::Point(5, 5), cv::Point(image.cols - 5, image.rows - 5), cv::Scalar(255), 3);
    // Identify foreground pixels (in the middle of the image)
    cv::rectangle(imageMask, cv::Point(image.cols / 2 - 10, image.rows / 2 - 10),
                  cv::Point(image.cols / 2 + 10, image.rows / 2 + 10), cv::Scalar(1), 10);

    segmenter.setMarkers(imageMask);
    segmenter.process(image);

    // 可视化结果
    // Display the image with markers
    cv::rectangle(image, cv::Point(5, 5), cv::Point(image.cols - 5, image.rows - 5), cv::Scalar(255, 255, 255), 3);
    cv::rectangle(image, cv::Point(image.cols / 2 - 10, image.rows / 2 - 10),
                  cv::Point(image.cols / 2 + 10, image.rows / 2 + 10), cv::Scalar(1, 1, 1), 10);
    cv::namedWindow("Image with marker");
    cv::imshow("Image with marker", image);

    // Display watersheds
    cv::namedWindow("Watershed");
    cv::imshow("Watershed", segmenter.getWatersheds());

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
