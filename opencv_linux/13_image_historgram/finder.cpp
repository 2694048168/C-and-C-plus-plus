/**
 * @File    : finder.cpp
 * @Brief   : 均值平移算法查找目标 mean-shift algorithm
 * @Author  : Wei Li
 * @Date    : 2021-07-29
*/

#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

#include "colorhistogram.hpp"
#include "contentFinder.hpp"

// --------------------------------
int main(int argc, char **argv)
{
    cv::Mat image = cv::imread("./../images/baboon01.jpg");
    if (!image.data)
    {
        std::cerr << "--Error reading image file." << std::endl;
        return 1;
    }

    // 初始化定位窗口位置
    cv::Rect rect(110, 45, 35, 45);
    cv::rectangle(image, rect, cv::Scalar(0, 0, 255));
    cv::namedWindow("Image 1");
    cv::imshow("Image 1", image);

    // Baboon's face ROI
    cv::Mat imageROI = image(rect);

    // 在 HSV 色彩空间中计算 狒狒脸部有非常独特的粉红色，使用像素的色调很容易标识狒狒脸部
    int minSat = 65; // 考虑饱和度的影响，设置阈值进行忽略
    ColorHistogram hc;
    cv::Mat colorhist = hc.getHueHistogram(imageROI, minSat);

    ContentFinder finder;
    finder.setHistogram(colorhist);
    finder.setThreshold(0.2f);

    // Convert to HSV space ,just for display
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    // 切分三个分量，存储在 vector 中三幅图像
    std::vector<cv::Mat> v;
    cv::split(hsv, v);

    // 消除低饱和度的像素
    cv::threshold(v[1], v[1], minSat, 255, cv::THRESH_BINARY);
    cv::namedWindow("Saturation mask");
    cv::imshow("Saturation mask", v[1]);

    // 利用反向投影直方图和移动均值算法进行目标检测
    image = cv::imread("./../images/baboon02.jpg");
    if (!image.data)
    {
        std::cerr << "--Error readimg image file." << std::endl;
        return -1;
    }
    cv::namedWindow("Image 2");
    cv::imshow("Image 2", image);
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // Get back-projection of hue histogram
    int ch[1] = {0};
    finder.setThreshold(-1.0f); // no thresholding
    cv::Mat result = finder.find(hsv, 0.0f, 180.0f, ch);
    // Display back projection result
    cv::namedWindow("Backprojection on second image");
    cv::imshow("Backprojection on second image", result);

    // 初始化搜索的定位位置
    cv::rectangle(image, rect, cv::Scalar(0, 0, 255));
    // mean-shift algorithm
    cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                              10, // iterate max 10 times
                              1); // or until the change in centroid position is less than 1px
    std::cout << "meanshift= " << cv::meanShift(result, rect, criteria) << std::endl;

    // draw output window
	cv::rectangle(image, rect, cv::Scalar(0,255,0));
	// Display image
	cv::namedWindow("Image 2 result");
	cv::imshow("Image 2 result",image);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
