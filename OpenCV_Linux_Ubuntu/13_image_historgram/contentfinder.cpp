/**
 * @File    : contentFinder.cpp
 * @Brief   : 根据图像直方图属性查找图像特定内容； 反向投影直方图检测特定图像内容
 * @Author  : Wei Li
 * @Date    : 2021-07-28
*/

#include <iostream>

#include <opencv2/highgui.hpp>

#include "histogram.hpp"
#include "contentFinder.hpp"
#include "colorhistogram.hpp"


// -----------------------------
int main(int argc, char **argv)
{
    cv::Mat image = cv::imread("./../images/waves.jpg", 0);
    if (!image.data)
    {
        std::cerr << "Error reading image file." << std::endl;
        return 1;
    }

    // 定义特定内容的区域(ROI) —— 云彩
    cv::Mat imageROI;
    imageROI = image(cv::Rect(216, 33, 24, 30));
    // Display reference patch
    cv::namedWindow("Reference");
    cv::imshow("Reference", imageROI);
    // Find histogram of reference
    Histogram1D h;
    cv::Mat hist = h.getHistogram(imageROI);
    cv::namedWindow("Reference Hist");
    cv::imshow("Reference Hist", h.getHistogramImage(imageROI));

    // 实例化对象
    ContentFinder finder;
    // 直方图反向投影
    finder.setHistogram(hist);
    finder.setThreshold(-1.0f);
    cv::Mat result1;
    result1 = finder.find(image);

    // 为提高可读性，对得到的二值化图像做了反色处理(negative image)，属于该区域的概率从亮（低概率）到暗（高概率）
    cv::Mat negative_img;
    result1.convertTo(negative_img, CV_8U, -1.0, 255.0);
    cv::namedWindow("Backprojection result");
    cv::imshow("Backprojection result", negative_img);

    // Get binary back-projection
    finder.setThreshold(0.12f);
    result1 = finder.find(image);

    // Draw a rectangle around the reference area
    cv::rectangle(image, cv::Rect(216, 33, 24, 30), cv::Scalar(0, 0, 0));

    // Display image
    cv::namedWindow("Image");
    cv::imshow("Image", image);
    // Display result
    cv::namedWindow("Detection Result");
    cv::imshow("Detection Result", result1);

    // 对彩色图像进行处理
	cv::Mat color = cv::imread("./../images/waves.jpg");
    if (!color.data)
    {
        std::cerr << "Error reading image file." << std::endl;
        return 1;
    }

	ColorHistogram hc;

    // extract region of interest
    imageROI = color(cv::Rect(0, 0, 100, 45)); // blue sky area

    // Get 3D colour histogram (8 bins per channel)
    hc.setSize(8); // 8x8x8
    cv::Mat shist = hc.getHistogram(imageROI);

    // set histogram to be back-projected
    finder.setHistogram(shist);
    finder.setThreshold(0.05f);

    // Get back-projection of color histogram
    result1 = finder.find(color);

    cv::namedWindow("Color Detection Result");
    cv::imshow("Color Detection Result", result1);

    // Second color image
    cv::Mat color2 = cv::imread("./../images/dog.jpg");
    if (!color2.data)
    {
        std::cerr << "Error reading image file." << std::endl;
        return 1;
    }

    cv::namedWindow("Second Image");
    cv::imshow("Second Image", color2);

    // Get back-projection of color histogram
    cv::Mat result2 = finder.find(color2);

    cv::namedWindow("Result color (2)");
    cv::imshow("Result color (2)", result2);

    // Get ab color histogram
    hc.setSize(256); // 256x256
    cv::Mat colorhist = hc.getabHistogram(imageROI);

    // display 2D histogram
    colorhist.convertTo(negative_img, CV_8U, -1.0, 255.0);
    cv::namedWindow("ab histogram");
    cv::imshow("ab histogram", negative_img);

    // set histogram to be back-projected
    finder.setHistogram(colorhist);
    finder.setThreshold(0.05f);

    // 不同色彩空间的处理
    // Convert to Lab space
    cv::Mat lab;
    cv::cvtColor(color, lab, cv::COLOR_BGR2Lab);

    // Get back-projection of ab histogram
    int ch[2] = {1, 2};
    result1 = finder.find(lab, 0, 256.0f, ch);

    cv::namedWindow("Result ab (1)");
    cv::imshow("Result ab (1)", result1);

    // Second colour image
    cv::cvtColor(color2, lab, cv::COLOR_BGR2Lab);

    // Get back-projection of ab histogram
    result2 = finder.find(lab, 0, 256.0, ch);

    cv::namedWindow("Result ab (2)");
    cv::imshow("Result ab (2)", result2);

    // Draw a rectangle around the reference sky area
    cv::rectangle(color, cv::Rect(0, 0, 100, 45), cv::Scalar(0, 0, 0));
    cv::namedWindow("Color Image");
    cv::imshow("Color Image", color);

    // Get Hue colour histogram
    hc.setSize(180); // 180 bins
    colorhist = hc.getHueHistogram(imageROI);

    // set histogram to be back-projected
    finder.setHistogram(colorhist);

    // Convert to HSV space
    cv::Mat hsv;
    cv::cvtColor(color, hsv, cv::COLOR_BGR2HSV);

    // Get back-projection of hue histogram
    ch[0] = 0;
    result1 = finder.find(hsv, 0.0f, 180.0f, ch);

    cv::namedWindow("Result Hue (1)");
    cv::imshow("Result Hue (1)", result1);

    // Second colour image
    color2 = cv::imread("./../images/dog.jpg");
    if (color2.empty())
    {
        std::cerr << "Error reading image file." << std::endl;
        return 1;
    }

    // Convert to HSV space
    cv::cvtColor(color2, hsv, cv::COLOR_BGR2HSV);

    // Get back-projection of hue histogram
    result2 = finder.find(hsv, 0.0f, 180.0f, ch);

    cv::namedWindow("Result Hue (2)");
    cv::imshow("Result Hue (2)", result2);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
