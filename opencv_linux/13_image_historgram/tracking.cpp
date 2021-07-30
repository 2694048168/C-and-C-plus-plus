/**
 * @File    : tracking.cpp
 * @Brief   : 用直方图实现视觉追踪
 * @Author  : Wei Li
 * @Date    : 2021-07-29
*/

#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "histogram.hpp"
#include "integral.hpp"

// -------------------------------
int main(int argc, char **argv)
{
    cv::Mat image = cv::imread("./../images/bike55.bmp", 0);
    if (!image.data)
    {
        std::cerr << "--Error reading image file." << std::endl;
        return 1;
    }

    // ROI
    int xo = 97, yo = 112;
    int width = 25, height = 30;
    cv::Mat roi(image, cv::Rect(xo, yo, width, height));

    // 计算感兴趣局域的直方图属性
    cv::Scalar sum = cv::sum(roi);
    std::cout << sum[0] << std::endl;

    // 计算积分图
    cv::Mat integralImage;
    cv::integral(image, integralImage, CV_32S);
    // get sum over an area using three additions/subtractions
    int sumInt = integralImage.at<int>(yo + height, xo + width) - integralImage.at<int>(yo + height, xo) - integralImage.at<int>(yo, xo + width) + integralImage.at<int>(yo, xo);
    std::cout << sumInt << std::endl;

    // histogram of 16 bins
    Histogram1D h;
    h.setNumBins(16);
    // compute histogram over image roi
    cv::Mat refHistogram = h.getHistogram(roi);

    cv::namedWindow("Reference Histogram");
    cv::imshow("Reference Histogram", h.getHistogramImage(roi, 16));
    std::cout << refHistogram << std::endl;

    // first create 16-plane binary image
    cv::Mat planes;
    convertToBinaryPlanes(image, planes, 16);
    // then compute integral image
    IntegralImage<float, 16> intHisto(planes);

    // for testing compute a histogram of 16 bins with integral image
    cv::Vec<float, 16> histogram = intHisto(xo, yo, width, height);
    std::cout << histogram << std::endl;

    cv::namedWindow("Reference Histogram (2)");
    cv::Mat im = h.getImageOfHistogram(cv::Mat(histogram), 16);
    cv::imshow("Reference Histogram (2)", im);

    // 已经计算好感兴趣区域的直方图属性(小女孩骑单车)，在其他图像中进行搜索
    cv::Mat secondImage = cv::imread("./../images/bike65.bmp", 0);
    if (!secondImage.data)
    {
        std::cerr << "--Error reading image file." << std::endl;
        return 1;
    }

    // first create 16-plane binary image
    convertToBinaryPlanes(secondImage, planes, 16);
    // then compute integral image
    IntegralImage<float, 16> intHistogram(planes);

    // compute histogram of 16 bins with integral image (testing)
    histogram = intHistogram(135, 114, width, height);
    std::cout << histogram << std::endl;

    cv::namedWindow("Current Histogram");
    cv::Mat im2 = h.getImageOfHistogram(cv::Mat(histogram), 16);
    cv::imshow("Current Histogram", im2);
    std::cout << "Distance= " << cv::compareHist(refHistogram, histogram, cv::HISTCMP_INTERSECT) << std::endl;

    // 执行搜索时，循环遍历可能出现目标的位置，
    // 并将它的直方图与基准直方图做比较，目的是找到与直方图最相似的位置
    double maxSimilarity = 0.0;
    int xbest, ybest;
    // 遍历原始图像中女孩位置周围的水平长条
    for (int y = 110; y < 120; y++)
    {
        for (int x = 0; x < secondImage.cols - width; x++)
        {
            // 用积分图像计算 16 个箱子的直方图
            histogram = intHistogram(x, y, width, height);
            // 计算与基准直方图的差距
            double distance = cv::compareHist(refHistogram,
                                              histogram,
                                              cv::HISTCMP_INTERSECT);
            // 找到最相似直方图的位置
            if (distance > maxSimilarity)
            {
                xbest = x;
                ybest = y;
                maxSimilarity = distance;
            }

            std::cout << "Distance(" << x << "," << y << ")=" << distance << std::endl;
        }
    }
    std::cout << "Best solution= (" << xbest << "," << ybest << ")=" << maxSimilarity << std::endl;

    // 在最准确的位置画矩形
    // draw a rectangle around target object
    cv::rectangle(image, cv::Rect(xo, yo, width, height), 0);
    cv::namedWindow("Initial Image");
	cv::imshow("Initial Image",image);

    cv::namedWindow("New Image");
	cv::imshow("New Image",secondImage);
    // draw rectangle at best location
	cv::rectangle(secondImage,cv::Rect(xbest,ybest,width,height),0);
    // draw rectangle around search area
	cv::rectangle(secondImage,cv::Rect(0,110,secondImage.cols,height+10),255);
	cv::namedWindow("Object location");
	cv::imshow("Object location",secondImage);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
