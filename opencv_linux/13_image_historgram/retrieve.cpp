/**
 * @File    : retrieve.cpp
 * @Brief   : 比较直方图搜索相似图像
 * @Author  : Wei Li
 * @Date    : 2021-07-29
*/

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "imageComparator.hpp"

int main(int argc, char **argv)
{
    // 参考图像
    cv::Mat image = cv::imread("./../images/waves.jpg");
    if (!image.data)
    {
        std::cerr << "--Error reading image file" << std::endl;
        return 1;
    }
    cv::namedWindow("Query Image");
    cv::imshow("Query Image", image);

    // 实例化对象
    ImageCompatator c;
    c.setReferenceImage(image);

    // 对图像进行检索，找到与参考图相似的图像
    cv::Mat input = cv::imread("./../images/dog.jpg");
    if (!input.data)
    {
        std::cerr << "--Error reading input image file" << std::endl;
        return -1;
    }
    std::cout << "waves vs dog: " << c.compare(input) << std::endl;

    input = cv::imread("./../images/marais.jpg");
    if (!input.data)
    {
        std::cerr << "--Error reading input image file" << std::endl;
        return -1;
    }
    std::cout << "waves vs marais: " << c.compare(input) << std::endl;

    input = cv::imread("./../images/bear.jpg");
    if (!input.data)
    {
        std::cerr << "--Error reading input image file" << std::endl;
        return -1;
    }
    std::cout << "waves vs bear: " << c.compare(input) << std::endl;

    input = cv::imread("./../images/beach.jpg");
    if (!input.data)
    {
        std::cerr << "--Error reading input image file" << std::endl;
        return -1;
    }
    std::cout << "waves vs beach: " << c.compare(input) << std::endl;

    input = cv::imread("./../images/polar.jpg");
    if (!input.data)
    {
        std::cerr << "--Error reading input image file" << std::endl;
        return -1;
    }
    std::cout << "waves vs polar: " << c.compare(input) << std::endl;

    input = cv::imread("./../images/moose.jpg");
    if (!input.data)
    {
        std::cerr << "--Error reading input image file" << std::endl;
        return -1;
    }
    std::cout << "waves vs moose: " << c.compare(input) << std::endl;

    input = cv::imread("./../images/lake.jpg");
    if (!input.data)
    {
        std::cerr << "--Error reading input image file" << std::endl;
        return -1;
    }
    std::cout << "waves vs lake: " << c.compare(input) << std::endl;

    input = cv::imread("./../images/fundy.jpg");
    if (!input.data)
    {
        std::cerr << "--Error reading input image file" << std::endl;
        return -1;
    }
    std::cout << "waves vs fundy: " << c.compare(input) << std::endl;

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
