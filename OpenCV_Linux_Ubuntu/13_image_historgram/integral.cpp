/**
 * @File    : integral.cpp
 * @Brief   : 积分图的应用——自适应的阈值化
 * @Author  : Wei Li
 * @Date    : 2021-07-29
*/

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "integral.hpp"

// ------------------------------
int main(int argc, char **argv)
{
    cv::Mat image = cv::imread("./../images/book.jpg", 0);
    if (!image.data)
    {
        std::cerr << "--Error reading image file." << std::endl;
        return 1;
    }

    // rotate the image for easier display
    cv::transpose(image, image);
    cv::flip(image, image, 0);
    // display original image
    cv::namedWindow("Original Image");
    cv::imshow("Original Image", image);

    // 对比固定的阈值和自适应的阈值
    // using a fixed threshold
    cv::Mat binaryFixed;
    cv::Mat binaryAdaptive;
    cv::threshold(image, binaryFixed, 70, 255, cv::THRESH_BINARY);
    // using as adaptive threshold
    int blockSize = 21; // size of the neighborhood 邻域的尺寸
    int threshold = 10; // pixel will be compared to (mean-threshold) 像素将与(mean-threshold)进行比较

    /** 实际上，不管选用什么阈值，图像都会丢失一部分文本，还有部分文本会消失在阴影下。
     * 要解决这个问题，有一个办法就是采用局部阈值，即根据每个像素的邻域计算阈值。
     * 这种策略称为自适应阈值化，将每个像素的值与邻域的平均值进行比较。
     * 如果某像素的值与它的局部平均值差别很大，就会被当作异常值在阈值化过程中剔除。
     * 因此自适应阈值化需要计算每个像素周围的局部平均值。这需要多次计算图像窗口的累计值，可以通过积分图像提高计算效率
     */
    // 讨论积分图像的效率
    int64 time;
    time = cv::getTickCount();
    /** OpenCV 函数进行自适应阈值处理
     * 除了在阈值化中使用局部平均值 ADAPTIVE_THRESH_MEAN_C
     * 可以使用高斯（ Gaussian）加权累计值（该方法的标志为 ADAPTIVE_THRESH_GAUSSIAN_C）
     */
    cv::adaptiveThreshold(image,                      // 输入图像
                          binaryAdaptive,             // 输出二值图像
                          255,                        // 输出的最大值
                          cv::ADAPTIVE_THRESH_MEAN_C, // 方法
                          cv::THRESH_BINARY,          // 阈值类型
                          blockSize,                  // 块的大小
                          threshold);                 // 使用的阈值
    time = cv::getTickCount() - time;
    std::cout << "Time (adaptiveThreshold) = " << time << std::endl;

    // 计算积分图
    IntegralImage<int, 1> integral(image);
    // 积分图结果 两者结果完全一致，效率问题
    std::cout << "sum = " << integral(18, 45, 30, 50) << std::endl;
    cv::Mat test(image, cv::Rect(18, 45, 30, 50));
    cv::Scalar t = cv::sum(test);
    std::cout << "sum test = " << t[0] << std::endl;

    cv::namedWindow("Fixed Threshold");
    cv::imshow("Fixed Threshold", binaryFixed);
    cv::namedWindow("Adaptive Threshold");
    cv::imshow("Adaptive Threshold", binaryAdaptive);

    // 手动进行积分图计算和自适应阈值处理
    cv::Mat binary = image.clone();

    time = cv::getTickCount();
    int num_lines = binary.rows; // number of lines
    int num_cols = binary.cols;  // total number of elements per line

    // 计算积分图
    cv::Mat integral_image;
    cv::integral(image, integral_image, CV_32S);

    // 逐行进行处理
    int halfSize = blockSize / 2;
    for (int j = halfSize; j < num_lines - halfSize - 1; ++j)
    {
        // 获取 j 行地址
        uchar *data = binary.ptr<uchar>(j);
        int *integral_data1 = integral_image.ptr<int>(j - halfSize);
        int *integral_data2 = integral_image.ptr<int>(j + halfSize + 1);

        // 处理 j 行里面的像素
        for (int i = halfSize; i < num_cols - halfSize - 1; ++i)
        {
            // 求和 (直方图统计概念)
            int sum = (integral_data2[i + halfSize + 1] - integral_data2[i - halfSize] 
            - integral_data1[i + halfSize + 1] + integral_data1[i - halfSize]) / (blockSize * blockSize);

            // 自适应阈值
            if (data[i] < (sum - threshold))
            {
                data[i] = 0;
            }
            else
            {
                data[i] = 255;
            }
        }
    }
    // 对未处理的边界像素，直接全部赋值为白色 255
    for (int j = 0; j < halfSize; j++)
    {
        uchar *data = binary.ptr<uchar>(j);

        for (int i = 0; i < binary.cols; i++)
        {
            data[i] = 255;
        }
    }
    for (int j = binary.rows - halfSize - 1; j < binary.rows; j++)
    {
        uchar *data = binary.ptr<uchar>(j);

        for (int i = 0; i < binary.cols; i++)
        {
            data[i] = 255;
        }
    }
    for (int j = halfSize; j < num_lines - halfSize - 1; j++)
    {
        uchar *data = binary.ptr<uchar>(j);

        for (int i = 0; i < halfSize; i++)
        {
            data[i] = 255;
        }
        for (int i = binary.cols - halfSize - 1; i < binary.cols; i++)
        {
            data[i] = 255;
        }
    }

    time= cv::getTickCount()-time;
	std::cout << "time integral= " << time << std::endl; 
	cv::namedWindow("Adaptive Threshold (integral)");
	cv::imshow("Adaptive Threshold (integral)",binary);

    // 用 OpenCV 的 "图像运算符" 来编写自适应阈值化过程
    time = cv::getTickCount();
    cv::Mat filtered;
    cv::Mat binaryFiltered;
    // boxFilter 计算矩形区域内像素的平均值
    cv::boxFilter(image, filtered, CV_8U, cv::Size(blockSize, blockSize));
    // 检查像素是否大于(mean + threshold)
    binaryFiltered = image >= (filtered - threshold);
    time = cv::getTickCount() - time;
    std::cout << "Time filtered = " << time << std::endl;
    cv::namedWindow("Adaptive Threshold (filtered)");
	cv::imshow("Adaptive Threshold (filtered)",binaryFiltered);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
