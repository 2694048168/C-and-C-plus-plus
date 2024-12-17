/**
 * @file 15_multiThreadImgProcessing.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <opencv2/opencv.hpp>

#include <iostream>
#include <thread>
#include <vector>

/**
  * @brief 一个多线程的实际例子是图像处理应用程序
  * 在这个应用程序中,单个图像被分割成较小的部分,每个部分由单独的线程处理;
  * 这种方法可以通过利用多个处理器核心显著提高应用程序的性能;
  * 
  */
void process_image(cv::Mat &image, cv::Rect roi)
{
    // 图像处理
    cv::Mat roi_image = image(roi);
    cv::cvtColor(roi_image, roi_image, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(roi_image, roi_image, cv::Size(3, 3), 0);
    cv::Canny(roi_image, roi_image, 100, 200);
}

// ------------------------------------------
int main(int /* argc */, char ** /* argv */)
{
    cv::Mat image = cv::imread("./images/ConcurrencyParallelism.png");

    // 将图像分割成感兴趣的区域 (ROIs)
    unsigned int num_threads = 4;
    unsigned int roi_width   = image.cols / num_threads;

    std::vector<cv::Rect> imgROIVec;
    for (size_t idx{0}; idx < num_threads; ++idx)
    {
        int x      = idx * roi_width;
        int y      = 0;
        int width  = roi_width;
        int height = image.rows;
        if (idx == num_threads - 1)
        {
            // 最后一个 ROI 占据剩余宽度
            width = image.cols - x;
        }
        imgROIVec.emplace_back(cv::Rect(x, y, width, height));
    }

    // 在单独的线程中处理每个 ROI
    std::vector<std::thread> threadTaskVec;
    for (size_t idx{0}; idx < num_threads; ++idx)
    {
        threadTaskVec.emplace_back(std::thread(process_image, std::ref(image), imgROIVec[idx]));
    }

    // 等待所有线程完成
    for (auto &task : threadTaskVec)
    {
        task.join();
    }

    // 显示处理后的图像
    cv::imshow("Processed Image", image);
    cv::waitKey(0);

    return 0;
}
