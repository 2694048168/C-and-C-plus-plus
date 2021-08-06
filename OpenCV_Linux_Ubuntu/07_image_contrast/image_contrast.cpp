/**
 * @File    : image_contrast.cpp
 * @Brief   : 扫描图像并访问相邻像素，图像锐化处理(拉普拉斯算子)
 * @Author  : Wei Li
 * @Date    : 2021-07-27
*/

#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// 可以用以下方法计算锐化的数值：
// sharpened_pixel = 5 * current - left - right - up - down;
void sharpen(const cv::Mat &image, cv::Mat &result)
{
    result.create(image.size(), image.type());
    int num_channels = image.channels();
    int num_rows = image.rows;
    int num_cols = image.cols;
    // 最后一列以及适配灰度图和彩色图
    int num_pixels_row = (num_cols - 1) * num_channels;

    // 处理所有行(除了第一行和最后一行)
    for (int row_index = 1; row_index < num_rows - 1; ++row_index)
    {
        // 可以使用 迭代器 的方式代替原始指针方式
        const uchar *ptr_previous = image.ptr<const uchar>(row_index - 1);
        const uchar *ptr_current = image.ptr<const uchar>(row_index);
        const uchar *ptr_next = image.ptr<const uchar>(row_index + 1);

        uchar *ptr_output = result.ptr<uchar>(row_index);

        // 处理每一行的像素(除了第一列和最后一列)
        // 巧妙处理灰度图和彩色图的适配 col_index=num_channels
        for (int col_index = num_channels; col_index < num_pixels_row; ++col_index)
        {
            // 针对每一个像素点进行锐化算子处理
            // Saturation Arithmetics ： I(x,y) = min(max(round(r),0),255)
            // cv::saturate_cast<uchar> 进行饱和算法处理计算结果的溢出情况
            *ptr_output++ = cv::saturate_cast<uchar>(
                5 * ptr_current[col_index] 
                - ptr_previous[col_index] 
                - ptr_next[col_index] 
                - ptr_current[col_index - num_channels] 
                - ptr_current[col_index + num_channels]);
        }
    }

    // 未进行处理的像素设置为 0(最外围的一圈) zero-padding
    if (num_channels == 1)
    {
        result.row(0).setTo(cv::Scalar(0));
        result.row(num_rows - 1).setTo(cv::Scalar(0));
        result.col(0).setTo(cv::Scalar(0));
        result.col(num_cols - 1).setTo(cv::Scalar(0));
    }
    else
    {
        result.row(0).setTo(cv::Scalar(0, 0, 0));
        result.row(num_rows - 1).setTo(cv::Scalar(0, 0, 0));
        result.col(0).setTo(cv::Scalar(0, 0, 0));
        result.col(num_cols - 1).setTo(cv::Scalar(0, 0, 0));
    }
}


// 使用类型信号处理里面的卷积操作，对图像进行卷积核滤波
// 针对灰度图进行锐化滤波
void sharpen2D(const cv::Mat &image, cv::Mat &result)
{
    // 初始化卷积核
    cv::Mat conv_kernel(3, 3, CV_32F, cv::Scalar(0));
    // 锐化滤波器
    conv_kernel.at<float>(1,1) = 5.0;
    conv_kernel.at<float>(0,1) = -1.0;
    conv_kernel.at<float>(2,1) = -1.0;
    conv_kernel.at<float>(1,0) = -1.0;
    conv_kernel.at<float>(1,2) = -1.0;

    cv::filter2D(image, result, image.depth(), conv_kernel);
}

// -----------------------------
int main(int argc, char **argv)
{
    auto gray_image = cv::imread("./../images/boldt.jpg", 0);
    auto color_image = cv::imread("./../images/boldt.jpg", 1);
    // if (gray_image.empty() && color_image.empty())
    if (!gray_image.data && !color_image.data)
    {
        std::cerr << "Error reading image file.";
        return 1;
    }

    cv::Mat result_img;

    // test Filter2D with gray-level image.
    double time_stamp_sec = static_cast<double>(cv::getTickCount());
    sharpen2D(gray_image, result_img);
    time_stamp_sec = (static_cast<double>(cv::getTickCount()) - time_stamp_sec) / cv::getTickFrequency();
    std::cout << "Filter Time = " << time_stamp_sec << std::endl;

    const cv::String win_name_filter = "SharpenImageFilter";
    cv::namedWindow(win_name_filter);
    cv::imshow(win_name_filter, result_img);

    // test sharpen operator with color image.
    time_stamp_sec = static_cast<double>(cv::getTickCount());
    sharpen(color_image, result_img);
    time_stamp_sec = (static_cast<double>(cv::getTickCount()) - time_stamp_sec) / cv::getTickFrequency();
    std::cout << "Sharpen Operator Time = " << time_stamp_sec << std::endl;

    const cv::String win_name = "SharpenImageOperator";
    cv::namedWindow(win_name);
    cv::imshow(win_name, result_img);
    
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
