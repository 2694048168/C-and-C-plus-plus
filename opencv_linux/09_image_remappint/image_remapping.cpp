/**
 * @File    : image_remapping.cpp
 * @Brief   : 重映射是通过修改像素的位置，生成一个新版本的图像。
 * 为了构建新图像，需要知道目标图像中每个像素的原始位置。
 * 因此需要的映射函数应该能根据像素的新位置得到像素的原始位置。
 * 这个转换过程描述了如何把新图像的像素映射回原始图像，称为反向映射.
 * 
 * 这个过程不会修改像素值，而是把每个像素的位置重新映射到新的位置。
 * 这可用来创建图像特效，或者修正因镜片等原因导致的图像扭曲.
 * @Author  : Wei Li
 * @Date    : 2021-07-27
*/

#include <iostream>
#include <cmath>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


// 创建波浪效果
void wave(const cv::Mat &image, cv::Mat &result)
{
    // 映射参数
    cv::Mat srcX(image.rows, image.cols, CV_32F);
    cv::Mat srcY(image.rows, image.cols, CV_32F);

    // 创建映射参数
    for (int row_index = 0; row_index < image.rows; ++row_index)
    {
        for (int col_index = 0; col_index < image.cols; ++col_index)
        {
            // 像素值保持在同一列
            srcX.at<float>(row_index, col_index) = col_index;
            // 原始的 row_index 行像素，根据正弦曲线进行波动，导致必须要进行插值处理
            srcY.at<float>(row_index, col_index) = row_index + 3 * std::sin(col_index / 6.0);

            // 水平翻转效果
			// srcX.at<float>(i,j)= image.cols-j-1;
			// srcY.at<float>(i,j)= i;
        }
    }

    // 应用映射参数
    // 原始图像， 映射后图像， X方向映射参数， Y方向映射参数， 插值填补方法
    // sin 函数进行转换，但这也导致必须在真实像素之间插入虚拟像素的值
    cv::remap(image, result, srcX, srcY, cv::INTER_LINEAR);
}

// ------------------------------
int main(int argc, char** argv)
{
    const cv::String filename = "./../images/boldt.jpg";
    cv::Mat image = cv::imread(filename, 0);
    if (image.empty())
    {
        std::cerr << "Error reading image file." << std::endl;
        return 1;
    }

    const cv::String win_name1 = "OriginalImage";
    cv::namedWindow(win_name1);
    cv::imshow(win_name1, image);

    cv::Mat result;
    wave(image, result);

    const cv::String win_name2 = "RemappingImage";
    cv::namedWindow(win_name2);
    cv::imshow(win_name2, result);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
