/**
 * @File    : image_filter.cpp
 * @Brief   : 图像滤波操作：低通滤波器；利用滤波进行下采样和上采样； 中值滤波器
 * @Author  : Wei Li
 * @Date    : 2021-07-30
*/

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// ----------------------------
int main(int argc, char **argv)
{
    cv::Mat image = cv::imread("./../images/boldt.jpg", 0);
    if (!image.data)
    {
        std::cerr << "--Error reading boldt image file." << std::endl;
        return 1;
    }
    cv::namedWindow("Original Image");
    cv::imshow("Original Image", image);

    // Blur the image with a mean filter 均值滤波器对图像进行模糊
    cv::Mat result;
    cv::blur(image, result, cv::Size(5, 5));
    cv::namedWindow("Mean filtered Image");
    cv::imshow("Mean filtered Image", result);
    // 滤波核的大小不同
    cv::blur(image, result, cv::Size(9, 9));
    cv::namedWindow("Mean filtered Image (9x9)");
    cv::imshow("Mean filtered Image (9x9)", result);
    cv::blur(image, result, cv::Size(3, 3));
    cv::namedWindow("Mean filtered Image (3x3)");
    cv::imshow("Mean filtered Image (3x3)", result);
    cv::blur(image, result, cv::Size(7, 7));
    cv::namedWindow("Mean filtered Image (7x7)");
    cv::imshow("Mean filtered Image (7x7)", result);

    // 利用高斯分布初始化滤波核的数值从而进行图像滤波
    // Gaussian Blur the image
    cv::GaussianBlur(image, result,
                     cv::Size(5, 5), // size of the filter
                     1.5);           // parameter controlling the shape of the Gaussian 方差(标准差)
    cv::namedWindow("Gaussian filtered Image");
    cv::imshow("Gaussian filtered Image", result);
    // 控制滤波器的大小和高斯形状
    cv::GaussianBlur(image, result, cv::Size(9, 9), 1.7);
    cv::namedWindow("Gaussian filtered Image (9x9)");
    cv::imshow("Gaussian filtered Image (9x9)", result);

    // OpenCV 函数可以获取一维高斯滤波器
    // 高斯函数是可分离的，二维高斯滤波器可以分离为两个一维高斯进行计算，效率更高
    // Get the gaussian kernel (1.5)
    cv::Mat gauss = cv::getGaussianKernel(9, 1.5, CV_32F);
    // Display kernel values
    cv::Mat_<float>::const_iterator it = gauss.begin<float>();
    cv::Mat_<float>::const_iterator itend = gauss.end<float>();
    std::cout << "1.5 = [";
    for (; it != itend; ++it)
    {
        std::cout << *it << " ";
    }
    std::cout << "]" << std::endl;

    // Get the gaussian kernel (0.5)
    gauss = cv::getGaussianKernel(9, 0.5, CV_32F);
    it = gauss.begin<float>();
    itend = gauss.end<float>();
    std::cout << "0.5 = [";
    for (; it != itend; ++it)
    {
        std::cout << *it << " ";
    }
    std::cout << "]" << std::endl;

    // Get the gaussian kernel (2.5)
    gauss = cv::getGaussianKernel(9, 2.5, CV_32F);
    it = gauss.begin<float>();
    itend = gauss.end<float>();
    std::cout << "2.5 = [";
    for (; it != itend; ++it)
    {
        std::cout << *it << " ";
    }
    std::cout << "]" << std::endl;

    // Get the gaussian kernel (9 elements)
    gauss = cv::getGaussianKernel(9, -1, CV_32F);
    it = gauss.begin<float>();
    itend = gauss.end<float>();
    std::cout << "9 = [";
    for (; it != itend; ++it)
    {
        std::cout << *it << " ";
    }
    std::cout << "]" << std::endl;

    // OpenCV Sobel 滤波器,内核系数
    cv::Mat kx, ky;
    cv::getDerivKernels(kx, ky, 2, 2, 7, true);
    cv::Mat_<float>::const_iterator kit = kx.begin<float>();
    cv::Mat_<float>::const_iterator kitend = kx.end<float>();
    std::cout << "[";
    for (; kit != kitend; ++kit)
    {
        std::cout << *kit << " ";
    }
    std::cout << "]" << std::endl;

    // 利用 均值滤波器， 中值滤波器进行椒盐噪声去除
    image = cv::imread("./../images/salted.bmp", 0);
    if (!image.data)
    {
        std::cerr << "--Error reading salted image file." << std::endl;
        return 1;
    }
    cv::namedWindow("S&P Image");
    cv::imshow("S&P Image", image);

    // Blur the image with a mean filter
    cv::blur(image, result, cv::Size(5, 5));
    cv::namedWindow("Mean filtered S&P Image");
    cv::imshow("Mean filtered S&P Image", result);

    // Applying a median filter
    cv::medianBlur(image, result, 5);
    cv::namedWindow("Median filtered Image");
    cv::imshow("Median filtered Image", result);

    // 图像进行下采样操作
    // Reduce by 4 the size of the image (the wrong way)
    image = cv::imread("./../images/boldt.jpg", 0);
    cv::Mat reduced(image.rows / 4, image.cols / 4, CV_8U);

    for (int i = 0; i < reduced.rows; ++i)
        for (int j = 0; j < reduced.cols; ++j)
            reduced.at<uchar>(i, j) = image.at<uchar>(i * 4, j * 4);

    // Display the reduced image
    cv::namedWindow("Badly reduced Image");
    cv::imshow("Badly reduced Image", reduced);
    bool check_write = cv::imwrite("./reduce_image.jpg", reduced);
    /** 这幅图像的质量明显降低了，例如原始图像中城堡顶部倾斜的边缘在缩小后的图像中看起来像是楼梯。
     * 图像的纹理部分也能看到锯齿状的变形（如砖墙）。
     * 这些令人讨厌的伪影是一种叫作空间假频的现象造成的。
     * 当你试图在图像中包含高频成分，但由于图像太小而无法包含时，就会出现这种现象。
     * 实际上，在小图像（即像素较少的图像）中展现精致纹理和尖锐边缘的效果很差
     * 不如在较高分辨率的图像中展现它们的效果好（想想高清电视机和普通电视机的差别）。
     * 图像中精致的细节对应着高频，因此需要在缩小图像之前去除它的高频成分
     */
    if (!check_write)
    {
        std::cerr << "--Error writing image to disk." << std::endl;
        return -1;
    }
    cv::resize(reduced, reduced, cv::Size(), 4, 4, cv::INTER_NEAREST);
    cv::namedWindow("Badly reduced");
    cv::imshow("Badly reduced", reduced);
    cv::imwrite("./badlyreducedimage.bmp", reduced);

    // 可以用低通滤波器实现。因此在删除部分列和行之前，
    // 必须先在原始图像上应用低通滤波器，这样才能使图像在缩小到四分之一后不出现伪影
    // first remove high frequency component
    cv::GaussianBlur(image, image, cv::Size(11, 11), 1.75);
    // keep only 1 of every 4 pixels
    cv::Mat reduced2(image.rows / 4, image.cols / 4, CV_8U);
    for (int i = 0; i < reduced2.rows; i++)
        for (int j = 0; j < reduced2.cols; j++)
            reduced2.at<uchar>(i, j) = image.at<uchar>(i * 4, j * 4);

    // Display the reduced image
    cv::namedWindow("Reduced Image, original size");
    cv::imshow("Reduced Image, original size", reduced2);
    cv::imwrite("reducedimage.bmp", reduced2);

    // 利用插值方法进行下/上采样
    // 双线性插值方法；双三次插值方法；最邻近插值方法;利用卷积操作进行插值(上采样模块)
    // resizing with NN
    cv::Mat newImage;
    cv::resize(reduced2, newImage, cv::Size(), 4, 4, cv::INTER_NEAREST);
    cv::namedWindow("Reduced Image");
    cv::imshow("Reduced Image", newImage);

    // resizing with bilinear
    cv::resize(reduced2, newImage, cv::Size(), 4, 4, cv::INTER_LINEAR);
    cv::namedWindow("Bilinear resizing");
    cv::imshow("Bilinear resizing", newImage);

    // resizing with bilinear
    cv::resize(reduced2, newImage, cv::Size(), 4, 4, cv::INTER_CUBIC);
    cv::namedWindow("Bicubic resizing");
    cv::imshow("Bicubic resizing", newImage);

    /**图像金字塔操作
     * 为避免混叠现象的发生，在缩减图像之前必须进行低通滤波。
     * 低通滤波的作用是消除在缩减后的图像中无法表示的高频部分,
     * 这一现象称为 Nyquist-Shannon 定理，它表明如果把图像缩小一半，那么其可见的频率带宽也将减少一半。
     * 
     * OpenCV 中有一个专用函数，利用这个原理实现了图像缩减，即 cv::pyrDown & cv::pyrUp
     * 在这种提升像素采样的过程中，先在每两行和每两列之间分别插入值为 0 的像素，
     * 然后对扩展后的图像应用同样的 5×5 高斯滤波器（但系数要扩大 4 倍）。
     * 先缩小一幅图像再把它放大，显然不能完全让它恢复到原始状态，因为缩小过程中丢失的信息是无法恢复的。
     * 
     * 这两个函数可用来创建图像金字塔。
     * 它是一个数据结构，由一幅图像不同尺寸的版本堆叠起来，用于高效的图像分析
     * 每层图像的尺寸是后一层的 2 倍，但是这个比例还可以更小，也不一定是整数（可以是 1.2）
     * 如果要在图像中快速检测一个物体，可以先在金字塔顶部的小图像上检测。
     * 当定位到感兴趣的物体时，在金字塔的更低层次进行更精细的搜索，更低层次的图像分辨率更高.
     */
    cv::Mat pyramid_image(image.rows, image.cols + image.cols / 2 + image.cols / 4 + image.cols / 8, CV_8U, cv::Scalar(255));
    image.copyTo(pyramid_image(cv::Rect(0, 0, image.cols, image.rows)));

    cv::pyrDown(image, reduced); // reduce image size by half
    reduced.copyTo(pyramid_image(cv::Rect(image.cols, image.rows / 2, image.cols / 2, image.rows / 2)));
    cv::pyrDown(reduced, reduced2); // reduce image size by another half
    reduced2.copyTo(pyramid_image(cv::Rect(image.cols + image.cols / 2, image.rows - image.rows / 4, image.cols / 4, image.rows / 4)));
    cv::pyrDown(reduced2, reduced); // reduce image size by another half
    reduced.copyTo(pyramid_image(cv::Rect(image.cols + image.cols / 2 + image.cols / 4, image.rows - image.rows / 8, image.cols / 8, image.rows / 8)));

    // Display the pyramid
    cv::namedWindow("Pyramid of images");
    cv::imshow("Pyramid of images", pyramid_image);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
