/**
 * @File    : derivatives.cpp
 * @Brief   : 用定向滤波器检测边缘: Sobel 等边缘检测算子
 * @Author  : Wei Li
 * @Date    : 2021-07-30
*/

#include <iostream>
#include <iomanip>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "laplacianZC.hpp"

// ------------------------------
int main(int argc, char **argv)
{
    cv::Mat image = cv::imread("./../images/boldt.jpg", 0);
    if (!image.data)
    {
        std::cerr << "--Error reading bolda image file." << std::endl;
        return 1;
    }
    cv::namedWindow("Original Image");
    cv::imshow("Original Image", image);

    // 计算 Soble 算子(两个方向：水平和垂直)
    // Compute Sobel X derivative 水平方向滤波器
    cv::Mat sobelX;
    cv::Sobel(image,     // 输入
              sobelX,    // 输出
              CV_8U,     // 图像类型
              1, 0,      // 内核规格
              3,         // 正方形内核的尺寸
              0.4, 128); // 比例和偏移量
    cv::namedWindow("Sobel X Image");
    cv::imshow("Sobel X Image", sobelX);

    // Compute Sobel Y derivative 垂直方向滤波
    cv::Mat sobelY;
    cv::Sobel(image,     // 输入
              sobelY,    // 输出
              CV_8U,     // 图像类型
              0, 1,      // 内核规格
              3,         // 正方形内核的尺寸
              0.4, 128); // 比例和偏移量
    cv::namedWindow("Sobel Y Image");
    cv::imshow("Sobel Y Image", sobelY);

    // Sobel 算子的内核中既有正数又有负数，
    // Sobel 滤波器的计算结果通常是 16 位的有符号整数图像（ CV_16S）
    // 组合这两个结果（垂直和水平方向），得到 Sobel 滤波器的范数
    // Compute norm of Sobel
    cv::Sobel(image, sobelX, CV_16S, 1, 0);
    cv::Sobel(image, sobelY, CV_16S, 0, 1);
    cv::Mat sobel;
    //compute the L1 norm
    sobel = abs(sobelX) + abs(sobelY);
    double sobmin, sobmax;
    cv::minMaxLoc(sobel, &sobmin, &sobmax);
    std::cout << "sobel value range: " << sobmin << "  " << sobmax << std::endl;

    // 计算二阶导数
    // Compute Sobel X derivative (7x7)
    cv::Sobel(image, sobelX, CV_8U, 1, 0, 7, 0.001, 128);
    cv::namedWindow("Sobel X Image (7x7)");
    cv::imshow("Sobel X Image (7x7)", sobelX);

    // Print window pixel values
    for (int i = 0; i < 12; i++)
    {
        for (int j = 0; j < 12; j++)
            std::cout << std::setw(5) << static_cast<int>(sobel.at<short>(i + 79, j + 215)) << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

    /**在检测边缘时，通常只计算范数
     * 但如果需要同时计算范数和方向，可以使用 OpenCV 函数：
        cv::Sobel(image,sobelX,CV_32F,1,0);
        cv::Sobel(image,sobelY,CV_32F,0,1);
        // 计算梯度的 L2 范数和方向
        cv::Mat norm, dir;
        // 将笛卡儿坐标换算成极坐标，得到幅值和角度
        cv::cartToPolar(sobelX,sobelY,norm,dir);
        // 默认情况下，得到的方向用弧度表示。如果要使用角度，只需要增加一个参数并设为 true
     * 
     */
    // 在 convertTo 方法中使用可选的缩放参数可得到一幅图像，
    // 图像中的白色用 0 表示，更黑的灰色阴影用大于 0 的值表示
    // 这幅图像可以很方便地显示 Sobel 算子的范数
    // Conversion to 8-bit image
    // sobelImage = -alpha*sobel + 255
    cv::Mat sobelImage;
    sobel.convertTo(sobelImage, CV_8U, -255. / sobmax, 255);
    cv::namedWindow("Sobel Image");
    cv::imshow("Sobel Image", sobelImage);

    // 上图可以看出把 Sobel 算子称作边缘检测器的原因
    // 可以对这幅图像阈值化，得到图像轮廓的二值分布图
    // Apply threshold to Sobel norm (low threshold value)
    cv::Mat sobelThresholded;
    cv::threshold(sobelImage, sobelThresholded, 225, 255, cv::THRESH_BINARY);
    cv::namedWindow("Binary Sobel Image (low)");
    cv::imshow("Binary Sobel Image (low)", sobelThresholded);

    // Apply threshold to Sobel norm (high threshold value)
    cv::threshold(sobelImage, sobelThresholded, 190, 255, cv::THRESH_BINARY);
    cv::namedWindow("Binary Sobel Image (high)");
    cv::imshow("Binary Sobel Image (high)", sobelThresholded);

    // 拉普拉斯算子
    cv::Mat laplace;
    cv::Laplacian(image, laplace, CV_8U, 1, 1, 128);
    cv::namedWindow("Laplacian Image");
    cv::imshow("Laplacian Image", laplace);

    int cx(238), cy(90);
    int dx(12), dy(12);

    // Extract small window
    cv::Mat window(image, cv::Rect(cx, cy, dx, dy));
    cv::namedWindow("Image window");
    cv::imshow("Image window", window);
    cv::imwrite("window.bmp", window);

    // Compute Laplacian using LaplacianZC class
    // 实例化对象
    LaplacianZC laplacian;
    laplacian.setAperture(7);
    cv::Mat flap = laplacian.computeLaplacian(image);

    // display min max values of the lapalcian
    double lapmin, lapmax;
    cv::minMaxLoc(flap, &lapmin, &lapmax);

    // display laplacian image
    laplace = laplacian.getLaplacianImage();
    cv::namedWindow("Laplacian Image (7x7)");
    cv::imshow("Laplacian Image (7x7)", laplace);

    // Print image values
    std::cout << std::endl;
    std::cout << "Image values:\n\n";
    for (int i = 0; i < dx; i++)
    {
        for (int j = 0; j < dy; j++)
            std::cout << std::setw(5) << static_cast<int>(image.at<uchar>(i + cy, j + cx)) << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Print Laplacian values
    std::cout << "Laplacian value range=[" << lapmin << "," << lapmax << "]\n";
    std::cout << std::endl;
    for (int i = 0; i < dx; i++)
    {
        for (int j = 0; j < dy; j++)
            std::cout << std::setw(5) << static_cast<int>(flap.at<float>(i + cy, j + cx) / 100) << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;

    /** 原理解释
     * 与 Sobel 算子相比，拉普拉斯算子在计算时可以使用更大的内核，
     * 并且对图像噪声更加敏感，因此是更理想的选择（除非要重点考虑计算效率）。
     * 因为这些更大的内核是用高斯函数的二阶导数计算的，
     * 因此这个算子也常称为高斯拉普拉斯算子（ Laplacian of Gaussian， LoG）。
     * 注意，拉普拉斯算子内核的值的累加和总是 0 
     * 这保证在强度值恒定的区域中，拉普拉斯算子将变为 0
     * 因为拉普拉斯算子度量的是图像函数的曲率，所以它在平坦区域中应该等于 0
     * 
     * 拉普拉斯算子, 从内核的定义可以明显看出，任何孤立的像素值（即与周围像素差别很大的值）都会被拉普拉斯算子放大，
     * 这是因为该算子对噪声非常敏感。
     * 但更值得关注的是图像边缘附近的拉普拉斯值。
     * 图像边缘是灰度值在不同区域之间快速过渡的产物。
     * 
     * 观察图像函数在边缘上的变化（例如从暗到亮的边缘）可以发现一个规律：
     * 如果灰度级上升，那么肯定存在从正曲率（强度值开始上升）到负曲率（强度值即将到达高地）的平缓过渡。
     * 因此，如果拉普拉斯值从正数过渡到负数（反之亦然），就说明这个位置很可能是边缘，或者说边缘位于拉普拉斯函数的过零点
     * 
     * 为了说明这个观点，来看测试图像的一个小窗口中的拉普拉斯值
     * 选取城堡塔楼屋顶的底部边缘位置，具体位置见下图中的白色小框。
     * 在选中的图像窗口中可以看到这个边缘。这意味着可以检测到亚像素级精度的图像边缘，至少从理论上是成立的
     * 
     * 在拉普拉斯图像上追踪过零点曲线需要很大的耐心，
     * 但你可以用一个简化的算法来检测过零点的大致位置。
     * 这种算法首先对拉普拉斯图像阈值化（采用的阈值为 0），得到正数和负数之间的分割区域，
     * 这两个区域之间的边界就是过零点。
     * 所以，可以用形态学运算来提取这些轮廓，也就是用拉普拉斯图像减去膨胀后的图像（Beucher 梯度）
     * 生成了一个过零点的二值图像：
     */

    // Compute and display the zero-crossing points
    cv::Mat zeros;
    zeros = laplacian.getZeroCrossings(flap);
    cv::namedWindow("Zero-crossings");
    cv::imshow("Zero-crossings", 255 - zeros);

    // Print window pixel values
    std::cout << "Zero values:\n\n";
    for (int i = 0; i < dx; i++)
    {
        for (int j = 0; j < dy; j++)
            std::cout << std::setw(2) << static_cast<int>(zeros.at<uchar>(i + cy, j + cx)) / 255 << " ";
        std::cout << std::endl;
    }

    /**拉普拉斯算子是一种高通滤波器
     * 1. 用拉普拉斯算子增强图像的对比度
     * 
     * 2. 高斯差分
     *    用两个不同带宽的高斯滤波器对一幅图像做滤波，然后将这两个结果相减，就能得到由较高的频率构成的图像
     *  这些频率被一个滤波器保留，被另一个滤波器丢弃。这种运算称为高斯差分（ Difference of Gaussians， DoG）
     */
    // down-sample and up-sample the image
    cv::Mat reduced, rescaled;
    cv::pyrDown(image, reduced);
    cv::pyrUp(reduced, rescaled);

    // Display the rescaled image
    cv::namedWindow("Rescaled Image");
    cv::imshow("Rescaled Image", rescaled);

    // compute a difference of Gaussians pyramid
    cv::Mat dog;
    cv::subtract(rescaled, image, dog, cv::Mat(), CV_16S);
    cv::Mat dogImage;
    dog.convertTo(dogImage, CV_8U, 1.0, 128);

    // Display the DoG image
    cv::namedWindow("DoG Image (from pyrdown/pyrup)");
    cv::imshow("DoG Image (from pyrdown/pyrup)", dogImage);

    // Apply two Gaussian filters
    cv::Mat gauss05;
    cv::Mat gauss15;
    cv::GaussianBlur(image, gauss05, cv::Size(), 0.5);
    cv::GaussianBlur(image, gauss15, cv::Size(), 1.5);

    // compute a difference of Gaussians
    cv::subtract(gauss15, gauss05, dog, cv::Mat(), CV_16S);
    dog.convertTo(dogImage, CV_8U, 2.0, 128);

    // Display the DoG image
    cv::namedWindow("DoG Image");
    cv::imshow("DoG Image", dogImage);

    // Apply two Gaussian filters
    cv::Mat gauss20;
    cv::GaussianBlur(image, gauss20, cv::Size(), 2.0);
    cv::Mat gauss22;
    cv::GaussianBlur(image, gauss22, cv::Size(), 2.2);

    // compute a difference of Gaussians
    cv::subtract(gauss22, gauss20, dog, cv::Mat(), CV_32F);
    dog.convertTo(dogImage, CV_8U, 10.0, 128);

    // Display the DoG image
    cv::namedWindow("DoG Image (2)");
    cv::imshow("DoG Image (2)", dogImage);

    // Display the zero-crossings of DoG
    zeros = laplacian.getZeroCrossings(dog);
    cv::namedWindow("Zero-crossings of DoG");
    cv::imshow("Zero-crossings of DoG", 255 - zeros);

    // Display the image with window
    cv::rectangle(image, cv::Rect(cx, cy, dx, dy), cv::Scalar(255, 255, 255));
    cv::namedWindow("Original Image with window");
    cv::imshow("Original Image with window", image);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
