// basic function and data
#include "opencv2/core.hpp"
// digital image processsing with Class and Function
#include "opencv2/imgproc.hpp"
// image file read and write
#include "opencv2/imgcodecs.hpp"
// the UI interface for user
#include "opencv2/highgui.hpp"

#include <iostream>

/* 输出实现功能以及文件图像帮助信息
** static 静态成员函数只有一份，没有 this 指针
** argv[0] 第 0 个命令行参数默认为可执行程序的文件名 filename（全路径的）
** 默认对 lena.jpg 图像文件做离散傅里叶变换并显示其功率图谱
**
** C++ static keyword Usage
** 1、declarations of namespace members with static storage duration and internal linkage
** 2、definitions of block scope variables with static storage duration and initialized once
** 3、declarations of class members not bound to specific instances
*/
static void help(char **argv)
{
    std::cout << std::endl
        << "This program demonstrated the use of the discrete Fourier transform (DFT). " << std::endl
        << "The dft of an image is taken and it's power spectrum is displayed." << std::endl
        << std::endl
        << "Usage:" << std::endl
        << argv[0] << " [image_name -- default lena.jpg]" << std::endl
        << std::endl;
}

int main(int argc, char **argv)
{
    help(argv);

    // 使用 三目运算符 实现加载图像的默认选项和命令行参数选项
    // 命令行参数 using：./DFT.exe ./../../image/cat.jpg
    const std::string filename = argc >= 2 ? argv[1] : "./../../image/lena.jpg";

    // 读取图像灰度图数据到 mat 容器中
    cv::Mat image = imread(filename, cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        std::cout << "Error opening image" << std::endl;
        // OpenCV 定义的程序退出状态宏 define EXIT_FAILURE 1 
        return EXIT_FAILURE;
    }

    //! [expand]
    cv::Mat padded_img; //expand input image to optimal size
    int m_row = cv::getOptimalDFTSize(image.rows);
    int n_col = cv::getOptimalDFTSize(image.cols); // on the border add zero values
    cv::copyMakeBorder(image, padded_img, 0, m_row - image.rows, 0, n_col - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    //! [expand]

    //! [complex_and_real]
    cv::Mat planes[] = {cv::Mat_<float>(padded_img), cv::Mat::zeros(padded_img.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI); // Add to the expanded another plane with zeros
                                //! [complex_and_real]

    //! [dft]
    cv::dft(complexI, complexI); // this way the result may fit in the source matrix
                             //! [dft]

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    //! [magnitude]
    cv::split(complexI, planes);                    // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv::magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
    cv::Mat magI = planes[0];
    //! [magnitude]

    //! [log]
    magI += cv::Scalar::all(1); // switch to logarithmic scale
    cv::log(magI, magI);
    //! [log]

    //! [crop_rearrange]
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;

    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp; // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
    //! [crop_rearrange]

    //! [normalize]
    cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX); // Transform the matrix with float values into a
                                              // viewable image form (float between values 0 and 1).
                                              //! [normalize]

    cv::imshow("Input Image", image); // Show the result
    cv::imshow("spectrum magnitude", magI);
    cv::waitKey();

    return 0;
}
