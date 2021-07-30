/**
 * @File    : salt_pepper_noise.cpp
 * @Brief   : 访问图像的像素值，椒盐噪声
 * @Author  : Wei Li
 * @Date    : 2021-07-26
*/

#include <iostream>
#include <random>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// add salt and pepper noise to an image.
// 复制图像时仍共享了同一块图像数据, 在需要修改图像内容时，图像参数没必要采用引用传递的方式
void salt_pepper_noise(cv::Mat image, int num)
{
    // C++ 11 random
    std::default_random_engine generator;
    std::uniform_int_distribution<int> randomRow(0, image.rows - 1);
    std::uniform_int_distribution<int> randomCol(0, image.cols - 1);

    for (int k = 0; k < num; ++k)
    {
        // random image-pixel coordinate.
        int row_coordinate = randomRow(generator);
        int col_coordinate = randomCol(generator);

        // Gray-level image
        if (image.type() == CV_8UC1)
        {
            image.at<uchar>(row_coordinate, col_coordinate) = 255;
        }
        else if (image.type() == CV_8UC3)
        {
            // Color image [B-G-R]
            image.at<cv::Vec3b>(row_coordinate, col_coordinate)[0] = 255;
            image.at<cv::Vec3b>(row_coordinate, col_coordinate)[1] = 255;
            image.at<cv::Vec3b>(row_coordinate, col_coordinate)[2] = 255;
            // image.at<cv::Vec3b>(row_coordinate, col_coordinate) = cv::Vec3b(255, 255, 255);
        }
    }
}

// to illustrate the use of cv::Mat_ works only for a 1-channel image
void salt_noise(cv::Mat image, int n)
{

    // must be a gray-level image
    CV_Assert(image.type() == CV_8UC1);

    // C++ 11 random number generator
    std::default_random_engine generator;
    std::uniform_int_distribution<int> randomRow(0, image.rows - 1);
    std::uniform_int_distribution<int> randomCol(0, image.cols - 1);

    // use image with a Mat_ template
    // cv::Mat_ 模板类对 operator() 做了运算符重载， 是 cv::Mat 类的模板子类
    // 使用操作符 operator()和使用 at 方法产生的结果是完全相同的
    cv::Mat_<uchar> img(image);

    //  or with references:
    //	cv::Mat_<uchar>& im2= reinterpret_cast<cv::Mat_<uchar>&>(image);

    for (int k = 0; k < n; ++k)
    {

        // random image coordinate
        int x = randomCol(generator);
        int y = randomRow(generator);

        // add salt noise
        img(y, x) = 255;
    }
}

// ------------------------------
int main(int argc, char **argv)
{
    // 1 == cv::IMREAD_COLOR
    cv::Mat image = cv::imread("./../images/boldt.jpg", 1);

    salt_pepper_noise(image, 3000);

    cv::namedWindow("SaltImage");
    cv::imshow("SaltImage", image);
    cv::imwrite("./salted.bmp", image);
    cv::waitKey(0);
    // cv::destroyWindow("SaltImage");


    // 0 == cv::IMREAD_GRAYSACLE
    image = cv::imread("./../images/boldt.jpg", 0);

    salt_noise(image, 500);

    cv::namedWindow("SaltImage");
    cv::imshow("SaltImage", image);
    cv::waitKey();
    // void destroyWindow(const string& winname);
    // cv::destroyWindow("SaltImage");

    // 关闭所有的 HighGUI 窗口
    cv::destroyAllWindows();

    return 0;
}
