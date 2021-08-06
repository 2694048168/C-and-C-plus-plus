/**
 * @File    : color_detector.hpp
 * @Brief   : 采用 策略设计模式 来实现，一种面向对象的设计模式，用很巧妙的方法将算法封装进类。
 * 采用这种模式后，可以很轻松地替换算法，或者组合多个算法以实现更复杂的功能。
 * 而且这种模式能够尽可能地将算法的复杂性隐藏在一个直观的编程接口后面，更有利于算法的部署.
 * 
 * 一旦用策略设计模式把算法封装进类，就可以通过创建类的实例来部署算法，实例通常是在程序初始化的时候创建的。
 * 在运行构造函数时，类的实例会用默认值初始化算法的各种参数，使其立即进入可用状态。
 * 还可以用适当的方法来读写算法的参数值。在 GUI 应用程序中，可以用多种部件（文本框、滑动条等）显示和修改参数，用户操作起来很容易。
 * 
 * @Author  : Wei Li
 * @Date    : 2021-07-27
*/

#ifndef COLOR_DETECTOR
#define COLOR_DETECTOR

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// 策略设计模式
class ColorDetector
{
public:
    // 构造函数(对算法的参数进行初始化)
    // ColorDetector() = default;  // 显示的默认构造函数(编译器生成) 调用的时候出现二义性？？？
    ColorDetector() : maxDist(100), target(0, 0, 0), useLab(false) {}             // (无参构造函数)默认构造函数
    ColorDetector(bool useLab) : maxDist(100), target(0, 0, 0), useLab(useLab) {} // Lab color space
    ColorDetector(uchar blue, uchar green, uchar red, int mxDist = 100, bool useLab = false)
        : maxDist(mxDist), useLab(useLab)
    {
        // [B-G-R]
        setTargetColor(blue, green, red);
    }

    // 默认析构函数 编译成为静态库，不要要析构函数？？？
    // colorDetection.cpp:(.text+0x102e): undefined reference to `ColorDetector::~ColorDetector()'
    // ~ColorDetector();

    // get 方法和 set 方法
    cv::Vec3b getTargetColor() const
    {
        return target;
    }
    void setTargetColor(cv::Vec3b color)
    {
        target = color;
    }
    // 重载版本 [B-G-R] color space
    void setTargetColor(uchar blue, uchar green, uchar red)
    {
        target = cv::Vec3b(blue, green, red);

        // RGB color space convert to Lab color space
        if (useLab)
        {
            cv::Mat temp_img(1, 1, CV_8UC3);
            temp_img.at<cv::Vec3b>(0, 0) = cv::Vec3b(blue, green, red);
            // 转换到 Lab 颜色空间 : enum cv::ColorConversionCodes
            cv::cvtColor(temp_img, temp_img, cv::COLOR_BGR2Lab);
            target = temp_img.at<cv::Vec3b>(0, 0);
        }
    }

    // 设置颜色距离阈值 threshold
    // Threshold must be positive, otherwise distance threshold is set to 0.
    void setColorDistanceThreshold(int distance)
    {
        if (distance < 0)
        {
            distance = 0;
        }
        maxDist = distance;
    }
    int getColorDistanceThreshold() const
    {
        return maxDist;
    }

    // 度量与目标颜色的距离
    int getDistanceToTargetColor(const cv::Vec3b &color) const
    {
        return getColorDistance(color, target);
    }
    // 利用城市街区来度量距离(三个颜色通道直接累加) (也可以采用欧氏距离来度量)
    int getColorDistance(const cv::Vec3b &color1, const cv::Vec3b &color2) const
    {
        return std::abs(color1[0] - color2[0]) + std::abs(color1[1] - color2[1]) + std::abs(color1[2] - color2[2]);

        // 换一种写法
        // cv::Vec3b dist;
        // cv::absdiff(color1, color2, dist);
        // return cv::sum(dist)[0];

        // 欧氏距离度量
        // return static_cast<int>(cv::norm<int,3>(
        //     cv::Vec3i(color1[0] - color2[0], color1[1] - color2[1], color1[2] - color2[2])));
    }

    // 图像处理，获取一个通道的二值化图像
    cv::Mat process(const cv::Mat &image);

    // () 运算符重载
    cv::Mat operator()(const cv::Mat &image)
    {
        cv::Mat input;
        if (useLab)
        {
            cv::cvtColor(image, input, cv::COLOR_BGR2Lab);
        }
        else
        {
            input = image;
        }

        cv::Mat output;
        // compute absolute difference with target color
        cv::absdiff(input, cv::Scalar(target), output);
        // split the channels into 3 images
        std::vector<cv::Mat> images;
        cv::split(output, images);
        // add the 3 channels (saturation might occurs here)
        output = images[0] + images[1] + images[2];
        // apply threshold
        cv::threshold(output,                 // input image
                      output,                 // output image
                      maxDist,                // threshold (must be < 256)
                      255,                    // max value
                      cv::THRESH_BINARY_INV); // thresholding type

        return output;
    }

private:
    // 算法允许的最小差距(该像素的颜色与目标颜色之间的距离差)
    int maxDist;
    // 目标颜色
    cv::Vec3b target;
    // 存储二值化结果的图像
    cv::Mat result;
    // 颜色空间转换后的图像
    cv::Mat converted;
    // 是否进行了 Lab 颜色空间
    bool useLab;
};

#endif // COLOR_DETECTOR