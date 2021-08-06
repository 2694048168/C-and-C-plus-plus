/**
 * @File    : color_space.cpp
 * @Brief   : 图像颜色的表示方法(色彩空间)
 * @Author  : Wei Li
 * @Date    : 2021-07-28
*/

/**色彩空间
 * https://docs.opencv.org/4.5.3/d8/d01/group__imgproc__color__conversions.html
 * 
 * 1. RGB 色彩空间的基础是对加色法三原色（红、绿、蓝）的应用
 *    RGB 并不是感知均匀的色彩空间，两种具有一定差距的颜色可能看起来非常接近，而另外两种具有同样差距的颜色看起来却差别很大
 *    RGB 色彩空间计算颜色之间的差距并不是衡量两个颜色相似度的最好方式。
 * 
 * 2. CIE L*a*b* 色彩空间
 *   具有感知均匀特性的颜色表示法，图像像素与目标颜色之间的欧几里得距离，来度量颜色之间的视觉相似度
 * 
 * 3. YCrCb 色彩空间，它是在 JPEG 压缩中使用的色彩空间
 * 
 * 4. CIE L*u*v* 是另一种感知均匀的色彩空间
 *    实现视觉感知上的均匀，两种色彩空间都扭曲了 RGB 的颜色范围，这些转换过程都是非线性的（因此计算量巨大）。
 *    L*a*b*和 L*u*v*对亮度通道使用同样的转换公式，但对色度通道则使用不同的表示法。
 * 
 * 5. CIE XYZ 色彩空间
 *    它是一种标准色彩空间，用与设备无关的方式表示任何可见颜色。
 *    在 L*a*b*和 L*u*v*色彩空间的计算中，用 XYZ 色彩空间作为一种中间表示法。 
 *    RGB 与 XYZ 之间的转换是线性的。还有一点非常有趣，就是 Y 通道对应着图像的灰度版本
 * 
 * 6. HSV 和 HLS 这两种色彩空间 (色调、饱和度和亮度表示颜色)
 *    它们把颜色分解成加值的色调和饱和度组件或亮度组件
 *    人们用这种方式来描述的颜色会更加自然, 符合人类对于颜色的感知方式
 */

#include <iostream>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void detectHScolor(const cv::Mat &image,         // input image
                   double minHue, double maxHue, // Hue interval 色调区间
                   double minSat, double maxSat, // saturation interval 饱和度区间
                   cv::Mat &mask)                // 输出掩码
{

    // convert into HSV space
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // 将 3 个通道分割到 3 幅图像
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    // channels[0] is the Hue
    // channels[1] is the Saturation
    // channels[2] is the Value

    // Hue masking 色调掩码
    cv::Mat mask1; // below maxHue
    cv::threshold(channels[0], mask1, maxHue, 255, cv::THRESH_BINARY_INV);
    cv::Mat mask2; // over minHue
    cv::threshold(channels[0], mask2, minHue, 255, cv::THRESH_BINARY);

    cv::Mat hueMask; // hue mask
    if (minHue < maxHue)
        hueMask = mask1 & mask2;
    else // if interval crosses the zero-degree axis 如果区间穿越 0 度中轴线
        hueMask = mask1 | mask2;

    // Saturation masking 饱和度掩码
    // below maxSat
    cv::threshold(channels[1], mask1, maxSat, 255, cv::THRESH_BINARY_INV);
    // over minSat
    cv::threshold(channels[1], mask2, minSat, 255, cv::THRESH_BINARY);

    cv::Mat satMask; // saturation mask
    satMask = mask1 & mask2;

    // combined mask  组合掩码
    mask = hueMask & satMask;
}

// ------------------------------
int main(int argc, char **argv)
{
    cv::Mat image = cv::imread("./../images/boldt.jpg");
    if (!image.data)
    {
        std::cerr << "Error reading image file." << std::endl;
        return 1;
    }
    cv::namedWindow("Original image");
    cv::imshow("Original image", image);

    // convert into HSV 色彩空间
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    // channels[0] is the Hue
    // channels[1] is the Saturation
    // channels[2] is the Value
    cv::namedWindow("Value");
    cv::imshow("Value", channels[2]);
    cv::namedWindow("Saturation");
    cv::imshow("Saturation", channels[1]);
    cv::namedWindow("Hue");
    cv::imshow("Hue", channels[0]);

    // 色彩空间转换之间的计算溢出情况？？？饱和算法处理
    // image with fixed value
    cv::Mat newImage;
    cv::Mat tmp(channels[2].clone());
    // Value channel will be 255 for all pixels
    channels[2] = 255;
    // merge back the channels
    cv::merge(channels, hsv);
    // re-convert to BGR
    cv::cvtColor(hsv, newImage, cv::COLOR_HSV2BGR);

    cv::namedWindow("Fixed Value Image");
    cv::imshow("Fixed Value Image", newImage);

    // image with fixed saturation
    channels[1] = 255;
    channels[2] = tmp;
    cv::merge(channels, hsv);
    cv::cvtColor(hsv, newImage, cv::COLOR_HSV2BGR);

    cv::namedWindow("Fixed saturation");
    cv::imshow("Fixed saturation", newImage);

    // image with fixed value and fixed saturation
    channels[1] = 255;
    channels[2] = 255;
    cv::merge(channels, hsv);
    cv::cvtColor(hsv, newImage, cv::COLOR_HSV2BGR);

    cv::namedWindow("Fixed saturation/value");
    cv::imshow("Fixed saturation/value", newImage);

    // 人为的创建一些颜色变换
    // 检测人类肤色？？？
    // artificial image shown the all possible HS colors
	cv::Mat hs(128, 360, CV_8UC3);  
	for (int h = 0; h < 360; h++) {
		for (int s = 0; s < 128; s++) {
			hs.at<cv::Vec3b>(s, h)[0] = h/2;     // all hue angles
			hs.at<cv::Vec3b>(s, h)[1] = 255-s*2; // from high saturation to low
			hs.at<cv::Vec3b>(s, h)[2] = 255;     // constant value
		}
	}

	cv::cvtColor(hs, newImage, cv::COLOR_HSV2BGR);

	cv::namedWindow("Hue/Saturation");
	cv::imshow("Hue/Saturation", newImage);

	// Testing skin detection

	// read the image
	image= cv::imread("./../images/girl.jpg");
    if (!image.data)
    {
        std::cerr << "Error reading image file." << std::endl;
        return 1;
    }

	// show original image
	cv::namedWindow("Original image");
	cv::imshow("Original image",image);

	// detect skin tone
	cv::Mat mask;
	detectHScolor(image, 
		160, 10, // hue from 320 degrees to 20 degrees 
		25, 166, // saturation from ~0.1 to 0.65
		mask);

	// show masked image
	cv::Mat detected(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
	image.copyTo(detected, mask);
	cv::imshow("Detection result",detected);

	// A test comparing luminance and brightness

	// create linear intensity image
	cv::Mat linear(100,256,CV_8U);
	for (int i=0; i<256; i++) {

		linear.col(i)= i;
	}

	// create a Lab image
	linear.copyTo(channels[0]);
	cv::Mat constante(100,256,CV_8U,cv::Scalar(128));
	constante.copyTo(channels[1]);
	constante.copyTo(channels[2]);
	cv::merge(channels,image);

	// convert back to BGR
	cv::Mat brightness;
	cv::cvtColor(image,brightness, cv::COLOR_Lab2BGR);
	cv::split(brightness, channels);

	// create combined image
	cv::Mat combined(200,256, CV_8U);
	cv::Mat half1(combined,cv::Rect(0,0,256,100));
	linear.copyTo(half1);
	cv::Mat half2(combined,cv::Rect(0,100,256,100));
	channels[0].copyTo(half2);

	cv::namedWindow("Luminance vs Brightness");
	cv::imshow("Luminance vs Brightness",combined);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
