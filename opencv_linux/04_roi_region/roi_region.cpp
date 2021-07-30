/**
 * @File    : roi_region.cpp
 * @Brief   : 感兴趣区域 Region Of Interest， ROI
 * @Author  : Wei Li
 * @Date    : 2021-07-25
*/

#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>


// -----------------------------
int main(int argc, char** argv)
{
    cv::Mat image = cv::imread("./../images/puppy.bmp");
    if (image.empty())
    {
        std::cout << "Error reading image file for image." << std::endl;
        return 1;
    }
    // 利用 VSCode 智能插件保证路径不会写错 Path Intellisense
    cv::Mat logo = cv::imread("./../images/smalllogo.png");
    if (logo.empty())
    {
        std::cout << "Error reading image file for logo." << std::endl;
        return 1;
    }

    // define image ROI at image bottom-right.
    cv::Mat imageROI(image, cv::Rect(image.cols - logo.cols, 
                                    image.rows - logo.rows, // ROI coordinates
                                    logo.cols, logo.rows)); // ROI size
    // insert logo to image.
    logo.copyTo(imageROI);

    // void namedWindow(const string& winname, int flags=WINDOW_AUTOSIZE);
    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Image", image);
    cv::waitKey(0);

    // re-read the original image.
    image = cv::imread("./../images/puppy.bmp");
    if (image.empty())
    {
        std::cout << "Error reading image file." << std::endl;
        return 1;
    }

    // imageROI= image(cv::Rect(image.cols - logo.cols, image.rows - logo.rows,
	// 	                     logo.cols, logo.rows));
    imageROI = image(cv::Range(image.rows - logo.rows, image.rows),
                     cv::Range(image.cols - logo.cols, image.cols));

    // use the logo as a mask (must be gray-level) 图像掩码操作
    cv::Mat mask(logo);
    logo.copyTo(imageROI, mask);
    cv::imshow("Image", image);
    cv::waitKey(0);

    // 关闭特定窗口
    // void destroyWindow(const string& winname);
    // 关闭所有的 HighGUI 窗口
    // void destroyAllWindows();
    cv::destroyAllWindows();
    
    return 0;
}
