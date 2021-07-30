/**
 * @File    : grabcut_segmentation.cpp
 * @Brief   : GrabCut 图像分割图像前景和背景
 * @Author  : Wei Li
 * @Date    : 2021-07-28
*/

#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// -------------------------------
int main(int argc, char **argv)
{
    cv::String filename = "./../images/boldt.jpg";
    cv::Mat image = cv::imread(filename);
    if (!image.data)
    {
        std::cerr << "Error reading image file." << std::endl;
        return 1;
    }

    const cv::String win_name_origin = "Origin Image";
    cv::namedWindow(win_name_origin);
    cv::imshow(win_name_origin, image);

    // 定义一个带边框的矩形; 矩形外部的像素会被标记为背景
    cv::Rect rectangle(50, 25, 210, 180);
    cv::Mat result;           // 分割结果（四种可能的值）
    cv::Mat bgModel, fgModel; // 模型（内部使用）
    // GrabCut 分割算法 OpenCV 封装的函数
    cv::grabCut(image,                  // 输入图像
                result,                 // 分割结果
                rectangle,              // 包含前景的矩形
                bgModel, fgModel,       // 模型
                5,                      // 迭代次数
                cv::GC_INIT_WITH_RECT); // 使用矩形

    /**分割结果（四种可能的值）
     * cv::GC_BGD：这个值表示明确属于背景的像素（例如本例中矩形之外的像素）         enum=0
     * cv::GC_FGD：这个值表示明确属于前景的像素（本例中没有这种像素）              enum=3
     * cv::GC_PR_BGD：这个值表示可能属于背景的像素                             enum=2
     * cv::GC_PR_FGD：这个值表示可能属于前景的像素（即本例中矩形之内像素的初始值）  enum=1
     * 
     * 这样可以进行位操作运算来获取需要的像素
     */
    // 取得标记为“可能属于前景”的像素
    cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
    // result = result & 1;
    cv::Mat foreground(image.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    image.copyTo(foreground, result); // 不复制背景像素

    // draw rectangle on original image
    cv::rectangle(image, rectangle, cv::Scalar(255, 255, 255), 1);
    cv::namedWindow("Image with rectangle");
    cv::imshow("Image with rectangle", image);

    // display result
    cv::namedWindow("Foreground object");
    cv::imshow("Foreground object", foreground);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
