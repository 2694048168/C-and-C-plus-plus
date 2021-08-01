/**
 * @File    : blobs.cpp
 * @Brief   : 提取组件的轮廓 & 计算组件的形状描述符 
 * Extract continuous area Calculate the shape descriptor of the area
 * @Author  : Wei Li
 * @Date    : 2021-07-31
*/

#include <iostream>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// --------------------------------
int main(int argc, char **argv)
{
    cv::Mat image = cv::imread("./../images/binaryGroup.bmp", 0);
    if (!image.data)
    {
        std::cerr << "--Error reading image files." << std::endl;
        return 1;
    }
    std::string win_name = "Binary Image";
    cv::namedWindow(win_name);
    cv::imshow(win_name, image);

    // 提取出图像中连续区域的轮廓
    // 用于存储轮廓的向量
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(image,
                     contours,               // 存储轮廓的向量
                     cv::RETR_EXTERNAL,      // 检索外部轮廓
                     cv::CHAIN_APPROX_NONE); // 每个轮廓的全部像素

    // Print contours' length
    std::cout << "Contours: " << contours.size() << std::endl;
    std::vector<std::vector<cv::Point>>::const_iterator itContours = contours.begin();
    for (; itContours != contours.end(); ++itContours)
    {
        std::cout << "Size: " << itContours->size() << std::endl;
    }

    // 在白色图像上画黑色轮廓
    cv::Mat result(image.size(), CV_8U, cv::Scalar(255));
    cv::drawContours(result, contours,
                     -1, // 画全部轮廓
                     0,  // 用黑色画
                     2); // 宽度为 2
    cv::String win_name_con = "Contours";
    cv::namedWindow(win_name_con);
    cv::imshow(win_name_con, result);

    // ------------------------------------------------------------------------
    // 提取轮廓的算法很简单，它系统地扫描图像，直到找到连续区域
    // 从区域的起点开始，沿着它的轮廓对边界像素做标记
    // 处理完这个轮廓后，就从上个位置继续扫描，直到发现新的区域
    // 也可以对识别出的连续区域进行独立的分析。例如知道感兴趣物体的大小，就可以将部分区域删除
    // 采用区域边界的最小值和最大值，具体做法是迭代遍历存放轮廓的向量，并且删除无效的轮廓：
    // ------------------------------------------------------------------------
    // 删除太短或太长的轮廓
    int cmin = 50;   // 最小轮廓长度
    int cmax = 1000; // 最大轮廓长度
    std::vector<std::vector<cv::Point>>::
        iterator itc = contours.begin();
    // 针对所有轮廓
    while (itc != contours.end())
    {
        // 验证轮廓大小
        if (itc->size() < cmin || itc->size() > cmax)
            itc = contours.erase(itc);
        else
            ++itc;
    }

    // 在原始图像上绘制连续的区域
    cv::Mat original = cv::imread("./../images/group.jpg");
    if (!original.data)
    {
        std::cerr << "--Error reading image file unsuccfully." << std::endl;
        return 1;
    }

    cv::drawContours(original, contours,
                     -1,                        // draw all contours
                     cv::Scalar(255, 255, 255), // in white
                     2);                        // with a thickness of 2
    cv::namedWindow("Contours on Animals");
    cv::imshow("Contours on Animals", original);

    // Let's now draw black contours on white image
    result.setTo(cv::Scalar(255));
    cv::drawContours(result, contours, -1, 0, 1);
    cv::namedWindow("Contours on Animals[result]");
    cv::imshow("Contours on Animals[result]", result);

    // 计算区域的形状描述子
    // 连续区域通常代表着场景中的某个物体
    // 为了识别该物体，或将它与其他图像元素做比较，需要对此区域进行测量，以提取出部分特征
    image = cv::imread("./../images/binaryGroup.bmp", 0);
    if (!image.data)
    {
        std::cerr << "--Error reading image file unsuccessfunlly." << std::endl;
        return 1;
    }

    // 测试边界框 bounding box
    cv::Rect r0 = cv::boundingRect(contours[0]);
    // 画矩形
    cv::rectangle(result, r0, 0, 2);

    // 测试覆盖圆 enclosing circle
    float radius;
    cv::Point2f center;
    cv::minEnclosingCircle(contours[1], center, radius);
    // 画圆形
    cv::circle(result, center, static_cast<int>(radius), cv::Scalar(0), 2);

    // 测试多边形逼近 approximate polygon
    std::vector<cv::Point> poly;
    cv::approxPolyDP(contours[2], poly, 5, true);
    // 画多边形
    cv::polylines(result, poly, true, 0, 2);
    std::cout << "Polygon size: " << poly.size() << std::endl;

    // 测试凸包 convex hull
    std::vector<cv::Point> hull;
    cv::convexHull(contours[3], hull);
    // 画多边形
    cv::polylines(result, hull, true, 0, 2);

    // 计算轮廓矩是另一种功能强大的描述子（在所有区域内部画出重心）：
    // 测试轮廓矩
    // 迭代遍历所有轮廓
    itc = contours.begin();
    while (itc != contours.end())
    {
        // 计算所有轮廓矩
        cv::Moments mom = cv::moments(cv::Mat(*itc++));
        // 画重心
        cv::circle(result,
                   // 将重心位置转换成整数
                   cv::Point(mom.m10 / mom.m00, mom.m01 / mom.m00),
                   2, cv::Scalar(0), 2); // 画黑点
    }
    cv::namedWindow("Some Shape descriptors");
    cv::imshow("Some Shape descriptors", result);

    // cv::findContours 函数也能检测二值图像中的所有闭合轮廓，包括区域内部空穴构成的轮廓
    // 实现方法是在调用函数时指定另一个标志：
    image = cv::imread("./../images/binaryGroup.bmp", 0);
    if (!image.data)
    {
        std::cerr << "--Error reading image file unsuccessfunlly." << std::endl;
        return 1;
    }
    // Get the contours of the connected components
    cv::findContours(image,
                     contours,               // 存放轮廓的向量
                     cv::RETR_LIST,          // 检索全部轮廓
                     cv::CHAIN_APPROX_NONE); // 全部像素

    // draw black contours on white image
    result.setTo(255);
    cv::drawContours(result, contours, -1, 0, 2);
    cv::namedWindow("All Contours");
    cv::imshow("All Contours", result);
    /**注意，背景森林中增加了额外的轮廓
     * 可以把这些轮廓分层次组织起来 主区域是父轮廓，它内部的空穴是子轮廓；
     * 如果空穴内部还有区域，那它们就是上述子轮廓的子轮廓，以此类推
     * 使用 cv::RETR_TREE 标志可得到这个层次结构
     */
    // std::vector<cv::Vec4i> hierarchy;
    // cv::findContours(image, contours,        // 存放轮廓的向量
    //                  hierarchy,              // 层次结构
    //                  cv::RETR_TREE,          // 树状结构的轮廓
    //                  cv::CHAIN_APPROX_NONE); // 每个轮廓的全部像素

    // get a MSER image (或者使用形态学滤波器来清理图像)
    cv::Mat components;
    components = cv::imread("./../images/mser.bmp", 0);
    // create a binary version 创建二值图像
    components = components == 255;
    // open the image (white background) 打开图像（包含背景）
    cv::morphologyEx(components, components, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1, -1), 3);
    cv::namedWindow("MSER image");
    cv::imshow("MSER image", components);

    contours.clear();
    // 翻转图像（背景必须是黑色的）
    cv::Mat componentsInv = 255 - components;
    // 得到连续区域的轮廓
    cv::findContours(componentsInv,
                     contours,               // 轮廓的向量
                     cv::RETR_EXTERNAL,      // 检索外部轮廓
                     cv::CHAIN_APPROX_NONE); // // retrieve all pixels of each contours

    // white image
    cv::Mat quadri(components.size(), CV_8U, 255);

    // 针对全部轮廓
    std::vector<std::vector<cv::Point>>::iterator it = contours.begin();
    while (it != contours.end())
    {
        poly.clear();
        // 用多边形逼近轮廓
        cv::approxPolyDP(*it, poly, 10, true);
        // 是否为四边形？
        if (poly.size() == 4)
        {
            // 画出来
            cv::polylines(quadri, poly, true, 0, 2);
        }
        ++it;
    }
    cv::namedWindow("MSER quadrilateral");
    cv::imshow("MSER quadrilateral", quadri);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
