/**
 * @File    : contours.cpp
 * @Brief   : 使用 Canny 算子检测图像轮廓
 *            使用霍夫变换检测图像中的线条
 *            将一条线拟合到一组点
 * @Author  : Wei Li
 * @Date    : 2021-07-31
*/

#include <iostream>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "line_finder.hpp"
#include "edge_detector.hpp"

#define PI 3.1415926

// ---------------------------------
int main(int argc, char **argv)
{
    cv::Mat image = cv::imread("./../images/road.jpg", 0);
    if (!image.data)
    {
        std::cerr << "--Error reading road image file." << std::endl;
        return 1;
    }
    cv::namedWindow("Original Image");
    cv::imshow("Original Image", image);

    // 计算 Sobel
    EdgeDEtector ed;
    ed.computeSobel(image);
    cv::namedWindow("Sobel (orientation)");
    cv::imshow("Sobel (orientation)", ed.getSobelOrientationImage());
    bool check_writed = cv::imwrite("ori.bmp", ed.getSobelOrientationImage());
    if (!check_writed)
    {
        std::cout << "--Error writing image file to disk unsuccessfully." << std::endl;
    }

    // Display the Sobel low threshold
    cv::namedWindow("Sobel (low threshold)");
    cv::imshow("Sobel (low threshold)", ed.getBinaryMap(125));

    // Display the Sobel high threshold
    cv::namedWindow("Sobel (high threshold)");
    cv::imshow("Sobel (high threshold)", ed.getBinaryMap(350));

    // 使用 Canny 算子
    cv::Mat contours;
    cv::Canny(image, contours, 125, 350);
    cv::namedWindow("Canny Contours");
    // 注意，因为正常的结果是用非零像素表示轮廓的，
    // 所以这里在显示轮廓时做了反转处理
    cv::imshow("Canny Contours", 255 - contours);

    // 霍夫变换（ Hough transform）是一种常用于检测此类具体特征的经典算法
    // 该算法起初用于检测图像中的直线，后来经过扩展，也能检测其他简单的图像结构
    // Create a test image
    cv::Mat test(200, 200, CV_8U, cv::Scalar(0));
    cv::line(test, cv::Point(100, 0), cv::Point(200, 200), cv::Scalar(255));
    cv::line(test, cv::Point(0, 50), cv::Point(200, 200), cv::Scalar(255));
    cv::line(test, cv::Point(0, 200), cv::Point(200, 0), cv::Scalar(255));
    cv::line(test, cv::Point(200, 0), cv::Point(0, 200), cv::Scalar(255));
    cv::line(test, cv::Point(100, 0), cv::Point(100, 200), cv::Scalar(255));
    cv::line(test, cv::Point(0, 100), cv::Point(200, 100), cv::Scalar(255));
    cv::namedWindow("Test Image");
    cv::imshow("Test Image", test);
    cv::imwrite("test.bmp", test);

    // Hough transform for line detection
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(contours, lines, 1, PI / 180, 50);

    // 绘制 lines
    cv::Mat result(contours.rows, contours.cols, CV_8U, cv::Scalar(255));
    image.copyTo(result);
    std::cout << "Lines detected: " << lines.size() << std::endl;

    std::vector<cv::Vec2f>::const_iterator it = lines.begin();
    while (it != lines.end())
    {
        float rho = (*it)[0];   // 第一个元素是距离 rho
        float theta = (*it)[1]; // 第二个元素是角度 theta

        if (theta < PI / 4. || theta > 3. * PI / 4.)
        { // 垂直线（大致）
            // 直线与第一行的交叉点
            cv::Point pt1(rho / cos(theta), 0);
            // 直线与最后一行的交叉点
            cv::Point pt2((rho - result.rows * sin(theta)) / cos(theta), result.rows);
            // draw a white line
            cv::line(result, pt1, pt2, cv::Scalar(255), 1);
        }
        else
        { // 水平线（大致）
            // 直线与第一列的交叉点
            cv::Point pt1(0, rho / sin(theta));
            // 直线与最后一列的交叉点
            cv::Point pt2(result.cols,
                          (rho - result.cols * cos(theta)) / sin(theta));
            // 画白色的线
            cv::line(image, pt1, pt2, cv::Scalar(255), 1);
        }

        std::cout << "line: (" << rho << "," << theta << ")\n";
        ++it;
    }
    cv::namedWindow("Lines with Hough");
    cv::imshow("Lines with Hough", result);

    // 实例化对象
    LineFinder ld;
    // Set probabilistic Hough parameters
    ld.setLineLengthAdnGap(100, 20);
    ld.setMinVote(60);

    // 检测线条
    std::vector<cv::Vec4i> li = ld.findLines(contours);
    // auto li = ld.findLines(contours);
    ld.drawDetectedLines(image);
    cv::namedWindow("Lines with HoughP");
    cv::imshow("Lines with HoughP", image);
    /** 为解决问题并检测到线段（即包含端点的直线），提出了霍夫变换的改进版。这就是概率霍夫变换
     * 霍夫变换只是寻找图像中边缘像素的对齐区域
     * 因为有些像素只是碰巧排成了直线，所以霍夫变换可能产生错误的检测结果
     * 也可能因为多条参数相近的直线穿过了同一个像素对齐区域，而导致检测出重复的结果
     */
    std::vector<cv::Vec4i>::const_iterator it2 = li.begin();
    while (it2 != li.end())
    {

        std::cout << "(" << (*it2)[0] << "," << (*it2)[1] << ")-("
                  << (*it2)[2] << "," << (*it2)[3] << ")" << std::endl;

        ++it2;
    }

    // 点集的直线拟合
    image = cv::imread("./../images/road.jpg", 0);
    if (!image.data)
    {
        std::cerr << "--Error reading road image file." << std::endl;
        return 1;
    }
    int n = 0;
    cv::line(image, cv::Point(li[n][0], li[n][1]), cv::Point(li[n][2], li[n][3]), cv::Scalar(255), 5);
    cv::namedWindow("One line of the Image");
    cv::imshow("One line of the Image", image);

    // Extract the contour pixels of the first detected line
    cv::Mat oneline(image.size(), CV_8U, cv::Scalar(0));
    cv::line(oneline, cv::Point(li[n][0], li[n][1]), cv::Point(li[n][2], li[n][3]), cv::Scalar(255), 3);
    cv::bitwise_and(contours, oneline, oneline);
    cv::namedWindow("One line");
    cv::imshow("One line", 255 - oneline);

    std::vector<cv::Point> points;
    // Iterate over the pixels to obtain all point positions
    for (int y = 0; y < oneline.rows; y++)
    {
        uchar *rowPtr = oneline.ptr<uchar>(y);

        for (int x = 0; x < oneline.cols; x++)
        {

            // if on a contour
            if (rowPtr[x])
            {

                points.push_back(cv::Point(x, y));
            }
        }
    }

    // find the best fitting line
    cv::Vec4f line;
    cv::fitLine(points, line, cv::DIST_L2, 0, 0.01, 0.01);
    std::cout << "line: (" << line[0] << "," << line[1] << ")(" << line[2] << "," << line[3] << ")\n";

    int x0 = line[2]; // a point on the line
    int y0 = line[3];
    int x1 = x0 + 100 * line[0]; // add a vector of length 100
    int y1 = y0 + 100 * line[1];
    image = cv::imread("./../images/road.jpg", 0);
    // draw the line
    cv::line(image, cv::Point(x0, y0), cv::Point(x1, y1), 0, 2);
    cv::namedWindow("Fitted line");
    cv::imshow("Fitted line", image);

    // eliminate inconsistent lines
    ld.removeLinesOfInconsistentOrientations(ed.getOrientation(), 0.4, 0.1);

    // Display the detected line image
    image = cv::imread("./../images/road.jpg", 0);

    ld.drawDetectedLines(image);
    cv::namedWindow("Detected Lines (2)");
    cv::imshow("Detected Lines (2)", image);

    // Create a Hough accumulator
    cv::Mat acc(200, 180, CV_8U, cv::Scalar(0));

    // Choose a point
    int x = 50, y = 30;

    // loop over all angles
    for (int i = 0; i < 180; i++)
    {
        double theta = i * PI / 180.;

        // find corresponding rho value
        double rho = x * std::cos(theta) + y * std::sin(theta);
        int j = static_cast<int>(rho + 100.5);

        std::cout << i << "," << j << std::endl;

        // increment accumulator
        acc.at<uchar>(j, i)++;
    }

    // draw the axes
    cv::line(acc, cv::Point(0, 0), cv::Point(0, acc.rows - 1), 255);
    cv::line(acc, cv::Point(acc.cols - 1, acc.rows - 1), cv::Point(0, acc.rows - 1), 255);

    cv::imwrite("hough1.bmp", 255 - (acc * 100));

    // Choose a second point
    x = 30, y = 10;

    // loop over all angles
    for (int i = 0; i < 180; i++)
    {
        double theta = i * PI / 180.;
        double rho = x * cos(theta) + y * sin(theta);
        int j = static_cast<int>(rho + 100.5);

        acc.at<uchar>(j, i)++;
    }
    cv::namedWindow("Hough Accumulator");
    cv::imshow("Hough Accumulator", acc * 100);
    cv::imwrite("hough2.bmp", 255 - (acc * 100));

    // Detect circles
    image = cv::imread("./../images/chariot.jpg", 0);

    cv::GaussianBlur(image, image, cv::Size(5, 5), 1.5);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(image, circles, cv::HOUGH_GRADIENT,
                     2,       // accumulator resolution (size of the image / 2)
                     20,      // minimum distance between two circles
                     200,     // Canny high threshold
                     60,      // minimum number of votes
                     15, 50); // min and max radius

    std::cout << "Circles: " << circles.size() << std::endl;

    // Draw the circles
    image = cv::imread("./../images/chariot.jpg", 0);

    std::vector<cv::Vec3f>::const_iterator itc = circles.begin();
    while (itc != circles.end())
    {
        cv::circle(image,
                   cv::Point((*itc)[0], (*itc)[1]), // circle centre
                   (*itc)[2],                       // circle radius
                   cv::Scalar(255),                 // color
                   2);                              // thickness

        ++itc;
    }
    cv::namedWindow("Detected Circles");
    cv::imshow("Detected Circles", image);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
