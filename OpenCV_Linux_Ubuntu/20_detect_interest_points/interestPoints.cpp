/**
 * @File    : interestPoints.cpp
 * @Brief   : 检测图像中的角点；快速检测特征；尺度不变特征的检测；多尺度 FAST 特征的检测
 * @Author  : Wei Li
 * @Date    : 2021-08-01
*/

#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "harrisDetector.hpp"

// -------------------------------
int main(int argc, char **argv)
{
    // ---------------- Harris ----------------
    const cv::String filename = "./../images/church01.jpg";
    cv::Mat image = cv::imread(filename, 0);
    if (!image.data)
    {
        std::cerr << "--Error reading image file unsuccessfully." << std::endl;
        return 1;
    }

    // rotate the image (to produce a horizontal image)
    cv::transpose(image, image);
    cv::flip(image, image, 0);
    const cv::String winname = "OriginalImage";
    cv::namedWindow(winname);
    cv::imshow(winname, image);

    // Detect Harris corners
    cv::Mat cornerStrength;
    // 输入图像 角点强度的图像 邻域尺寸 口径尺寸 Harris 参数
    cv::cornerHarris(image, cornerStrength, 3, 3, 0.01);
    // 对角点强度阈值化
    cv::Mat harrisCorners;
    double threshold = 0.0001;
    cv::threshold(cornerStrength, harrisCorners, threshold, 255, cv::THRESH_BINARY_INV);
    cv::namedWindow("Harris");
    cv::imshow("Harris", harrisCorners);

    // 实例化对象
    HarrisDetector harris;
    // 计算 Harris 值
    harris.detect(image);
    // 检测 Harris 角点
    std::vector<cv::Point> pts;
    harris.getCorners(pts, 0.02);
    // 绘制检测的角点
    harris.drawOnImage(image, pts);
    cv::namedWindow("Corners");
    cv::imshow("Corners", image);

    // ---------------- GFTT ----------------
    //  OpenCV 中用 good-features-to-track（GFTT）实现这个算法
    // 这个算法得名于它检测的特征非常适合作为视觉跟踪程序的起始集合
    image = cv::imread(filename, 0);
    if (!image.data)
    {
        std::cerr << "--Error reading image file unsuccessfully." << std::endl;
        return 1;
    }
    // rotate the image (to produce a horizontal image)
    cv::transpose(image, image);
    cv::flip(image, image, 0);

    // Compute good features to track 计算适合跟踪的特征
    std::vector<cv::KeyPoint> keypoints;
    // // GFTT 检测器
    cv::Ptr<cv::GFTTDetector> ptrGFTT = cv::GFTTDetector::create(500,  // 关键点的最大数量
                                                                 0.01, // 质量等级
                                                                 10);  // 角点之间允许的最短距离

    // 检测 GFTT
    ptrGFTT->detect(image, keypoints);
    std::vector<cv::KeyPoint>::const_iterator it = keypoints.begin();
    while (it != keypoints.end())
    {
        // draw a circle at each corner location
        cv::circle(image, it->pt, 3, cv::Scalar(255, 255, 255), 1);
        ++it;
    }
    cv::namedWindow("GFTT");
    cv::imshow("GFTT", image);

    // 查看适合目标追踪的关键点
    image = cv::imread(filename, 0);
    if (!image.data)
    {
        std::cerr << "--Error reading image file unsuccessfully." << std::endl;
        return 1;
    }
    // rotate the image (to produce a horizontal image)
    cv::transpose(image, image);
    cv::flip(image, image, 0);

    // draw the keypoints
    cv::drawKeypoints(image,                                   // original image
                      keypoints,                               // vector of keypoints
                      image,                                   // the resulting image
                      cv::Scalar(255, 255, 255),               // color of the points
                      cv::DrawMatchesFlags::DRAW_OVER_OUTIMG); //drawing flag

    cv::namedWindow("Good Features to Track Detector");
    cv::imshow("Good Features to Track Detector", image);

    // ---------------- FAST ----------------
    // 特征点算子，叫作 FAST（Features from Accelerated Segment Test， 加速分割测试获得特征）
    // 这种算子专门用来快速检测兴趣点——只需对比几个像素，就可以判断它是否为关键点
    image = cv::imread(filename, 0);
    if (!image.data)
    {
        std::cerr << "--Error reading image file unsuccessfully." << std::endl;
        return 1;
    }
    // rotate the image (to produce a horizontal image)
    cv::transpose(image, image);
    cv::flip(image, image, 0);

    keypoints.clear();
    // FAST detector
    cv::Ptr<cv::FastFeatureDetector> ptrFAST = cv::FastFeatureDetector::create(40);
    // detect the keypoints
    ptrFAST->detect(image, keypoints);
    // draw the keypoints
    cv::drawKeypoints(image, keypoints, image, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    std::cout << "Number of keypoints (FAST): " << keypoints.size() << std::endl;
    cv::namedWindow("FAST");
    cv::imshow("FAST", image);

    // FAST feature without non-max suppression
    image = cv::imread(filename, 0);
    if (!image.data)
    {
        std::cerr << "--Error reading image file unsuccessfully." << std::endl;
        return 1;
    }
    // rotate the image (to produce a horizontal image)
    cv::transpose(image, image);
    cv::flip(image, image, 0);

    keypoints.clear();
    // detect the keypoints
    ptrFAST->setNonmaxSuppression(false);
    ptrFAST->detect(image, keypoints);
    // draw the keypoints
    cv::drawKeypoints(image, keypoints, image, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    cv::namedWindow("FAST Features (all)");
    cv::imshow("FAST Features (all)", image);

    // 应用程序不同，检测特征点时采用的策略也不同
    // 例如在事先明确兴趣点数量的情况下，可以对检测过程进行动态适配
    // 简单的做法是采用范围较大的阈值检测出很多兴趣点，然后从中提取出 n 个强度最大的
    image = cv::imread(filename, 0);
    if (!image.data)
    {
        std::cerr << "--Error reading image file unsuccessfully." << std::endl;
        return 1;
    }
    // rotate the image (to produce a horizontal image)
    cv::transpose(image, image);
    cv::flip(image, image, 0);

    int total(100);         // requested number of keypoints
    int hstep(5), vstep(3); // a grid of 4 columns by 3 rows
    // hstep= vstep= 1; // try without grid
    int hsize(image.cols / hstep), vsize(image.rows / vstep);
    int subtotal(total / (hstep * vstep)); // number of keypoints per grid
    cv::Mat imageROI;
    std::vector<cv::KeyPoint> gridpoints;

    std::cout << "Grid of " << vstep << " by " << hstep << " each of size " << vsize << " by " << hsize << std::endl;

    // detection with low threshold
    ptrFAST->setThreshold(20);
    // non-max suppression
    ptrFAST->setNonmaxSuppression(true);

    // The final vector of keypoints 最终的关键点容器
    keypoints.clear();
    // detect on each grid 检测每个网格
    for (int i = 0; i < vstep; i++)
        for (int j = 0; j < hstep; j++)
        {
            // create ROI over current grid
            imageROI = image(cv::Rect(j * hsize, i * vsize, hsize, vsize));
            // detect the keypoints in grid 在网格中检测关键点
            gridpoints.clear();
            ptrFAST->detect(imageROI, gridpoints);
            std::cout << "Number of FAST in grid " << i << "," << j << ": " << gridpoints.size() << std::endl;
            if (gridpoints.size() > subtotal)
            {
                for (auto it = gridpoints.begin(); it != gridpoints.begin() + subtotal; ++it)
                {
                    std::cout << "  " << it->response << std::endl;
                }
            }

            // get the strongest FAST features 获取强度最大的 FAST 特征
            auto itEnd(gridpoints.end());
            if (gridpoints.size() > subtotal)
            { // select the strongest features 选取最强的特征
                std::nth_element(gridpoints.begin(), gridpoints.begin() + subtotal, gridpoints.end(),
                                 [](cv::KeyPoint &a, cv::KeyPoint &b)
                                 { return a.response > b.response; });
                itEnd = gridpoints.begin() + subtotal;
            }

            // add them to the global keypoint vector 加入全局特征容器
            for (auto it = gridpoints.begin(); it != itEnd; ++it)
            {
                // 转换成图像上的坐标
                it->pt += cv::Point2f(j * hsize, i * vsize);
                keypoints.push_back(*it);
                std::cout << "  " << it->response << std::endl;
            }
        }

    // draw the keypoints
    cv::drawKeypoints(image, keypoints, image, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    cv::namedWindow("FAST Features (grid)");
    cv::imshow("FAST Features (grid)", image);

    // ---------------- FAST ----------------
    /**计算机视觉界引入了尺度不变特征的概念
     * 它的理念是，不仅在任何尺度下拍摄的物体都能检测到一致的关键点，
     * 而且每个被检测的特征点都对应一个尺度因子。
     * 
     * 理想情况下，对于两幅图像中不同尺度的同一个物体点，
     * 计算得到的两个尺度因子之间的比率应该等于图像尺度的比率
     */
    // SURF 特征，它的全称为加速稳健特征（Speeded Up Robust Feature）
    // 它们不仅是尺度不变特征，而且是具有较高计算效率的特征
    image = cv::imread(filename, 0);
    if (!image.data)
    {
        std::cerr << "--Error reading image file unsuccessfully." << std::endl;
        return 1;
    }
    // rotate the image (to produce a horizontal image)
    cv::transpose(image, image);
    cv::flip(image, image, 0);
    keypoints.clear();

    // Construct the SURF feature detector object
    cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> ptrSURF = cv::xfeatures2d::SurfFeatureDetector::create(2000.0);
    // detect the keypoints
    ptrSURF->detect(image, keypoints);

    // Detect the SURF features
    ptrSURF->detect(image, keypoints);

    cv::Mat featureImage;
    cv::drawKeypoints(image, keypoints, featureImage, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    std::cout << "Number of SURF keypoints: " << keypoints.size() << std::endl;
    cv::namedWindow("SURF");
    cv::imshow("SURF", featureImage);

    image = cv::imread("./../images/church03.jpg", cv::IMREAD_GRAYSCALE);
    if (!image.data)
    {
        std::cerr << "--Error reading image file unsuccessfully." << std::endl;
        return 1;
    }
    // rotate the image (to produce a horizontal image)
    cv::transpose(image, image);
    cv::flip(image, image, 0);
    // Detect the SURF features
    ptrSURF->detect(image, keypoints);
    cv::drawKeypoints(image, keypoints, featureImage, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::namedWindow("SURF (2)");
    cv::imshow("SURF (2)", featureImage);

    // ---------------- FAST ----------------
    /**SURF 算法是 SIFT 算法的加速版，
     * 而 SIFT（Scale-Invariant Feature Transform， 尺度不变特征转换）是另一种著名的尺度不变特征检测法
     * 
     * SIFT 检测特征时也采用了图像空间和尺度空间的局部最大值，但它使用拉普拉斯滤波器响应，而不是 Hessian 行列式值
     * 这个拉普拉斯算子是利用高斯滤波器的差值，在不同尺度（即逐步加大 σ 值）下计算得到的。
     * 为了提高性能， σ 值每翻一倍，图像的尺寸就缩小一半
     * 每个金字塔级别代表一个八度（ octave），每个尺度是一图层（ layer）
     * 一个八度通常有三个图层
     */
    image = cv::imread("./../images/church01.jpg", cv::IMREAD_GRAYSCALE);
    if (!image.data)
    {
        std::cerr << "--Error reading image file unsuccessfully." << std::endl;
        return 1;
    }
    cv::transpose(image, image);
    cv::flip(image, image, 0);
    keypoints.clear();

    // Construct the SIFT feature detector object
    cv::Ptr<cv::SIFT> ptrSIFT = cv::SIFT::create();
    // detect the keypoints
    ptrSIFT->detect(image, keypoints);
    // Detect the SIFT features
    ptrSIFT->detect(image, keypoints);
    cv::drawKeypoints(image, keypoints, featureImage, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    std::cout << "Number of SIFT keypoints: " << keypoints.size() << std::endl;
    cv::namedWindow("SIFT");
    cv::imshow("SIFT", featureImage);

    // ---------------- BRISK ----------------
    /**FAST 是一种快速检测图像关键点的方法
     * 使用 SURF 和 SIFT 算法时，侧重点在于设计尺度不变特征 而再之后提出的兴趣点检测新方法既能快速检测，又不随尺度改变而变化
     * 
     * BRISK（Binary Robust Invariant Scalable Keypoints， 二元稳健恒定可扩展关键点）检测法，
     * 它基于上一节介绍的 FAST 特征检测法
     * 
     * 另一种检测方法 ORB（Oriented FAST and Rotated BRIEF，定向 FAST 和旋转 BRIEF）
     * 在需要进行快速可靠的图像匹配时，这两种特征点检测法是非常优秀的解决方案
     * 如果能搭配上相关的二值描述子，它们的性能能进一步提高
     */
    image = cv::imread("./../images/church01.jpg", cv::IMREAD_GRAYSCALE);
    if (!image.data)
    {
        std::cerr << "--Error reading image file unsuccessfully." << std::endl;
        return 1;
    }
    cv::transpose(image, image);
    cv::flip(image, image, 0);
    cv::imshow("BRISK", featureImage);

    keypoints.clear();
    // Construct another BRISK feature detector object
    cv::Ptr<cv::BRISK> ptrBRISK = cv::BRISK::create(
        60, // threshold for BRISK points to be accepted
        5); // number of octaves

    // Detect the BRISK features
    ptrBRISK->detect(image, keypoints);
    cv::drawKeypoints(image, keypoints, featureImage, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    std::cout << "Number of BRISK keypoints: " << keypoints.size() << std::endl;
    cv::namedWindow("BRISK");
    cv::imshow("BRISK", featureImage);

    // ---------------- ORB ----------------
    image = cv::imread("./../images/church01.jpg", cv::IMREAD_GRAYSCALE);
    if (!image.data)
    {
        std::cerr << "--Error reading image file unsuccessfully." << std::endl;
        return 1;
    }
    // rotate the image (to produce a horizontal image)
    cv::transpose(image, image);
    cv::flip(image, image, 0);

    keypoints.clear();
    // Construct the BRISK feature detector object
    cv::Ptr<cv::ORB> ptrORB = cv::ORB::create(
        75,  // total number of keypoints
        1.2, // scale factor between layers
        8);  // number of layers in pyramid
    // detect the keypoints
    ptrORB->detect(image, keypoints);
    cv::drawKeypoints(image, keypoints, featureImage, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    std::cout << "Number of ORB keypoints: " << keypoints.size() << std::endl;
    cv::namedWindow("ORB");
    cv::imshow("ORB", featureImage);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
