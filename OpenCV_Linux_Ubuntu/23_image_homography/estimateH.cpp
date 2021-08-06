/**
 * @File    : estimateH.cpp
 * @Brief   : 计算两幅图像之间的单应矩阵    
 * @Author  : Wei Li
 * @Date    : 2021-08-06
*/

#include <iostream>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/stitching.hpp>

// -----------------------------
int main(int argc, char **argv)
{
    cv::Mat image1 = cv::imread("./../images/parliament1.jpg", 0);
    cv::Mat image2 = cv::imread("./../images/parliament2.jpg", 0);
    if (!image1.data || !image2.data)
    {
        std::cerr << "--Error reading image files unsuccessfully." << std::endl;
        return 1;
    }
    cv::namedWindow("Image 1");
    cv::imshow("Image 1", image1);
    cv::namedWindow("Image 2");
    cv::imshow("Image 2", image2);

    // vector of keypoints and descriptors
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    cv::Mat descriptors1, descriptors2;

    // 1. Construction of the SIFT feature detector
    cv::Ptr<cv::SIFT> ptrFeature2D = cv::SIFT::create(74);
    // 2. Detection of the SIFT features and associated descriptors
    ptrFeature2D->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
    ptrFeature2D->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
    std::cout << "Number of feature points (fig1): " << keypoints1.size() << std::endl;
    std::cout << "Number of feature points (fig2): " << keypoints2.size() << std::endl;

    // 3. Match the two image descriptors
    // Construction of the matcher with crosscheck
    cv::BFMatcher matcher(cv::NORM_L2, true);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    std::cout << "Number of matching points (fig1 & fig2): " << matches.size() << std::endl;
    // draw the matches
    cv::Mat imageMatches;
    cv::drawMatches(image1, keypoints1, image2, keypoints2,
                    matches, imageMatches,
                    cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::namedWindow("Matches (pure rotation case)");
    cv::imshow("Matches (pure rotation case)", imageMatches);

    // Convert keypoints into Point2f
    std::vector<cv::Point2f> points1, points2;
    for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
    {
        // Get the position of left keypoints
        float x = keypoints1[it->queryIdx].pt.x;
        float y = keypoints1[it->queryIdx].pt.y;
        points1.push_back(cv::Point2f(x, y));
        // Get the position of right keypoints
        x = keypoints2[it->trainIdx].pt.x;
        y = keypoints2[it->trainIdx].pt.y;
        points2.push_back(cv::Point2f(x, y));
    }
    std::cout << "Convert keypoints into Point2f: " << points1.size() << " " << points2.size() << std::endl;

    // 找到第一幅图像和第二幅图像之间的单应矩阵
    std::vector<char> inliers;
    cv::Mat homography = cv::findHomography(points1, points2, // 对应的点
                                            inliers,          // 输出的局内匹配项
                                            cv::RANSAC,       // RANSAC 方法
                                            1.);              // 到重复投影点的最大距离

    // Draw the inlier points
    cv::drawMatches(image1, keypoints1, image2, keypoints2,
                    matches, imageMatches,
                    cv::Scalar(0, 0, 255), cv::Scalar(0, 0, 255),
                    inliers, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::namedWindow("Homography inlier points");
    cv::imshow("Homography inlier points", imageMatches);

    // 将第一幅图像扭曲到第二幅图像
    cv::Mat result;
    cv::warpPerspective(image1,                                  // 输入图像
                        result,                                  // 输出图像
                        homography,                              // 单应矩阵
                        cv::Size(2 * image1.cols, image1.rows)); // 输出图像的尺寸

    // 把第一幅图像复制到完整图像的第一个半边
    cv::Mat half(result, cv::Rect(0, 0, image2.cols, image2.rows));
    image2.copyTo(half); // 把 image2 复制到 image1 的 ROI 区域
    cv::namedWindow("Image mosaic");
    cv::imshow("Image mosaic", result);

    // OpenCV 中的 contrib 包提供了完整的图像拼接方法，可以用多幅图像生成高质量的全景图
    // 用 cv::Stitcher 生成全景图
    std::vector<cv::Mat> images;
    images.push_back(cv::imread("./../images/parliament1.jpg"));
    images.push_back(cv::imread("./../images/parliament2.jpg"));

    // 输出的全景图
    cv::Mat panorama;
    // 创建拼接器
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create();
    // 拼接图像
    cv::Stitcher::Status status = stitcher->stitch(images, panorama);
    if (status == cv::Stitcher::OK)
    {
        cv::namedWindow("Panorama");
        cv::imshow("Panorama", panorama);
    }

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
