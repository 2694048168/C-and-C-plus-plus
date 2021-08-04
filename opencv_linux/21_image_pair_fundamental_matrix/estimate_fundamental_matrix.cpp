/**
 * @File    : estimate_fundamental_matrix.cpp
 * @Brief   : 计算图像对的基础矩阵
 * @Author  : Wei Li
 * @Date    : 2021-08-02
*/

#include <iostream>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/objdetect.hpp>

// ------------------------------
int main(int argc, char **argv)
{
    cv::Mat image1 = cv::imread("./../images/church01.jpg", 0);
    cv::Mat image2 = cv::imread("./../images/church03.jpg", 0);
    if (!image1.data || !image2.data)
    {
        std::cerr << "--Error reading image file unsuccessfully." << std::endl;
        return 1;
    }
    cv::namedWindow("Right Image");
    cv::imshow("Right Image", image1);
    cv::namedWindow("Left Image");
    cv::imshow("Left Image", image2);

    // 图像的关键点和描述子
    std::vector<cv::KeyPoint> keypoint1;
    std::vector<cv::KeyPoint> keypoint2;
    cv::Mat descriptors1, descriptors2;

    // 构建 SIFT 特征检测器
    cv::Ptr<cv::SIFT> ptrFeature2D = cv::SIFT::create(74);
    // Detection of the SURF features
    ptrFeature2D->detectAndCompute(image1, cv::noArray(), keypoint1, descriptors1);
    ptrFeature2D->detectAndCompute(image2, cv::noArray(), keypoint2, descriptors2);
    std::cout << "Number of SIFT points (1): " << keypoint1.size() << std::endl;
    std::cout << "Number of SIFT points (2): " << keypoint2.size() << std::endl;

    // 绘制检测到的关键点
    cv::Mat imageKP;
    cv::drawKeypoints(image1, keypoint1, imageKP, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::namedWindow("Right SIFT Features");
    cv::imshow("Right SIFT Features", imageKP);
    cv::drawKeypoints(image2, keypoint2, imageKP, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::namedWindow("Left SIFT Features");
    cv::imshow("Left SIFT Features", imageKP);

    // 构建匹配器 类的构造函数
    cv::BFMatcher matcher(cv::NORM_L2, true);
    // 匹配两张图像的描述子
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    std::cout << "Number of matched points: " << matches.size() << std::endl;
    // 手动选择一些匹配点, 用于计算两张图像的基础矩阵
    std::vector<cv::DMatch> selMatches;
    // 确保仔细检查所选匹配项是否有效
    selMatches.push_back(matches[2]);
    selMatches.push_back(matches[5]);
    selMatches.push_back(matches[16]);
    selMatches.push_back(matches[19]);
    selMatches.push_back(matches[14]);
    selMatches.push_back(matches[24]);
    selMatches.push_back(matches[17]);

    // Draw the selected matches
    cv::Mat imageMatches;
    cv::drawMatches(image1, keypoint1, // 1st image and its keypoints
                    image2, keypoint2, // 2nd image and its keypoints
                    selMatches,        // the selected matches
                    // matches,			// the matches
                    imageMatches, // the image produced
                    cv::Scalar(255, 255, 255),
                    cv::Scalar(255, 255, 255),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS); // color of the lines
    cv::namedWindow("Matches");
    cv::imshow("Matches", imageMatches);

    // Convert 1 vector of keypoints into
    // 2 vectors of Point2f
    std::vector<int> pointIndexes1;
    std::vector<int> pointIndexes2;
    for (std::vector<cv::DMatch>::const_iterator it = selMatches.begin();
         it != selMatches.end(); ++it)
    {

        // Get the indexes of the selected matched keypoints
        pointIndexes1.push_back(it->queryIdx);
        pointIndexes2.push_back(it->trainIdx);
    }

    // Convert keypoints into Point2f
    std::vector<cv::Point2f> selPoints1, selPoints2;
    cv::KeyPoint::convert(keypoint1, selPoints1, pointIndexes1);
    cv::KeyPoint::convert(keypoint2, selPoints2, pointIndexes2);

    // check by drawing the points
    std::vector<cv::Point2f>::const_iterator it = selPoints1.begin();
    while (it != selPoints1.end())
    {

        // draw a circle at each corner location
        cv::circle(image1, *it, 3, cv::Scalar(255, 255, 255), 2);
        ++it;
    }

    it = selPoints2.begin();
    while (it != selPoints2.end())
    {

        // draw a circle at each corner location
        cv::circle(image2, *it, 3, cv::Scalar(255, 255, 255), 2);
        ++it;
    }

    // Compute F matrix from 7 matches
    cv::Mat fundamental = cv::findFundamentalMat(
        selPoints1,     // points in first image
        selPoints2,     // points in second image
        cv::FM_7POINT); // 7-point method

    std::cout << "F-Matrix size= " << fundamental.rows << "," << fundamental.cols << std::endl;
    cv::Mat fund(fundamental, cv::Rect(0, 0, 3, 3));
    // draw the left points corresponding epipolar lines in right image
    std::vector<cv::Vec3f> lines1;
    cv::computeCorrespondEpilines(
        selPoints1, // image points
        1,          // in image 1 (can also be 2)
        fund,       // F matrix
        lines1);    // vector of epipolar lines

    std::cout << "size of F matrix:" << fund.rows << "x" << fund.cols << std::endl;

    // for all epipolar lines
    for (std::vector<cv::Vec3f>::const_iterator it = lines1.begin();
         it != lines1.end(); ++it)
    {

        // draw the epipolar line between first and last column
        cv::line(image2, cv::Point(0, -(*it)[2] / (*it)[1]),
                 cv::Point(image2.cols, -((*it)[2] + (*it)[0] * image2.cols) / (*it)[1]),
                 cv::Scalar(255, 255, 255));
    }

    // draw the left points corresponding epipolar lines in left image
    std::vector<cv::Vec3f> lines2;
    cv::computeCorrespondEpilines(cv::Mat(selPoints2), 2, fund, lines2);
    for (std::vector<cv::Vec3f>::const_iterator it = lines2.begin();
         it != lines2.end(); ++it)
    {

        // draw the epipolar line between first and last column
        cv::line(image1, cv::Point(0, -(*it)[2] / (*it)[1]),
                 cv::Point(image1.cols, -((*it)[2] + (*it)[0] * image1.cols) / (*it)[1]),
                 cv::Scalar(255, 255, 255));
    }

    // combine both images
    cv::Mat both(image1.rows, image1.cols + image2.cols, CV_8U);
    image1.copyTo(both.colRange(0, image1.cols));
    image2.copyTo(both.colRange(image1.cols, image1.cols + image2.cols));

    // Display the images with points and epipolar lines
    cv::namedWindow("Epilines");
    cv::imshow("Epilines", both);

    // Convert keypoints into Point2f
    std::vector<cv::Point2f> points1, points2, newPoints1, newPoints2;
    cv::KeyPoint::convert(keypoint1, points1);
    cv::KeyPoint::convert(keypoint2, points2);
    cv::correctMatches(fund, points1, points2, newPoints1, newPoints2);
    cv::KeyPoint::convert(newPoints1, keypoint1);
    cv::KeyPoint::convert(newPoints2, keypoint2);
    cv::drawMatches(image1, keypoint1, // 1st image and its keypoints
                    image2, keypoint2, // 2nd image and its keypoints
                    matches,            // the matches
                    imageMatches,       // the image produced
                    cv::Scalar(255, 255, 255),
                    cv::Scalar(255, 255, 255),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS); // color of the lines
    cv::namedWindow("Corrected matches");
    cv::imshow("Corrected matches", imageMatches);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
