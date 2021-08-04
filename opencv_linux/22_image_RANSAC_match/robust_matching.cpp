/**
 * @File    : robust_matching.cpp
 * @Brief   : 用 RANSAC（随机抽样一致性）算法匹配图像
 * @Author  : Wei Li
 * @Date    : 2021-08-04
*/

#include <iostream>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>

#include "matcher.hpp"

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

    // Prepare the matcher (with default parameters)
    // here SIFT detector and descriptor
    RobustMatcher rmatcher(cv::SIFT::create(250));

    // Match the two images
    std::vector<cv::DMatch> matches;

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat fundamental = rmatcher.match(image1, image2, matches, keypoints1, keypoints2);
    cv::Mat imageMatches;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, imageMatches,
                    cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::namedWindow("Matches");
    cv::imshow("Matches", imageMatches);

    // Convert keypoints into Point2f
    std::vector<cv::Point2f> points1, points2;

    for (std::vector<cv::DMatch>::const_iterator it = matches.begin();
         it != matches.end(); ++it)
    {
        // Get the position of left keypoints
        float x = keypoints1[it->queryIdx].pt.x;
        float y = keypoints1[it->queryIdx].pt.y;
        points1.push_back(keypoints1[it->queryIdx].pt);
        cv::circle(image1, cv::Point(x, y), 3, cv::Scalar(255, 255, 255), 3);
        // Get the position of right keypoints
        x = keypoints2[it->trainIdx].pt.x;
        y = keypoints2[it->trainIdx].pt.y;
        cv::circle(image2, cv::Point(x, y), 3, cv::Scalar(255, 255, 255), 3);
        points2.push_back(keypoints2[it->trainIdx].pt);
    }

    // Draw the epipolar lines
    std::vector<cv::Vec3f> lines1;
    cv::computeCorrespondEpilines(points1, 1, fundamental, lines1);

    for (std::vector<cv::Vec3f>::const_iterator it = lines1.begin();
         it != lines1.end(); ++it)
    {
        cv::line(image2, cv::Point(0, -(*it)[2] / (*it)[1]),
                 cv::Point(image2.cols, -((*it)[2] + (*it)[0] * image2.cols) / (*it)[1]),
                 cv::Scalar(255, 255, 255));
    }

    std::vector<cv::Vec3f> lines2;
    cv::computeCorrespondEpilines(points2, 2, fundamental, lines2);

    for (std::vector<cv::Vec3f>::const_iterator it = lines2.begin();
         it != lines2.end(); ++it)
    {
        cv::line(image1, cv::Point(0, -(*it)[2] / (*it)[1]),
                 cv::Point(image1.cols, -((*it)[2] + (*it)[0] * image1.cols) / (*it)[1]),
                 cv::Scalar(255, 255, 255));
    }

    // Display the images with epipolar lines
    cv::namedWindow("Right Image Epilines (RANSAC)");
    cv::imshow("Right Image Epilines (RANSAC)", image1);
    cv::namedWindow("Left Image Epilines (RANSAC)");
    cv::imshow("Left Image Epilines (RANSAC)", image2);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
