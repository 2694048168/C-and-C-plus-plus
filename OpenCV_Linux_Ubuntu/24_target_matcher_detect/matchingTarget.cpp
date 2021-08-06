/**
 * @File    : matchingTarget.cpp
 * @Brief   : 检测图像中的平面目标
 * @Author  : Wei Li
 * @Date    : 2021-08-06
*/

#include "targetMatcher.hpp"

// ------------------------------
int main(int argc, char **argv)
{
    cv::Mat target = cv::imread("./../images/cookbook1.bmp", 0);
    cv::Mat image = cv::imread("./../images/objects.jpg", 0);
    if (!target.data || !image.data)
    {
        std::cerr << "--Error reading image files unsuccessfully." << std::endl;
        return 1;
    }
    cv::namedWindow("Target");
    cv::imshow("Target", target);
    cv::namedWindow("Image");
    cv::imshow("Image", image);

    // Prepare the matcher
    TargetMatcher tmatcher(cv::FastFeatureDetector::create(10), cv::BRISK::create());
    tmatcher.setNormType(cv::NORM_HAMMING);

    // definition of the output data
    std::vector<cv::DMatch> matches;
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    std::vector<cv::Point2f> corners;

    // set the target image
    tmatcher.setTarget(target);

    // match image with target
    tmatcher.detectTarget(image, corners);
    // draw the target corners on the image
    if (corners.size() == 4)
    { // we have a detection
        cv::line(image, cv::Point(corners[0]), cv::Point(corners[1]), cv::Scalar(255, 255, 255), 3);
        cv::line(image, cv::Point(corners[1]), cv::Point(corners[2]), cv::Scalar(255, 255, 255), 3);
        cv::line(image, cv::Point(corners[2]), cv::Point(corners[3]), cv::Scalar(255, 255, 255), 3);
        cv::line(image, cv::Point(corners[3]), cv::Point(corners[0]), cv::Scalar(255, 255, 255), 3);
    }
    cv::namedWindow("Target detection");
    cv::imshow("Target detection", image);

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
