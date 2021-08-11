/**
 * @File    : CameraCalibrator.hpp
 * @Brief   : 相机标定
 * @Author  : Wei Li
 * @Date    : 2021-08-08
*/
#ifndef CAMERACALIBRATOR_H
#define CAMERACALIBRATOR_H

#include <vector>
#include <iostream>

#include <opencv2/core.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/highgui.hpp>

class CameraCalibrator
{

    // input points:
    // the points in world coordinates
    // (each square is one unit)
    std::vector<std::vector<cv::Point3f>> objectPoints;
    // the image point positions in pixels
    std::vector<std::vector<cv::Point2f>> imagePoints;
    // output Matrices
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    // flag to specify how calibration is done
    int flag;
    // used in image undistortion
    cv::Mat map1, map2;
    bool mustInitUndistort;

public:
    CameraCalibrator() : flag(0), mustInitUndistort(true) {}

    // Open the chessboard images and extract corner points
    int addChessboardPoints(const std::vector<std::string> &filelist, cv::Size &boardSize, std::string windowName = "");
    // Add scene points and corresponding image points
    void addPoints(const std::vector<cv::Point2f> &imageCorners, const std::vector<cv::Point3f> &objectCorners);
    // Calibrate the camera
    double calibrate(const cv::Size imageSize);
    // Set the calibration flag
    void setCalibrationFlag(bool radial8CoeffEnabled = false, bool tangentialParamEnabled = false);
    // Remove distortion in an image (after calibration)
    cv::Mat remap(const cv::Mat &image, cv::Size &outputSize = cv::Size(-1, -1));

    // Getters
    cv::Mat getCameraMatrix() { return cameraMatrix; }
    cv::Mat getDistCoeffs() { return distCoeffs; }
};

#endif // CAMERACALIBRATOR_H