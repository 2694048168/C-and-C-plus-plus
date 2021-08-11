/**
 * @File    : CameraCalibrator.cpp
 * @Brief   : OpenCV 推荐使用国际象棋棋盘的图案生成用于标定的三维场景点的集合
 * @Author  : Wei Li
 * @Date    : 2021-08-08
*/

#include "CameraCalibrator.hpp"

// Open chessboard images and extract corner points
int CameraCalibrator::addChessboardPoints(
    const std::vector<std::string> &filelist, // list of filenames containing board images
    cv::Size &boardSize,                      // size of the board
    std::string windowName)
{   // name of window to display results
    // if null, no display shown
    // the points on the chessboard
    std::vector<cv::Point2f> imageCorners;
    std::vector<cv::Point3f> objectCorners;

    // 3D Scene Points:
    // Initialize the chessboard corners
    // in the chessboard reference frame
    // The corners are at 3D location (X,Y,Z)= (i,j,0)
    for (int i = 0; i < boardSize.height; i++)
    {
        for (int j = 0; j < boardSize.width; j++)
        {

            objectCorners.push_back(cv::Point3f(i, j, 0.0f));
        }
    }

    // 2D Image points:
    cv::Mat image; // to contain chessboard image
    int successes = 0;
    // for all viewpoints
    for (int i = 0; i < filelist.size(); i++)
    {

        // Open the image
        image = cv::imread(filelist[i], 0);

        // Get the chessboard corners
        bool found = cv::findChessboardCorners(image,         // image of chessboard pattern
                                               boardSize,     // size of pattern
                                               imageCorners); // list of detected corners

        // Get subpixel accuracy on the corners
        if (found)
        {
            cv::cornerSubPix(image, imageCorners,
                             cv::Size(5, 5), // half size of serach window
                             cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::MAX_ITER +
                                                  cv::TermCriteria::EPS,
                                              30,    // max number of iterations
                                              0.1)); // min accuracy

            // If we have a good board, add it to our data
            if (imageCorners.size() == boardSize.area())
            {

                // Add image and scene points from one view
                addPoints(imageCorners, objectCorners);
                successes++;
            }
        }

        if (windowName.length() > 0 && imageCorners.size() == boardSize.area())
        {

            //Draw the corners
            cv::drawChessboardCorners(image, boardSize, imageCorners, found);
            cv::imshow(windowName, image);
            cv::waitKey(100);
        }
    }

    return successes;
}

// Add scene points and corresponding image points
void CameraCalibrator::addPoints(const std::vector<cv::Point2f> &imageCorners, const std::vector<cv::Point3f> &objectCorners)
{

    // 2D image points from one view
    imagePoints.push_back(imageCorners);
    // corresponding 3D scene points
    objectPoints.push_back(objectCorners);
}

// Calibrate the camera
// returns the re-projection error
double CameraCalibrator::calibrate(const cv::Size imageSize)
{
    // undistorter must be reinitialized
    mustInitUndistort = true;

    //Output rotations and translations
    std::vector<cv::Mat> rvecs, tvecs;

    // start calibration
    return calibrateCamera(objectPoints, // the 3D points
                           imagePoints,  // the image points
                           imageSize,    // image size
                           cameraMatrix, // output camera matrix
                           distCoeffs,   // output distortion matrix
                           rvecs, tvecs, // Rs, Ts
                           flag);        // set options
    //					,CV_CALIB_USE_INTRINSIC_GUESS);
}

// remove distortion in an image (after calibration)
cv::Mat CameraCalibrator::remap(const cv::Mat &image, cv::Size &outputSize)
{

    cv::Mat undistorted;

    if (outputSize.height == -1)
        outputSize = image.size();

    if (mustInitUndistort)
    { // called once per calibration

        cv::initUndistortRectifyMap(
            cameraMatrix, // computed camera matrix
            distCoeffs,   // computed distortion matrix
            cv::Mat(),    // optional rectification (none)
            cv::Mat(),    // camera matrix to generate undistorted
            outputSize,   // size of undistorted
            CV_32FC1,     // type of output map
            map1, map2);  // the x and y mapping functions

        mustInitUndistort = false;
    }

    // Apply mapping functions
    cv::remap(image, undistorted, map1, map2,
              cv::INTER_LINEAR); // interpolation type

    return undistorted;
}

// Set the calibration options
// 8radialCoeffEnabled should be true if 8 radial coefficients are required (5 is default)
// tangentialParamEnabled should be true if tangeantial distortion is present
void CameraCalibrator::setCalibrationFlag(bool radial8CoeffEnabled, bool tangentialParamEnabled)
{

    // Set the flag used in cv::calibrateCamera()
    flag = 0;
    if (!tangentialParamEnabled)
        flag += CV_CALIB_ZERO_TANGENT_DIST;
    if (radial8CoeffEnabled)
        flag += CV_CALIB_RATIONAL_MODEL;
}