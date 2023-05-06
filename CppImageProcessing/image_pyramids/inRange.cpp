/**
 * @brief Goal
 * In this tutorial will learn how to:
 * Perform basic thresholding operations using OpenCV cv::inRange function.
 * Detect an object based on the range of pixel values in the HSV colorspace.
 *
 * @file inRange.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-03
 * @version OpenCV 4.7 examples
 *
 */

#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// global variable
extern const int max_value_H = 360 / 2;
extern const int max_value = 255;
extern const char *window_capture_name = "Video Capture";
extern const char *window_detection_name = "Object Detection";
extern int low_H = 0;
extern int low_S = 0;
extern int low_V = 0;
extern int high_H = max_value_H;
extern int high_S = max_value;
extern int high_V = max_value;

// callback function signature for Tracker in OpenCV.
/* 静态函数只能在声明它的文件当中可见, 不能被其他文件所调用,
也就是说该静态函数只能在其定义的.cpp(编译单元 compile unit)中调用,
在其它.cpp文件(编译单元 compile unit)的函数里是不能被调用的.
---------------------------------------------------------- */
static void on_low_H_thresh_trackbar(int, void *);
static void on_high_H_thresh_trackbar(int, void *);
static void on_low_S_thresh_trackbar(int, void *);
static void on_high_S_thresh_trackbar(int, void *);
static void on_low_V_thresh_trackbar(int, void *);
static void on_high_V_thresh_trackbar(int, void *);

/**
 * @brief main function
 */
int main(int argc, char const *argv[])
{
    /* cv::VideoCapture既支持视频文件的读取,也支持从摄像机中视频的读取
        1. cv::VideoCapture capture(const string& filename);
        2. cv::VideoCapture capture(int device);
        3. cv::VideoCapture capture();
    ---------------------------------------------------------- */
    // cv::VideoCapture capter(argc > 1 ? std::atoi(argv[1]) : 0);
    const char *path2vedio = "young_man_company_work.mp4";
    cv::VideoCapture capter(path2vedio);
    if (!capter.isOpened())
    {
        std::cout << "Could not open or find the vedio!\n";
        std::cout << "Usage: " << argv[0] << " <vedio or ID>" << std::endl;

        return EXIT_FAILURE;
    }
    double fps = capter.get(cv::CAP_PROP_FPS);

    // create two windows
    cv::namedWindow(window_capture_name);
    cv::namedWindow(window_detection_name);

    // Trackbars to set thresholds for HSV values
    cv::createTrackbar("Low H", window_detection_name,
                       &low_H,
                       max_value_H,
                       on_low_H_thresh_trackbar);
    cv::createTrackbar("High H", window_detection_name,
                       &high_H,
                       max_value_H,
                       on_high_H_thresh_trackbar);
    cv::createTrackbar("Low S", window_detection_name,
                       &low_S,
                       max_value,
                       on_low_S_thresh_trackbar);
    cv::createTrackbar("High S", window_detection_name,
                       &high_S,
                       max_value,
                       on_high_S_thresh_trackbar);
    cv::createTrackbar("Low V", window_detection_name,
                       &low_V,
                       max_value,
                       on_low_V_thresh_trackbar);
    cv::createTrackbar("High V", window_detection_name,
                       &high_V,
                       max_value,
                       on_high_V_thresh_trackbar);

    cv::Mat frame, frame_HSV, frame_threshold;
    while (true)
    {
        capter >> frame;
        if (frame.empty())
        {
            break;
        }

        cv::cvtColor(frame, frame_HSV, cv::COLOR_BGR2GRAY);
        cv::inRange(frame_HSV,
                    cv::Scalar(low_H, low_S, low_V),
                    cv::Scalar(high_H, high_S, high_V),
                    frame_threshold);

        cv::imshow(window_capture_name, frame);
        cv::imshow(window_detection_name, frame_threshold);

        char key = (char)cv::waitKey(fps);
        if (key == 'q' || key == 27)
        {
            break;
        }
    }

    return EXIT_SUCCESS;
}

static void on_low_H_thresh_trackbar(int, void *)
{
    low_H = std::min(high_H - 1, low_H);
    cv::setTrackbarPos("Low H", window_detection_name, low_H);
}

static void on_high_H_thresh_trackbar(int, void *)
{
    high_H = std::max(high_H, low_H + 1);
    cv::setTrackbarPos("High H", window_detection_name, high_H);
}

static void on_low_S_thresh_trackbar(int, void *)
{
    low_S = std::min(high_S - 1, low_S);
    cv::setTrackbarPos("Low S", window_detection_name, low_S);
}

static void on_high_S_thresh_trackbar(int, void *)
{
    high_S = std::max(high_S, low_S + 1);
    cv::setTrackbarPos("High S", window_detection_name, high_S);
}

static void on_low_V_thresh_trackbar(int, void *)
{
    low_V = std::min(high_V - 1, low_V);
    cv::setTrackbarPos("Low V", window_detection_name, low_V);
}

static void on_high_V_thresh_trackbar(int, void *)
{
    high_V = std::max(high_V, low_V + 1);
    cv::setTrackbarPos("High V", window_detection_name, high_V);
}