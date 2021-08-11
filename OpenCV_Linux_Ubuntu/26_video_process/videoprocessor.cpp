/**
 * @File    : videoprocessor.cpp
 * @Brief   : 读取视频序列 处理视频帧 写入视频帧
 * @Author  : Wei Li
 * @Date    : 2021-08-11
*/

#include <iostream>
#include <string>
#include <sstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "videoprocessor.hpp"

void draw(const cv::Mat &img, cv::Mat &out)
{
    img.copyTo(out);
    cv::circle(out, cv::Point(100, 100), 5, cv::Scalar(255, 0, 0), 2);
}

// processing function
void canny(cv::Mat &img, cv::Mat &out)
{
    if (img.channels() == 3)
    {
        cv::cvtColor(img, out, cv::COLOR_BGR2GRAY);
    }
    // compute Canny deges
    cv::Canny(out, out, 100, 200);
    // invert the image
    cv::threshold(out, out, 128, 255, cv::THRESH_BINARY_INV);
}

// ------------------------------
int main(int argc, char **argv)
{
    // open the video file
    cv::VideoCapture capture("./../images/bike.avi");
    // cv::VideoCapture capture("http://www.laganiere.name/bike.avi");
    // check if video  successfully opened
    if (!capture.isOpened())
    {
        std::cerr << "--Error opening video unsuccessfully." << std::endl;
        return 1;
    }

    // get the frame rate
    double rate = capture.get(cv::CAP_PROP_FPS);
    std::cout << "Frame rate: " << rate << "fps" << std::endl;

    bool stop = false;
    cv::Mat frame; // current video frame
    cv::namedWindow("Extracted Frame");

    // delay betweent each frame corresponds to video frame rate
    // int delay = 1000 / rate;
    int delay = 500 / rate;
    long long i = 0;
    std::string b = "bike";
    std::string ext = ".bmp";
    // for all frames in video
    while (!stop)
    {
        // read next frame if any
        if (!capture.read(frame))
        {
            break;
        }

        cv::imshow("Extracted Frame", frame);

        std::string name(b);
        std::ostringstream ss;
        ss << std::setfill('0') << std::setw(3) << i;
        name += ss.str();
        i++;
        name += ext;

        std::cout << name << std::endl;

        cv::Mat test;
        // cv::resize(frame, test, cv::Size(), 0.2, 0.2);
        // cv::imwrite(name, frame);
        // cv::imwrite(name, test);

        // introduce a delay
        // or press key to stop
        if (cv::waitKey(delay) >= 0)
        {
            stop = true;
        }
    }

    // Close the video file
    capture.release();
    cv::waitKey();

    // ----------------------------------------
    // Now using the VideoProcessor class
    // Create instance
    VideoProcessor processor;

    // Open video file
    processor.setInput("./../images/bike.avi");

    // Declare a window to display the video
    processor.displayInput("Input Video");
    processor.displayOutput("Output Video");

    // Play the video at the original frame rate
    processor.setDelay(500. / processor.getFrameRate());

    // Set the frame processor callback function
    processor.setFrameProcessor(canny);

    // output a video
    processor.setOutput("bikeCanny000.avi", -1, 15);

    // stop the process at this frame
    processor.stopAtFrameNo(100);
    // processor.stopAtFrameNo(51);

    // Start the process
    processor.run();

    cv::waitKey();

    cv::destroyAllWindows();
    return 0;
}
