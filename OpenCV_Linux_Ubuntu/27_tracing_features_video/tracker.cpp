/**
 * @File    : tracker.cpp
 * @Brief   : 跟踪视频中的特征点
 * @Author  : Wei Li
 * @Date    : 2021-08-11
*/

#include <iostream>

#include "featuretracker.hpp"

// ------------------------------
int main(int argc, char **argv)
{
    // Create video procesor instance
    VideoProcessor processor;

    // Create feature tracker instance
    FeatureTracker tracker;

    // Open video file
    processor.setInput("./../images/bike.avi");

    // set frame processor
    processor.setFrameProcessor(&tracker);

    // Declare a window to display the video
    processor.displayOutput("Tracked Features");

    // Play the video at the original frame rate
    processor.setDelay(1000. / processor.getFrameRate());

    processor.stopAtFrameNo(90);

    // Start the process
    processor.run();

    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}
