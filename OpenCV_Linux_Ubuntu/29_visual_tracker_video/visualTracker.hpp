/**
 * @File    : visualTracker.hpp
 * @Brief   : 跟踪视频中的物体
 * @Author  : Wei Li
 * @Date    : 2021-08-11
*/

#ifndef VISUAL_TRACKER_H
#define VISUAL_TRACKER_H

#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>

#include "videoprocessor.hpp"

class VisualTracker : public FrameProcessor
{
    cv::Ptr<cv::Tracker> tracker;
    cv::Rect2d box;
    bool reset;

public:
    // constructor specifying the tracker to be used
    VisualTracker(cv::Ptr<cv::Tracker> tracker) : reset(true), tracker(tracker) {}

    // set the bounding box to initiate tracking
    void setBoundingBox(const cv::Rect2d &bb)
    {
        box = bb;
        reset = true;
    }

    // callback processing method
    void process(cv::Mat &frame, cv::Mat &output)
    {
        if (reset)
        { // new tracking session
            reset = false;

            tracker->init(frame, box);
        }
        else
        { // update the target's position

            tracker->update(frame, box);
        }

        // draw bounding box on current frame
        frame.copyTo(output);
        cv::rectangle(output, box, cv::Scalar(255, 255, 255), 2);
    }
};

#endif // VISUAL_TRACKER_H