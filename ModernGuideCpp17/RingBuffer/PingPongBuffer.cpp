/**
 * @file PingPongBuffer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Double buffer
 * @version 0.1
 * @date 2025-01-10
 * 
 * @copyright Copyright (c) 2025
 * 
 * g++ PingPongBuffer.cpp -std=c++20
 * clang++ PingPongBuffer.cpp -std=c++20
 * 
 */

#include "opencv2/opencv.hpp"

#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>

/**
 * @brief Ping-Pong Double buffer
 * Ping-Pong 缓存通用用于实现双缓冲区(double buffer) 或缓存切换机制,
 * 尤其是需要高效数据交换和实时更新的场景, 如图像处理,视频流,GPU计算等应用中,
 * 使用两个缓存交替工作, 以减少延迟和提高性能.
 * 
 * 1. Ping-Pong 两个缓存是独立的内存缓存区;
 * 2. 程序在 Ping-buffer 中进行处理, 数据可以从 Pong-buffer 缓存读取, 反之亦然;
 * 3. Ping-Pong缓存交替切换可以减少处理过程中的阻塞或数据等待, 提高系统吞吐量;
 * 4. 图像处理: 一张图像正在被处理, 另一帧图像在另一个缓存中等待处理;
 * 5. GPU编程: CUDA中使用两个缓冲区, 一用于处理当前数据; 另一个用于准备下一数据;
 * 6. 视频编码/解码: 加码缓冲区和编码缓冲区可以交替使用, 实现高效流畅;
 * 
 */
class PingPongBuffer
{
public:
    PingPongBuffer(unsigned int width, unsigned int height)
        : mWidth{width}
        , mHeight{height}
        , mPingBuf{cv::Mat::zeros(height, width, CV_8UC3)}
        , mPongBuf{cv::Mat::zeros(height, width, CV_8UC3)}

    {
    }

    cv::Mat &getPingBuf()
    {
        return mPingBuf;
    }

    cv::Mat &getPongBuf()
    {
        return mPongBuf;
    }

    void swapBuffers()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        std::swap(mPingBuf, mPongBuf);
    }

private:
    unsigned int mWidth;
    unsigned int mHeight;
    cv::Mat      mPingBuf;
    cv::Mat      mPongBuf;
    std::mutex   mMutex;
};

// image processing thread-task
void imageProcessingTask(PingPongBuffer &buffer, std::atomic<bool> &running)
{
    while (running)
    {
        cv::Mat &ping = buffer.getPingBuf();

        cv::GaussianBlur(ping, ping, cv::Size(7, 7), 0);

        buffer.swapBuffers();

        // just for siml.
        std::this_thread::sleep_for(std::chrono::microseconds(30));
    }
}

// display-task for image
void displayImageTask(PingPongBuffer &buffer, std::atomic<bool> &running)
{
    while (running)
    {
        cv::Mat &pong = buffer.getPongBuf();

        cv::imshow("processed image", pong);
        if (cv::waitKey(1) == 22)
            running = false;

        // just for siml.
        std::this_thread::sleep_for(std::chrono::microseconds(30));
    }
}

// ---------------------------------------------
int main(int /* argc */, char ** /* argv */)
{
    PingPongBuffer buffer{640, 480};

    cv::Mat &ping = buffer.getPingBuf();

    cv::rectangle(ping, cv::Point(100, 100), cv::Point(500, 380), cv::Scalar(0, 0, 255), -1);
    std::atomic<bool> running{true};

    std::thread processTask(imageProcessingTask, std::ref(buffer), std::ref(running));
    std::thread displayTask(displayImageTask, std::ref(buffer), std::ref(running));

    if (processTask.joinable())
        processTask.join();
    if (displayTask.joinable())
        displayTask.join();

    cv::destroyAllWindows();

    return 0;
}
