/**
 * @file utility.h
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <Windows.h>
#include <opencv2/opencv.hpp>

#include <string>

namespace __DrawText__ {
void GetStringSize(HDC hDC, const char *str, int *width, int *height);
void PutTextExt(cv::Mat &dst, const char *str, cv::Point org, cv::Scalar color, int fontSize, const char *fn = "Arial",
                bool italic = false, bool underline = false);
}; // namespace __DrawText__
