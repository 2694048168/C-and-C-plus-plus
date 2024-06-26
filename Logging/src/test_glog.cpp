/**
 * @file test_glog.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Examples of Using glog
 * @version 0.1
 * @date 2024-04-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "glog/logging.h"

#include <direct.h>
#include <io.h>

#include <string>

bool createDirectory(std::string &path)
{
    int len = path.length();

    char tmpDirPath[256] = {0};
    for (int idx = 0; idx < len; ++idx)
    {
        tmpDirPath[idx] = path[idx];
        if (tmpDirPath[idx] == '\\' || tmpDirPath[idx] == '/')
        {
            if (_access(tmpDirPath, 0) == -1)
            {
                int ret = _mkdir(tmpDirPath);
                if (ret == -1)
                    return false;
            }
        }
    }

    return true;
}

// -----------------------------------
int main(int argc, const char **argv)
{
    // 初始化glog库
    google::InitGoogleLogging(argv[0]);

    std::string log_filepath = "./log/log_info";
    createDirectory(log_filepath);
    google::SetLogDestination(google::INFO, log_filepath.c_str());

    // 设置日志级别为 INFO
    // FLAGS_minloglevel = google::INFO;
    // 设置日志存放路径
    FLAGS_log_dir = log_filepath;

    // 记录一条INFO级别的日志
    LOG(INFO) << u8"这是一条 INFO 级别的日志";

    // 记录一条ERROR级别的日志
    LOG(ERROR) << u8"这是一条 ERROR 级别的日志";

    // 关闭glog库
    google::ShutdownGoogleLogging();

    return 0;
}
