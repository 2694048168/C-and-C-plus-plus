/**
 * @file test_basic.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-27
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "ProjectLogging/Logger.hpp"

#include <chrono>
#include <thread>

// -----------------------------------
int main(int argc, const char **argv)
{
    INITLOG();

    //单个日志
    auto log1 = GETLOG("Test1");
    auto log2 = GETLOG("Test1");

    //原始调用方式
    //SPDLOG_LOGGER_INFO(log1, "123");
    LOGGER_INFO(log2, "123");

    auto sw = LOGSW();
    // 延时2秒
    std::this_thread::sleep_for(std::chrono::seconds(2));
    INFO("Elapsed {0} {1}", "时间", sw);
    WARN("Elapsed {0} {1}", "时间", sw);

    //原始调用方式
    //SPDLOG_INFO("TEST");
    INFO("TEST");
}
