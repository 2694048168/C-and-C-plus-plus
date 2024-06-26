/**
 * @file test_logger.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-12
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "logger/logger.hpp"

// -----------------------------------
int main(int argc, const char **argv)
{
    initLogger(true);
    LOG_DEBUG("this is debug");
    LOG_INFO("this is info");
    LOG_TRACE("this is trace");
    LOG_WARN("this is warn");
    LOG_ERROR("this is error");
    LOG_FATAL("this is fatal");

    LOG_DEBUG("this is:" << 666);
    LOG_DEBUG_F(LOG4CPLUS_TEXT("this is %.2f"), 5.333);
    shutDown();

    close();

    return 0;
}
