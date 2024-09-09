/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-09
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "logging.hpp"

#include <filesystem>
#include <iostream>

// =======================================
int main(int argc, const char **argv)
{
    // 指定日志文件的输出路径
    std::string log_output_dir = "./logs";
    if (!std::filesystem::exists(log_output_dir))
    {
        std::filesystem::create_directory(log_output_dir);
    }

    // 创建Logger实例
    CustomLogger logger(log_output_dir);

    // 记录不同级别的日志
    logger.Log("This is an INFO message", LogLevel::INFO);
    logger.Log("This is a WARNING message", LogLevel::WARNING);
    logger.Log("This is an ERROR message", LogLevel::ERROR);
    logger.Log("This is an EXCEPTION message", LogLevel::EXCEPTION);

    // 等待一会，保证日志异步写入
    std::this_thread::sleep_for(std::chrono::seconds(5));

    std::cout << "Log message write into log-file successfully\n";
    return 0;
}
