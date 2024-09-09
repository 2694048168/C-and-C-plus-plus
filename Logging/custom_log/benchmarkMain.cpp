/**
 * @file benchmarkMain.cpp
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
    CustomLogger          logger(log_output_dir);
    CustomLoggerBenchmark benchmark(logger);

    // 单条日志写入测试
    benchmark.SingleLogTest();
    // 批量日志写入测试
    benchmark.BulkLogTest(100000); // 写入 100000 条日志

    // 等待日志线程完成写入
    std::this_thread::sleep_for(std::chrono::seconds(3));
    std::cout << "Log message write into log-file successfully\n";
    return 0;
}
