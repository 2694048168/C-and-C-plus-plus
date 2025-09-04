/**
 * @file Stopwatch.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief RAII 时间耗时统计度量
 * @version 0.1
 * @date 2025-09-04
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include <chrono>
#include <iostream>
#include <string_view>

class Stopwatch
{
public:
    Stopwatch() = delete;

    explicit Stopwatch(const std::string_view name)
        : m_name{name}
    {
        m_startTime = std::chrono::high_resolution_clock::now();
    }

    ~Stopwatch()
    {
        auto endTime  = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_startTime);
        // auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - m_startTime);
        std::cout << m_name << " time cost ---> " << duration.count() << " ms\n";
        // std::cout << m_name << " time cost ---> " << duration_ns.count() << " ns\n";
    }

private:
    std::string_view m_name;

    std::chrono::steady_clock::time_point m_startTime;
};
