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
        std::cout << m_name << " time cost ---> " << duration.count() << " ms\n";
    }

private:
    std::string_view m_name;

    std::chrono::steady_clock::time_point m_startTime;
};
