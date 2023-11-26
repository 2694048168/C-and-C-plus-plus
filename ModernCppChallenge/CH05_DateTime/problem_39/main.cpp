/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-11-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <chrono>
#include <format>
#include <iostream>
#include <ratio>
#include <thread>

/**
 * @brief Measuring function execution time
 * Write a function that can measure the execution time of a function
 * (with any number of arguments) in any required duration
 * (such as seconds, milliseconds, microseconds, and so on).
 * 
 */

/**
 * @brief Solution:

To measure the execution time of a function, you should retrieve the current time 
before the function execution, execute the function, then retrieve the current time again
and determine how much time passed between the two time points.

 For convenience, this can all be put in a variadic function template
that takes as arguments the function to execute and its arguments, and:
1. Uses std::high_resolution_clock by default to determine the current time.
2. Uses std::invoke() to execute the function to measure, with its specified arguments.
3. Returns a duration and not a number of ticks for a particular duration.
 This is important so that you don't lose resolution. It enables you to add execution time
 duration of various resolutions, such as seconds and milliseconds,
 which would not be possible by returning a tick count:
------------------------------------------------------ */
template<typename Time = std::chrono::microseconds, typename Clock = std::chrono::high_resolution_clock>
struct perf_timer
{
    template<typename F, typename... Args>
    static Time duration(F &&f, Args... args)
    {
        auto start = Clock::now();

        std::invoke(std::forward<F>(f), std::forward<Args>(args)...);

        auto end = Clock::now();

        return std::chrono::duration_cast<Time>(end - start);
    }
};

using namespace std::chrono_literals;

void func()
{
    // simulate work
    std::this_thread::sleep_for(2s);
}

void goWork(const int a, const int b)
{
    auto result = a + b;

    // C++20 format library
    std::string printInfo = std::format("[====]The sum of {} and {} is {}\n", a, b, result);
    std::cout << printInfo;
    // simulate work
    std::this_thread::sleep_for(1s);
}

// ------------------------------
int main(int argc, char **argv)
{
    auto t1 = perf_timer<std::chrono::microseconds>::duration(func);
    auto t2 = perf_timer<std::chrono::milliseconds>::duration(goWork, 24, 42);

    auto total    = std::chrono::duration<double, std::nano>(t1 + t2).count();
    auto total_ms = std::chrono::duration<double, std::milli>(t1 + t2).count();
    auto total_us  = std::chrono::duration<double, std::micro>(t1 + t2).count();

    std::string timeConsuming    = std::format("[====]Time consuming is {} ns\n", total);
    std::string timeConsuming_ms = std::format("[====]Time consuming is {} ms\n", total_ms);
    std::string timeConsuming_us  = std::format("[====]Time consuming is {} us\n", total_us);
    std::cout << timeConsuming << timeConsuming_ms << timeConsuming_us;
}