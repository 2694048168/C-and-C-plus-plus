/**
 * @file 06_chrono.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-17
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <chrono>
#include <iostream>

// Modern C++11 中提供了日期和时间相关的库chrono
// 通过chrono库可以很方便地处理日期和时间,
// chrono库主要包含三种类型的类:
//  时间间隔 duration; 时钟 clocks; 时间点 time point
// -----------------------------
int main(int argc, char **argv)
{
    /* 时间间隔 duration
    标准库中定义常用的时间间隔: 时、分、秒、毫秒、微秒、纳秒，都位于std::chrono命名空间
    std::chrono::microseconds;
    std::chrono::milliseconds;
    std::chrono::seconds;

    时钟周期 * 周期次数 = 总的时间间隔
    -------------------------------- */
    std::chrono::hours                           h(1);  /* 一小时 */
    std::chrono::milliseconds                    ms{3}; /* 3 毫秒 */
    std::chrono::duration<int, std::ratio<1000>> ks(3); /* 3000 秒 */

    std::cout << h.count() << "hours " << ms.count() << "ms " << ks.count() << "s\n";

    // chrono::duration<int, ratio<1000>> d3(3.5);  /* error */
    std::chrono::duration<double> dd(6.6); /* 6.6 秒 */
    std::cout << dd.count() << "s \n";

    // 使用小数表示时钟周期的次数
    std::chrono::duration<double, std::ratio<1, 30>> hz(3.5);
    std::cout << hz.count() << "\n";

    std::chrono::milliseconds ms_{3};      // 3 毫秒
    std::chrono::microseconds us = 2 * ms; // 6000 微秒

    // 时间间隔周期为 1/30 秒
    std::chrono::duration<double, std::ratio<1, 30>> hz_(3.5);

    std::cout << "3 ms duration has " << ms_.count() << " ticks\n"
              << "6000 us duration has " << us.count() << " ticks\n"
              << "3.5 hz duration has " << hz_.count() << " ticks\n";

    // 计算时间间隔
    std::chrono::minutes t1(10);
    std::chrono::seconds t2(60);
    std::chrono::seconds t3 = t1 - t2;
    std::cout << t3.count() << " second\n";

    /* 时间戳 time point
    ----------------------- */
    /* 时钟 clocks
    1. system_clock:系统的时钟, 系统的时钟可以修改, 甚至可以网络对时,因此使用系统时间计算时间差可能不准;
    2. steady_clock:是固定的时钟,相当于秒表,开始计时后,时间只会增长并且不能修改,适合用于记录程序耗时;
    3. high_resolution_clock: 和时钟类 steady_clock 是等价的,提供的时钟精度更高
    ------------------------------------------------------ */
    std::cout << "-----------------------------\n";
    // 获取开始时间点
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    // 执行业务流程
    std::cout << "print 1000 stars ....\n";
    for (int i = 0; i < 1000; ++i)
    {
        // std::cout << "*";
        continue;
    }
    std::cout << std::endl;

    // 获取结束时间点
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();

    // 计算差值
    auto dt = last - start;
    std::cout << "the total consumption time: " << dt.count() << " ns\n";

    // using high_resolution_clock = steady_clock;
    std::cout << "-----------------------------\n";
    std::chrono::high_resolution_clock ::time_point start_high = std::chrono::high_resolution_clock::now();
    // 执行业务流程
    std::cout << "print 1000 stars ....\n";
    for (int i = 0; i < 1000; ++i)
    {
        // std::cout << "*";
        continue;
    }
    std::cout << std::endl;

    // 获取结束时间点
    std::chrono::high_resolution_clock ::time_point last_high = std::chrono::high_resolution_clock::now();

    // 计算差值
    auto dt_high = last_high - start_high;
    std::cout << "the total consumption time: " << dt_high.count() << " ns\n";

    /* 转换函数
    duration_cast
    time_point_cast
    ---------------------------- */

    return 0;
}
