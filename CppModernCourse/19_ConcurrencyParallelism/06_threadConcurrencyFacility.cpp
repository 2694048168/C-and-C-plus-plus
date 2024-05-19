/**
 * @file 06_threadConcurrencyFacility.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <thread>

/**
 * @brief C++ stdlib 的＜thread＞库包含用于并发编程的底层设施,
 * 例如std::thread 类模拟了一个操作系统线程,
 * *然而最好不要直接使用 thread, 应该使用更高层次的抽象(比如任务)来设计程序中的并发性.
 * ＜thread＞ 库确实包括几个有用的函数, 用于操纵当前线程:
 * ?1. std::this_thread::yield();
 * ?2. std::this_thread::get_id();
 * ?3. std::this_thread::sleep_for();
 * ?4. std::this_thread::sleep_until();
 * TODO: 当需要这些函数时, 它们是必不可少的;否则不需要与＜thread＞头文件交互.
 */

// ------------------------------------
int main(int argc, const char **argv)
{
    const auto core_num = std::thread::hardware_concurrency();
    std::cout << "The core-number of HOST: " << core_num << '\n';

    const auto thread_id = std::this_thread::get_id();
    std::cout << "The thread-ID of main: " << thread_id << '\n';

    const auto        now = std::chrono::system_clock::now();
    const std::time_t t_c = std::chrono::system_clock::to_time_t(now);
    std::cout << "The system clock is currently at " << std::ctime(&t_c);

    using namespace std::chrono_literals;
    std::this_thread::sleep_for(3ms);
    std::this_thread::sleep_for(3s);

    const auto        now2 = std::chrono::system_clock::now();
    const std::time_t t_c2 = std::chrono::system_clock::to_time_t(now2);
    std::cout << "The system clock is currently at " << std::ctime(&t_c2);

    return 0;
}
