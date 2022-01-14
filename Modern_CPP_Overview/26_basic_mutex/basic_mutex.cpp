/**
 * @file basic_mutex.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief Mutex in C++ class and <mutex> header file
 * @version 0.1
 * @date 2022-01-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <thread>
#include <mutex>

int value = 1;
void critical_mutex(int change_value)
{
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);

    /* execute contention works 内容的读写一致性 */
    value = change_value;

    /* mtx will be released after leaving the scope. */
}

void critical_section(int change_value)
{
    static std::mutex mtx;
    std::unique_lock<std::mutex> lock(mtx);

    /* do contention operations */
    value = change_value;
    std::cout << value << std::endl;

    /* release the lock */
    lock.unlock();

    /* during this period, others are allowed to acquire value to write */
    /* start another group of contention operators and lock again */
    lock.lock();
    value += 1;
    std::cout << value << std::endl;
}

int main(int argc, char** argv)
{
    std::thread thread_1(critical_mutex, 42);
    std::thread thread_2(critical_mutex, 24);
    thread_1.join();
    thread_2.join();

    std::cout << value << std::endl;

    std::thread thread_3(critical_mutex, 33);
    std::thread thread_4(critical_mutex, 66);
    thread_3.join();
    thread_4.join();

    std::cout << value << std::endl;

    return 0;
}
