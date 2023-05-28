/**
 * @file multithreading_example.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

// the global varibales
std::condition_variable notify_zero;
std::condition_variable notify_odd;
std::condition_variable notify_even;

bool print_zero_can_start = true;
bool print_odd_can_start  = false;
bool print_even_can_start = false;

std::mutex m;

void print_zero()
{
    int counter = 0;

    while (true)
    {
        // comment this to see whether it actually works
        std::this_thread::sleep_for(std::chrono::seconds(1));

        std::unique_lock<std::mutex> lock{m};

        notify_zero.wait(lock, []() { return print_zero_can_start; });

        ++counter;
        std::cout << "[zero thread]\t" << 0 << std::endl;
        if (counter % 2 == 1)
        {
            print_odd_can_start = true;
        }
        if (counter % 2 == 0)
        {
            print_even_can_start = true;
        }
        print_zero_can_start = false;
        lock.unlock();

        if (counter % 2 == 1)
        {
            notify_odd.notify_one();
        }
        if (counter % 2 == 0)
        {
            notify_even.notify_one();
        }
    }
}

void print_odd()
{
    int odd_counter = 1;
    while (true)
    {
        std::unique_lock<std::mutex> lock{m};

        notify_odd.wait(lock, []() { return print_odd_can_start; });

        std::cout << "[odd thread]\t" << odd_counter << std::endl;
        odd_counter += 2;
        print_odd_can_start  = false;
        print_zero_can_start = true;
        lock.unlock();

        notify_zero.notify_one();
    }
}

void print_even()
{
    int even_counter = 2;
    while (true)
    {
        std::unique_lock<std::mutex> lock{m};

        notify_even.wait(lock, []() { return print_even_can_start; });

        std::cout << "[even thread]\t" << even_counter << std::endl;
        even_counter += 2;
        print_even_can_start = false;
        print_zero_can_start = true;
        lock.unlock();

        notify_zero.notify_one();
    }
}

/**
 * @brief 创建三个线程, 其中一个线程能持续输出"0", 
 *    一个线程能持续输出奇数, 一个线程能持续输出偶数, 
 *    要求保证输出的结果是按这样的顺序: "0 1 0 2 0 3 0 4 0 5..."
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "--------- [main thread start] ---------" << std::endl;

    std::thread thread0(print_zero);
    std::thread thread1(print_odd);
    std::thread thread2(print_even);

    thread0.join();
    thread1.join();
    thread2.join();

    // when the statements can output to the console?
    std::cout << "--------- [main thread end] ---------" << std::endl;

    return 0;
}