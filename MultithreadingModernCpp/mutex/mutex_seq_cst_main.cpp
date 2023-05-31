/**
 * @file mutex_seq_cst_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-31
 * 
 * @copyright Copyright (c) 2023
 * 
 * $ clang++ mutex_seq_cst_main.cpp -std=c++17
 * $ g++ mutex_seq_cst_main.cpp -std=c++17
 * $ cl mutex_seq_cst_main.cpp /std:c++17 /EHsc
 * 
 * ! 多次执行该二进制程序, 观察并思考为什么结果有差异
 * 
 */

#include <iostream>
#include <mutex>
#include <thread>

std::mutex mutex;
int        x = 0;
int        y = 0;

/**
 * @brief 
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::thread t1(
        []()
        {
            x = 1;
            std::lock_guard<std::mutex> lg(mutex);
            y = 0;
        });

    std::thread t2(
        []()
        {
            std::lock_guard<std::mutex> lg(mutex);
            y = x + 2;
        });

    if (t1.joinable())
        t1.join();
    if (t2.joinable())
        t2.join();

    std::cout << "the x value: " << x << std::endl;
    std::cout << "the y value: " << y << std::endl;

    return 0;
}