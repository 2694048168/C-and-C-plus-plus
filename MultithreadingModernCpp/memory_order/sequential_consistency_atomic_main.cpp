/**
 * @file sequential_consistency_atomic_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-31
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <atomic>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// Global variables
std::atomic<int> flag(0);

std::atomic<std::string> msg; // ! error

int         b = 0, x = 0, y = 0;
std::string text;

void f1()
{
    x = 3;
    y += x;
    msg  = "Hello world";
    flag = 1;
    b += flag;
}

void f2()
{
    text = "Hi";
    while (flag != 1)
        ;
    text = msg;
}

/**
 * @brief 
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::vector<std::thread> threads;

    threads.push_back(std::thread(f1));
    threads.push_back(std::thread(f2));

    for (std::thread &t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    std::cout << "msg: " << msg << std::endl;
    std::cout << "b: " << b << ", y: " << y << std::endl;
    std::cout << "text: " << text << std::endl;

    return 0;
}
