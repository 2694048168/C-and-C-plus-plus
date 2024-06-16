/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-06-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <chrono>
#include <iostream>
#include <thread>

// --------------------------------------
int main(int argc, const char **argv)
{
    std::cout << "CMake test Timeout example\n";

    using namespace std::literals::chrono_literals;
    std::this_thread::sleep_for(0.5s);

    return 0;
}
