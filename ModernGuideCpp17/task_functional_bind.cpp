/**
 * @file task_functional_bind.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 利用 functional and bind 方式进行任务队列实现
 * @version 0.1
 * @date 2024-03-10
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <functional>
#include <iostream>
#include <string_view>
#include <vector>

std::vector<std::function<void()>> g_taskFunction;

void printInfo()
{
    std::cout << "Hello world\n";
}

void printMessage(const std::string_view &message)
{
    std::cout << message << std::endl;
}

class Car
{
public:
    Car()  = default;
    ~Car() = default;

    inline void run()
    {
        std::cout << "the car running\n";
    }
};

// ====================================
int main(int argc, const char **argv)
{
    int  value       = 42;
    auto lambda_func = [&](int num)
    {
        std::cout << "The add value is: " << value + num << std::endl;
    };

    // 通过 std::functional and std::bind 将所有任务(任务执行对应函数地址)
    // 统一为 std::function<void()> 形式压入任务队列中
    // 然后多线程可以从任务队列中取出任务进行执行
    g_taskFunction.push_back(std::bind(printInfo));
    g_taskFunction.push_back(std::bind(printMessage, "C++ programming world"));
    g_taskFunction.push_back(std::bind(&Car::run, new Car));
    g_taskFunction.push_back(std::bind(lambda_func, 12));

    for (const auto &func : g_taskFunction)
    {
        func();
    }

    return 0;
}
