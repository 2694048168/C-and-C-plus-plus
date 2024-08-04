/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-08-04
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "ThreadPoolAsynchronous.hpp"

#include <iostream>

int calc(int x, int y)
{
    int res = x + y;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return res;
}

int main(int argc, const char **argv)
{
    ThreadPoolAsynchronous        pool(4);
    std::vector<std::future<int>> results;

    for (int i = 0; i < 10; ++i)
    {
        results.emplace_back(pool.addTask(calc, i, i * 2));
    }

    // 等待并打印结果
    for (auto &&res : results)
    {
        std::cout << "线程函数返回值: " << res.get() << std::endl;
    }

    return 0;
}
