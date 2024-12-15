/**
 * @file 11_promise_future.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <future>
#include <iostream>
#include <thread>

// 如何使用 promise 和 future 来控制线程交互
void promise_future_example()
{
    auto meaning = [](std::promise<int> &prom)
    {
        prom.set_value(42); // 计算"meaning of life"
    };

    std::promise<int> prom;
    std::thread(meaning, std::ref(prom)).detach();

    std::future<int> result = prom.get_future();
    // 程序会在 result.get() 中挂起，等待线程设置 prom 的共享状态
    std::cout << "the meaning of life: " << result.get() << "\n";
}

int main(int argc, const char *argv[])
{
    promise_future_example();

    return 0;
}
