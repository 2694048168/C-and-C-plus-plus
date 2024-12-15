/**
 * @file 13_async.cpp
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

void async_future_example()
{
    auto meaning = [](int n)
    {
        return n;
    };

    auto result = std::async(std::move(meaning), 42);
    std::cout << "the meaning of life: " << result.get() << "\n";
}

int main(int argc, const char *argv[])
{
    async_future_example();

    return 0;
}
