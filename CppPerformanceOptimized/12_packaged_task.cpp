/**
 * @file 12_packaged_task.cpp
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

// packaged_task 和线程
void task_future_example()
{
    auto meaning = std::packaged_task<int(int)>([](int n) { return n; });
    auto result  = meaning.get_future();

    auto t = std::thread(std::move(meaning), 42);
    std::cout << "the meaning of life: " << result.get() << "\n";
    t.join();
}

int main(int argc, const char *argv[])
{
    task_future_example();

    return 0;
}
