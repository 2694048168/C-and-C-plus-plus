/**
 * @file modern_ThreadPool_test.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-29
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "modern_ThreadPool.hpp"

#include <chrono>
#include <ctime>
#include <iostream>
#include <vector>

// ------------------------------
int main(int argc, char **argv)
{
    clock_t start = clock(); // 记录开始时间

    int        thread_number = 3;          // 线程池中线程的数量
    int        task_number   = 12;         // 任务数量
    ThreadPool pool(thread_number);        // 创建一个线程池，参数是线程池中线程的数量【类】
    std::vector<std::future<int>> results; // 用于保存线程执行结果的容器【::作用域限定符】【泛型】
    results.reserve(task_number);          // 预留空间，避免内存重新分配
    for (int i = 0; i < task_number; ++i)
    { // 循环创建若干个任务
        // lambda 表达式，代表一个要执行的任务，参数和返回值都是int类型【lambda】【auto】
        auto taskLambda = [i]
        {
            std::cout << "hello " << i << std::endl;              // 打印 hello【打印】
            std::this_thread::sleep_for(std::chrono::seconds(1)); // 睡眠1秒
            std::cout << "world " << i << std::endl;              // 打印 world
            return i * i;                                         // 返回 i 的平方
        };
        auto future = pool.enqueue(taskLambda); // 将任务放入线程池，返回一个 std::future<int> 对象
        // 将 std::future<int> 对象放入容器，这里使用了移动语义，表示对 future 进行引用，而不是复制【move】
        results.emplace_back(std::move(future));
    }
    for (auto &&result : results)
    { // 循环打印线程执行结果，&& 表示对 result 进行引用，而不是复制【&&】
        std::cout << result.get() << ' ';
    }

    // 打印总耗时（秒数）
    std::cout << "\n[=====]Time consumption: " << (double)(clock() - start) / CLOCKS_PER_SEC << "s\n";

    return 0;
}
