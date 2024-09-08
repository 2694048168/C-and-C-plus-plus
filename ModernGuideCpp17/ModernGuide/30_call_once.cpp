/**
 * @file 30_call_once.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 在某些特定情况下,某些函数只能在多线程环境下调用一次,
 * 比如要初始化某个对象,而这个对象只能被初始化一次,
 * 就可以使用std::call_once()来保证函数在多线程环境下只能被调用一次.
 * *使用call_once()的时候,需要一个once_flag作为call_once()的传入参数,该函数的原型如下:
// 定义于头文件 <mutex>
template< class Callable, class... Args >
void call_once( std::once_flag& flag, Callable&& f, Args&&... args );
 * 1. flag: once_flag类型的对象, 要保证这个对象能够被多个线程同时访问到;
 * 2. f: 回调函数,可以传递一个有名函数地址, 也可以指定一个匿名函数;
 * 3. args：作为实参传递给回调函数;
 * ?多线程操作过程中, std::call_once()内部的回调函数只会被执行一次;
 * 
 */

#include <cstddef>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

std::once_flag g_flag;

void do_once(int a, std::string b)
{
    std::cout << "name: " << b << ", age: " << a << std::endl;
}

void do_something(int age, std::string name)
{
    static int num = 1;
    std::call_once(g_flag, do_once, 19, "Ithaca");
    std::cout << "do_something() function num = " << num++ << std::endl;
}

// -------------------------------------
int main(int argc, const char **argv)
{
    const int num_core = std::thread::hardware_concurrency();

    std::vector<std::thread> vec_task;
    for (size_t idx{0}; idx < num_core; ++idx)
    {
        vec_task.emplace_back(std::thread(do_something, 20, "ace"));
    }
    // call_once()中指定的回调函数只被执行了一次

    // for (const auto &task_thread : vec_task)
    for (auto &task_thread : vec_task)
    {
        task_thread.join();
    }

    return 0;
}
