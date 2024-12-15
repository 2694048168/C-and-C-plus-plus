/**
 * @file 10_thread.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-13
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <thread>

void func(int n)
{
    std::cout << "thread " << n << std::endl;
}

int main(int argc, const char *argv[])
{
    std::thread t1;          // 线程变量，不是一个线程
    t1 = std::thread(func, 1); // 将一个线程赋值给线程变量

    t1.join();               // 等待线程结束

    std::thread t2(func, 2);
    std::thread t3(std::move(t2));
    std::thread t4([]() { return; }); // 也可以与lambda表达式配合使用
    
    t4.detach();
    t3.join();

    return 0;
}
