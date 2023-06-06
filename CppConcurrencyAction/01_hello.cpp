/**
 * @file 01_hello.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief
 * @version 0.1
 * @date 2023-06-06
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <iostream>
// 包括标准库中对多线程支持的声明，管理线程的函数和类
#include <thread>

// 每个线程都必须一个执行单元，新线程的执行从这里开始
void hello(const char *msg)
{
    std::cout << msg;
}

int main(int argc, const char **argv)
{
    // 对于应用程序来说, 初始线程是 main()
    std::cout << "Hello World from main thread\n";

    const char *msg = "Hello Concurrent World from a thread\n";

    // 在 std::thread 对象的构造函数中指定新函数 hello() 作为其执行函数
    std::thread t{hello, msg};

    // 让初始线程等待 std::thread 对象创建的线程
    t.join();

    return 0;
}