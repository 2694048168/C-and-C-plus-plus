/**
 * @file 00_programSupport.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdlib>
#include <iostream>

/**
 * @brief 探讨有关构建实际应用程序的基础知识.
 * ?1. 首先讨论内置于 C++ 中的程序支持功能, 允许与应用程序生命周期进行交互;
 * ?2. 介绍 Boost ProgramOptions, 这是一个优秀的用于开发控制台应用程序的库, 
 *    它包含接受用户输入的设施, 无须"重新造轮子";
 * ?3. 阐述一些关于预处理器和编译器的特殊话题, 在构建源代码超过单个文件的应用程序时可能会遇到这些问题;
 */

struct Tracer
{
    Tracer(std::string name_in)
        : name{std::move(name_in)}
    {
        std::cout << name << " constructed.\n";
    }

    ~Tracer()
    {
        std::cout << name << " destructed.\n";
    }

private:
    const std::string name;
};

Tracer static_tracer{"static Tracer"};

void run()
{
    std::cout << "Entering run()\n";
    // ...
    std::cout << "Exiting run()\n";
}

void run_atexit()
{
    std::cout << "Registering a callback\n";
    std::atexit([] { std::cout << "***std::atexit callback executing***\n"; });
    std::cout << "Callback registered\n";
}

void run_exit()
{
    std::cout << "Registering a callback\n";
    std::atexit([] { std::cout << "***std::atexit callback executing***\n"; });
    std::cout << "Callback registered\n";
    std::exit(0);
}

void run_abort()
{
    std::cout << "Registering a callback\n";
    std::atexit([] { std::cout << "***std::atexit callback executing***\n"; });
    std::cout << "Callback registered\n";
    std::abort();
}

// ----------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief 程序支持功能
     * 程序需要与操作环境的应用生命周期进行交互:
     * ?1. 处理程序的终止和清理工作;
     * ?2. 与环境交互;
     * ?3. 管理操作系统的信号;
     */
    std::cout << "=========================================\n";
    std::cout << "Entering main()\n";
    Tracer              local_tracer{"local Tracer"};
    thread_local Tracer thread_local_tracer{"thread_local Tracer"};
    const auto         *dynamic_tracer = new Tracer{"dynamic Tracer"};
    run();
    delete dynamic_tracer;
    std::cout << "Exiting main()\n";

    /**
     * @brief 处理程序的终止和清理工作
     * ＜cstdlib＞ 头文件包含几个管理程序终止和资源清理的函数.
     * 程序终止函数有两大类:
     * ?1. 导致程序终止的函数;
     * ?2. 在终止即将发生时注册一个回调的函数
     * 
     * *====终止回调与 std::atexit
     */
    std::cout << "\n============= std::atexit ===================\n";
    Tracer              local_tracer_atexit{"local Tracer"};
    thread_local Tracer thread_local_tracer_atexit{"thread_local Tracer"};
    const auto         *dynamic_tracer_atexit = new Tracer{"dynamic Tracer"};
    run_atexit();
    delete dynamic_tracer_atexit;

    /**
     * @brief 用 std::exit 退出
     * 在某些情况下, 例如在多线程程序中, 可能希望以其他方式优雅地退出程序.
     * 它接受一个对应于程序退出代码的单一整数,它将执行以下清理步骤:
     * 1）与当前线程相关的线程局部对象和静态对象被销毁, atexit 回调函数被调用;
     * 2）所有的 stdin、stdout 和 stderr 都被刷新;
     * 3）临时文件都会被删除;
     * 4）程序向操作环境报告给定的状态代码, 操作环境恢复控制;
     */
    std::cout << "\n============= std::exit ===================\n";
    Tracer              local_tracer_exit{"local Tracer"};
    thread_local Tracer thread_local_tracer_exit{"thread_local Tracer"};
    const auto         *dynamic_tracer_exit = new Tracer{"dynamic Tracer"};
    run_exit();
    delete dynamic_tracer_exit;

    /**
     * @brief 要结束程序, 还可以使用 std::abort,
     * !这个函数接受一个整数值的状态代码, 并立即将其返回给操作环境.
     * 没有对象的析构函数被调用, 也没有 std::atexit 回调被调用.
     */
    std::cout << "\n============= std::abort ===================\n";
    Tracer              local_tracer_abort{"local Tracer"};
    thread_local Tracer thread_local_tracer_abort{"thread_local Tracer"};
    const auto         *dynamic_tracer_abort = new Tracer{"dynamic Tracer"};
    run_abort();
    delete dynamic_tracer_abort;

    return 0;
}
