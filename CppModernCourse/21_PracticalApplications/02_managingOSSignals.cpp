/**
 * @file 02_managingOSSignals.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>

/**
 * @brief 管理操作系统的信号 Managing Operating System Signals
 * 操作系统信号是发送给进程的异步通知, 通知程序发生了一个事件.
 * ＜csignal＞头文件包含了六个宏常数, 代表从操作系统到程序的不同信号(这些信号是与操作系统无关的).
 * ?1. SIGTERM 代表终止请求;
 * ?2. SIGSEGV 代表无效的内存访问;
 * ?3. SIGINT 代表外部中断, 如键盘中断;
 * ?4. SIGILL 代表无效的程序镜像;
 * ?5. SIGABRT 代表异常的终止条件, 例如 std::abort;
 * ?6. SIGFPE 代表浮点错误, 例如除以零;
 * 
 * ====要为这些信号注册处理程序,请使用＜csignal＞头文件中的std::signal 函数
 * *Param1. 它接受与上面列出的信号宏之一相对应的单个 int 值作为第一个参数;
 * *Param2. 第二个参数是函数指针(不是函数对象), 该指针指向一个函数, 
 * 该函数接受一个对应于信号宏的int 并返回 void;
 * !此函数必须具有 C 语言链接, 只需将 extern "C" 添加到函数定义前.
 * !请注意, 由于中断的异步特性, 对全局可变状态的任何访问都必须同步.
 */
std::atomic_bool interrupted{};

extern "C" void handler(int signal)
{
    std::cout << "Handler invoked with signal " << signal << ".\n";
    interrupted = true;
}

// -----------------------------------
int main(int argc, const char **argv)
{
    using namespace std::chrono_literals;
    // 注册信号与对应处理程序
    std::signal(SIGINT, handler);

    while (!interrupted)
    {
        std::cout << "Waiting..." << std::endl;
        std::this_thread::sleep_for(1s);
    }
    std::cout << "Interrupted!\n";

    // 通常情况下,在现代操作系统中,可以通过按下 ＜Ctrl+C＞ 引起键盘中断

    return 0;
}
