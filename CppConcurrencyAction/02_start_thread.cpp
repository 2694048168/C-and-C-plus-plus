/**
 * @file 02_start_thread.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-06-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <assert.h>

#include <iostream>
#include <string>
// std::thread 可以通过有函数操作符类型的实例进行构造
#include <thread>

void do_some_work()
{
    std::cout << "do some work\n";
}

void do_some_work_else()
{
    std::cout << "do some work else\n";
}

class BackgroundTask
{
public:
    void operator()() const
    {
        do_some_work();
        do_some_work_else();
    }
};

// 函数已经返回，线程依旧访问局部变量
void do_something(int &i)
{
    std::cout << "the reference value: " << i << std::endl;
}

class Func
{
public:
    int &i;

    Func(int &i_)
        : i(i_)
    {
    }

    void operator()()
    {
        for (size_t j = 0; j < 10; ++j)
        {
            do_something(i);
        }
    }
};

void oops()
{
    int  some_local_state = 0;
    Func my_func(some_local_state);

    // ! 使用访问局部变量的函数去创建线程是一个糟糕的主意。
    std::thread my_thread{my_func};
    // 分离线程,该函数结束会进行析构销毁局部变量,而新的线程有可能会访问该变量(undefined behaved)
    // my_thread.detach();
    if (my_thread.joinable())
    {
        // 等待线程完成, 类似阻塞等待？
        my_thread.join();
    }
}

void do_something_in_current_thread() {}

void f()
{
    int  local_state = 0;
    Func my_func(local_state);

    std::thread t{my_func};
    // 如果等待线程，则需要细心挑选使用join()的位置
    // 当在线程运行后产生的异常，会在join()调用之前抛出，这样就会跳过join()。
    // 通常，在无异常的情况下使用join()时，需要在异常处理过程中调用join()，
    // 从而避免生命周期的问题。
    try
    {
        do_something_in_current_thread();
    }
    catch (...)
    {
        t.join();
        throw;
    }
    t.join();
}

// 使用RAII等待线程完成
class ThreadGuard
{
private:
    std::thread &t;

public:
    explicit ThreadGuard(std::thread &t_)
        : t(t_)
    {
    }

    ThreadGuard(const ThreadGuard &) = delete;
    // * 如果不删除拷贝构造函数和拷贝赋值操作, 会出现什么情况？
    // * 拷贝构造函数和拷贝赋值操作标记为 =delete，是为了不让编译器自动生成
    ThreadGuard &operator=(const ThreadGuard &) = delete;

    ~ThreadGuard()
    {
        // t.detach();
        if (t.joinable())
        {
            t.join();
        }
    }
};

void foo()
{
    int some_local_state = 0;

    Func my_func(some_local_state);

    std::thread t(my_func);

    ThreadGuard g(t);

    do_something_in_current_thread();
}

void do_background_work() {}

// 使用分离线程处理文档
// void edit_document(const std::string &filename)
// {
//     open_document_and_display_gui(filename);
//     while (!done_editing())
//     {
//         user_command cmd = get_user_input();
//         if (cmd.type == open_new_document)
//         {
//             std::string const new_name = get_filename_from_user();

//             std::thread t(edit_document, new_name);
//             t.detach();
//         }
//         else
//         {
//             process_user_input(cmd);
//         }
//     }
// }

int main(int argc, const char **argv)
{
    // 使用C++线程库启动线程，就是构造 std::thread 对象
    std::thread my_thread1{do_some_work};
    my_thread1.join();

    BackgroundTask task;

    // 提供的函数对象会复制到新线程的存储空间中，函数对象的执行和调用都在线程的内存空间中进行
    std::thread my_thread2{task};
    my_thread2.join();

    // ! 注意，当把函数对象传入到线程构造函数中时，需要避免“最令人头痛的语法解析”
    // ? 如果你传递了一个临时变量，而不是一个命名的变量,
    // ? C++编译器会将其解析为函数声明，而不是类型对象的定义。
    // std::thread my_thread3(BackgroundTask());
    std::thread my_thread4{BackgroundTask()};
    std::thread my_thread5((BackgroundTask()));
    my_thread4.join();
    my_thread5.join();

    // * Lambda表达式也能避免这个问题
    std::thread my_thread6(
        []()
        {
            do_some_work();
            do_some_work_else();
        });
    my_thread6.join();

    // 线程启动后是要等待线程结束，还是让其自主运行?
    // 当 std::thread 对象销毁之前还没有做出决定，
    // 程序就会终止( std::thread 的析构函数会调用 std::terminate() )
    // 因此，即便是有异常存在，也需要确保线程能够正确汇入(joined)或分离(detached)。
    oops();
    f();

    // 使用“资源获取即初始化方式”(RAII，Resource Acquisition Is Initialization)，
    // 提供一个类，在析构函数中使用join()。
    foo();

    // 使用detach()会让线程在后台运行，这就意味着与主线程不能直接交互
    // 分离线程通常称为守护线程(daemon threads)
    // 分离线程只能确定线程什么时候结束，发后即忘(fire and forget)的任务使用到就是分离线程
    std::thread daemon_thread{do_background_work};
    if (daemon_thread.joinable())
    {
        daemon_thread.detach();
    }
    assert(!daemon_thread.joinable());

    return 0;
}