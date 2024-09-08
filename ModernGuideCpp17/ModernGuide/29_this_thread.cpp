/**
 * @file 29_this_thread.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 命名空间 - std::this_thread
 * 在C++11中不仅添加了线程类,还添加了一个关于线程的命名空间std::this_thread,
 * *在这个命名空间中提供了四个公共的成员函数,通过这些成员函数就可以对当前线程进行相关的操作.
 * ====1. 调用命名空间std::this_thread中的get_id()方法可以得到当前线程的线程ID;
 *
 * ====2. sleep_for();
 *   ?线程被创建后也有这五种状态: 创建态，就绪态，运行态，阻塞态(挂起态)，退出态(终止态);
 * 线程和进程的执行有很多相似之处, 在计算机中启动的多个线程都需要占用CPU资源,
 * 但是CPU的个数是有限的并且每个CPU在同一时间点不能同时处理多个任务.
 * *为了能够实现并发处理,多个线程都是分时复用CPU时间片,快速的交替处理各个线程中的任务.
 * 因此多个线程之间需要争抢CPU时间片,抢到了就执行,抢不到则无法执行
 * (因为默认所有的线程优先级都相同,内核也会从中调度,不会出现某个线程永远抢不到CPU时间片的情况).
 * ?命名空间this_thread中提供了一个休眠函数sleep_for(),
 * 调用这个函数的线程会马上从运行态变成阻塞态并在这种状态下休眠一定的时长,
 * 因为阻塞态的线程已经让出了CPU资源,代码也不会被执行,所以线程休眠过程中对CPU来说没有任何负担.
 *
 * ====3. sleep_until();
 *   指定线程阻塞到某一个指定的时间点time_point类型, 之后解除阻塞;
 *   指定线程阻塞一定的时间长度duration 类型, 之后解除阻塞;
 * sleep_until()和sleep_for()函数的功能是一样的,只不过前者是基于时间点去阻塞线程,后者是基于时间段去阻塞线程
 * ?项目开发过程中根据实际情况选择最优的解决方案即可.
 *
 * =====4. yield();
 * 命名空间this_thread中提供了一个非常绅士的函数yield(),
 * 在线程中调用这个函数之后,处于运行态的线程会主动让出自己已经抢到的CPU时间片,最终变为就绪态,
 * 这样其它的线程就有更大的概率能够抢到CPU时间片了.
 * 使用这个函数的时候需要注意一点,线程调用了yield()之后会主动放弃CPU资源,
 * ?但是这个变为就绪态的线程会马上参与到下一轮CPU的抢夺战中,不排除它能继续抢到CPU时间片的情况,这是概率问题.
 * *std::this_thread::yield() 的目的是避免一个线程长时间占用CPU资源,从而导致多线程处理性能下降;
 * *std::this_thread::yield() 是让当前线程主动放弃了当前自己抢到的CPU资源,但是在下一轮还会继续抢;
 * 
 */

#include <iostream>
#include <thread>

void func()
{
    std::cout << "子线程: " << std::this_thread::get_id() << std::endl;
}

void func_sleep()
{
    for (int i = 0; i < 10; ++i)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // 程序休眠完成之后,会从阻塞态重新变成就绪态,
        // 就绪态的线程需要再次争抢CPU时间片,抢到之后才会变成运行态,这时候程序才会继续向下运行
        std::cout << "子线程: " << std::this_thread::get_id() << ", i = " << i << std::endl;
    }
}

void func_until()
{
    for (int i = 0; i < 10; ++i)
    {
        // 获取当前系统时间点
        auto now = std::chrono::system_clock::now();

        // 时间间隔为2s
        std::chrono::seconds sec(5);

        // 当前时间点之后休眠两秒
        std::this_thread::sleep_until(now + sec);
        std::cout << "子线程: " << std::this_thread::get_id() << ", i = " << i << std::endl;
    }
}

void func_yield()
{
    // 执行func()中的for循环会占用大量的时间,
    // 在极端情况下,如果当前线程占用CPU资源不释放就会导致其他线程中的任务无法被处理,
    // 或者该线程每次都能抢到CPU时间片,导致其他线程中的任务没有机会被执行.
    // 解决方案就是每执行一次循环,让该线程主动放弃CPU资源,重新和其他线程再次抢夺CPU时间片,
    // 如果其他线程抢到了CPU时间片就可以执行相应的任务了.
    for (int i = 0; i < 1000; ++i)
    {
        std::cout << "子线程: " << std::this_thread::get_id() << ", i = " << i << std::endl;
        std::this_thread::yield();
    }
}

// --------------------------------------
int main(int argc, const char **argv)
{
    std::cout << "主线程: " << std::this_thread::get_id() << std::endl;

    std::thread task(func);
    std::thread task_1(func_sleep);
    std::thread task_2(func_until);

    task_1.join();
    task_2.join();
    task.join();

    std::thread thread_task1(func_yield);
    std::thread thread_task2(func_yield);
    thread_task1.join();
    thread_task2.join();

    return 0;
}
