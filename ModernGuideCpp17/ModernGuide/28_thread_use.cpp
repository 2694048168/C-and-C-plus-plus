/**
 * @file 28_thread_use.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

/** C++11之前,C++语言没有对并发编程提供语言级别的支持, 这使得在编写可移植的并发程序时,存在诸多的不便.
 * C++11中增加了线程以及线程相关的类,很方便地支持了并发编程,使得编写的多线程程序的可移植性得到了很大的提高.
 * C++11中提供的线程类叫做std::thread,基于这个类创建一个新的线程非常的简单,只需要提供线程函数或者函数对象即可
 * ?并且可以同时指定线程函数的参数.
 * 
 * *joinable()函数用于判断主线程和子线程是否处理关联(连接)状态,该函数返回一个布尔类型.
 * ?1. 在创建的子线程对象的时候,如果没有指定任务函数,那么子线程不会启动,主线程和这个子线程也不会进行连接;
 * ?2. 在创建的子线程对象的时候,如果指定了任务函数,子线程启动并执行任务,主线程和这个子线程自动连接成功;
 * ?3. 子线程调用了detach()函数之后,父子线程分离,同时二者的连接断开,调用joinable()返回false;
 * ?4. 在子线程调用了join()函数,子线程中的任务函数继续执行,直到任务处理完毕,这时join()会清理(回收)当前子线程的相关资源,
 *   所以这个子线程和主线程的连接也就断开了, 因此调用join()之后再调用joinable()会返回false;
 * 
 * 静态函数
 * thread线程类还提供了一个静态方法,用于获取当前计算机的CPU核心数,
 * 根据这个结果在程序中创建出数量相等的线程,每个线程独自占有一个CPU核心,
 * *这些线程就不用分时复用CPU时间片,此时程序的并发效率是最高的.
 * 
 * 
 * 
 */

void func(int num, std::string str)
{
    for (int i = 0; i < 10; ++i)
    {
        std::cout << "子线程: i = " << i << " num: " << num << ", str: " << str << std::endl;
    }
}

void func1()
{
    for (int i = 0; i < 10; ++i)
    {
        std::cout << "子线程: i = " << i << std::endl;
    }
}

void download1()
{
    // 模拟下载, 总共耗时500ms，阻塞线程500ms
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    std::cout << "子线程1: " << std::this_thread::get_id() << ", 找到历史正文....\n";
}

void download2()
{
    // 模拟下载, 总共耗时300ms，阻塞线程300ms
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    std::cout << "子线程2: " << std::this_thread::get_id() << ", 找到历史正文....\n";
}

void doSomething()
{
    std::cout << "集齐历史正文, 呼叫罗宾...." << std::endl;
    std::cout << "历史正文解析中...." << std::endl;
    std::cout << "起航，前往拉夫德尔...." << std::endl;
    std::cout << "找到OnePiece, 成为海贼王, 哈哈哈!!!" << std::endl;
    std::cout << "若干年后，草帽全员卒...." << std::endl;
    std::cout << "大海贼时代再次被开启...." << std::endl;
}

// --------------------------------------
int main(int argc, const char **argv)
{
    std::cout << "主线程的线程ID: " << std::this_thread::get_id() << std::endl;

    std::thread t(func, 520, "i love you");
    std::thread t1(func1);
    std::cout << "线程t 的线程ID: " << t.get_id() << std::endl;
    std::cout << "线程t1的线程ID: " << t1.get_id() << std::endl;

    // 当启动了一个线程(创建了一个thread对象)之后,在这个线程结束的时候(std::terminate()),
    // 如何去回收线程所使用的资源呢？thread库给两种选择：
    // *1. 加入式 join()
    // *2. 分离式 detach()
    // 另外必须要在线程对象销毁之前在二者之间作出选择，否则程序运行期间就会有bug产生.

    // join()字面意思是连接一个线程,意味着主动地等待线程的终止(线程阻塞).
    // 在某个线程中通过子线程对象调用join()函数,调用这个函数的线程被阻塞,
    // 但是子线程对象中的任务函数会继续执行,当任务执行完毕之后join()会清理当前子线程中的相关资源然后返回,
    // 同时调用该函数的线程解除阻塞继续向下执行.
    if (t.joinable())
        t.join();

    if (t1.joinable())
        t1.join();

    // =============业务处理逻辑
    std::thread task_1(download1);
    std::thread task_2(download2);
    // 阻塞主线程，等待所有子线程任务执行完毕再继续向下执行
    task_1.join();
    task_2.join();
    doSomething();

    // detach()函数的作用是进行线程分离,分离主线程和创建出的子线程.
    // !在线程分离之后,主线程退出也会一并销毁创建出的所有子线程;
    // 在主线程退出之前,它可以脱离主线程继续独立的运行,任务执行完毕之后,这个子线程会自动释放自己占用的系统资源.
    // * 注意事项: 线程分离函数detach()不会阻塞线程,子线程和主线程分离之后,在主线程中就不能再对这个子线程做任何控制了.
    std::thread task_3(func, 520, "i love you");
    std::thread task_4(func1);
    std::cout << "线程t 的线程ID: " << t.get_id() << std::endl;
    std::cout << "线程t1的线程ID: " << t1.get_id() << std::endl;
    task_3.detach();
    task_4.detach();
    // 让主线程休眠, 等待子线程执行完毕
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // 当前CPU的逻辑处理器数量
    const int num = std::thread::hardware_concurrency();
    std::cout << "CPU logic_processor number: " << num << std::endl;

    return 0;
}
