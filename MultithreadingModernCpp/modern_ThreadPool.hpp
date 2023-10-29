/**
 * @file modern_ThreadPool.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-29
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef THREAD_POOL_H // 防止头文件被重复包含
#define THREAD_POOL_H // 与上面的宏定义对应，与底部的 #endif 是一对

#include <condition_variable> // 条件变量，比如 std::condition_variable
#include <functional>         // 函数式编程 std::function
#include <future>             // 异步编程，比如 std::future
#include <iostream>
#include <memory>    // 智能指针，比如 std::shared_ptr
#include <mutex>     // 互斥量，比如 std::mutex
#include <queue>     // 队列，比如 std::queue
#include <stdexcept> // 异常处理，比如 std::runtime_error
#include <thread>    // 线程，比如 std::thread
#include <vector>    // 容器，比如 std::vector

class ThreadPool
{                                       // 线程池【类】
public:                                 // 公有成员
    ThreadPool(size_t);                 // 构造函数声明，size_t是无符号整数
    template<class F, class... Args>    // 【模板】函数声明，向线程池中添加任务
    auto enqueue(F &&f, Args &&...args) // 参数是一个函数和函数的参数
        -> std::future<typename std::result_of<F(Args...)>::type>; // 返回值是一个 std::future 对象
    ~ThreadPool();                                                 // 析构函数声明

private:                                           // 【私有成员】
    std::vector<std::thread>          workers;     // 线程池中的线程列表
    std::queue<std::function<void()>> tasks;       // 任务队列
    std::mutex                        queue_mutex; // 互斥量，用于保证线程安全
    std::condition_variable           condition;   // 条件变量，用于实现线程同步
    bool                              stop;        // 线程池是否停止的标志（在析构函数中才会停止）
};

/**
 * 构造函数，往线程列表中添加 threads 个线程，每个线程都会立即执行，有任务就执行任务，没有任务就等待。
 * @param threads 线程池中线程的数量
 */
inline ThreadPool::ThreadPool(size_t threads) // inline 建议编译器将函数体内联展开，提高执行效率
    : stop(false)
{ // stop的默认值是false，表示线程池没有停止【参数默认值】
    for (size_t i = 0; i < threads; ++i)
    {
        auto threadLambda = [this]
        {
            // 无限循环，但并不会占满cpu，因为没有任务就会等待；当线程池停止、且任务队列为空时，从循环中退出
            for (;;)
            {
                std::function<void()> task; // 任务，是一个function函数
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex); // 获取锁
                    // 如果线程池没有停止，且任务队列为空，则等待。
                    // 线程池仅在析构函数中停止，这里可以简单理解为：如果没有任务就一直等任务。
                    this->condition.wait(lock, [this] {            // 【lambda】
                        return this->stop || !this->tasks.empty(); // 符合这个条件就往下执行
                    });
                    // 线程池停止，且任务队列为空，退出线程
                    if (this->stop && this->tasks.empty())
                    {
                        return; // 退出for循环，线程结束
                    }
                    task = std::move(this->tasks.front()); // 找到任务队列中最前面的任务【move】
                    this->tasks.pop();                     // 从任务队列中移除这个任务
                }                                          // 在这个反括号这里会自动释放锁
                task();                                    // 执行任务
            }
        };
        workers.emplace_back(threadLambda); // 将线程放入线程列表
    }
}

/**
 * 添加任务到任务队列中，返回一个 std::future 对象，用于获取任务执行结果。
 * @tparam F 函数类型
 * @tparam Args 参数类型
 * @param f 函数
 * @param args 参数
 * @return std::future 对象，用于获取任务执行结果
 */
template<class F, class... Args>
auto ThreadPool::enqueue(F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>
{
    // 由于泛型的存在，return_type 可能是任何类型，以下假设它是 int
    using return_type = typename std::result_of<F(Args...)>::type;

    // std::forward<F>(f)：f是F类型，是一个函数。这里使用std::forward，可以避免拷贝对象，是一种性能优化手段
    // std::forward<Args>(args)... 同理。其中，...表示将参数包展开：std::forward<Args>(arg1), ...
    // std::bind 将f和args绑定，即将函数和参数绑定，返回一个可调用对象，类型是 std::_Bind< <lambda> () >
    auto callableBind = std::bind(std::forward<F>(f), std::forward<Args>(args)...);

    // std::packaged_task<return_type()>：包装一个可调用对象，可以异步获取结果，结果的类型是return_type
    // std::make_shared用于创建一个智能指针shared_ptr，指向一个新创建的packaged_task对象。【智能指针】
    // task的类型：std::shared_ptr<std::packaged_task<int()> >
    auto task = std::make_shared<std::packaged_task<return_type()>>(callableBind);

    // res返回结果: std::future<int>
    std::future<return_type> res = task->get_future();
    {
        // 上锁，保证线程安全，防止多个线程同时往队列中添加任务
        std::unique_lock<std::mutex> lock(queue_mutex);

        // 如果线程池停止，则抛出异常
        if (stop)
        {
            throw std::runtime_error("enqueue on stopped ThreadPool"); // 【异常处理】
        }

        // 定义一个函数，用于执行 task
        auto taskFunction = [task]()
        {
            (*task)();
        };
        // 将这个函数放入任务队列
        tasks.emplace(taskFunction);
    } // 在这个反括号这里会自动释放锁
    condition.notify_one();
    return res;
}

/**
 * 析构函数，停止线程池，并等待所有线程执行完毕
 */
inline ThreadPool::~ThreadPool()
{
    std::cout << "ThreadPool destructor" << std::endl;
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    } // 在这个反括号这里会自动释放锁
    condition.notify_all();
    for (std::thread &worker : workers)
    {
        worker.join();
    }
}

#endif // 与上面的 #define THREAD_POOL_H 是一对