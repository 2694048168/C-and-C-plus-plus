/**
 * @file 00_concurrencyParallelism.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <future>
#include <ios>
#include <iostream>
#include <thread>

/**
 * @brief 并发(concurrency) && 并行(parallelism)
 * 在编程中,并发(concurrency)意味着在一个特定的时间段内运行两个或更多的任务;
 * 并行(parallelism)是指两个或多个任务在同一时刻运行;
 * *并发和并行编程以及如何使用互斥锁、条件变量和原子安全地共享数据.
 * *执行策略如何帮助提高代码的速度, 以及其中隐藏的危险.
 * 
 * ====并发程序有多个执行线程(简称线程),它们是指令序列.
 * 在大多数运行环境中,操作系统充当调度器,决定线程何时执行下一条指令.
 * 每个进程可以有多个线程,这些线程通常相互共享资源,如内存.
 * 由于调度器决定了线程的执行时间,因此程序员一般不能依赖它们的排序.
 * 作为交换,程序可以在同一时间段(或在同一时间)执行多个任务, 这往往会导致明显的速度提升.
 * 要观察到从串行版本到并发版本的速度提升效果, 系统将需要并发硬件, 例如多核处理器.
 * *异步任务, 这是一种使程序并发的高级方法.
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief std::async
     * 当调用 std::async 时, 第一个参数是启动策略 std::launch, 有两个值:
     * std::launch::async 或 std::launch::deferred.
     * 如果传递 launch::async,运行时将创建一个新线程来启动任务;
     * 如果传递 deferred, 运行时就会等待, 直到需要任务的结果时才执行(被称为惰性求值);
     * 第一个参数是可选的,默认为async|deferred, 也就是说, 采用哪种策略由实现者决定;
     * std::async 的第二个参数是函数对象, 代表要执行的任务,
     * 函数对象接受的参数数量或类型没有限制, 它可以返回任意类型.
     * ?std::async 函数是一个带有函数参数包的可变参数模板, 
     * 当异步任务启动时,在函数对象之外传递的任何额外参数将被用于调用函数对象,
     * 另外 std::async 返回一个 std::future 对象.
     * ?std::future<FuncReturnType> std::async([policy], func, Args&&... args);
     * 
     * ====future 是一个类模板,它持有异步任务的值.
     * 它有一个模板参数, 与异步任务的返回值的类型相对应;
     * 例如, 如果传递一个返回 std::string 的函数对象, async 将返回 future＜string＞.
     * *给定 future, 可以通过三种方式与异步任务进行交互
     * 
     */

    /**
      * @brief 第一可以使用 valid 方法查询 future 的有效性,
      * 有效的 future 有一个与之相关的共享状态.
      * 异步任务也有一个共享状态, 因此它们可以交流结果.
      * 任何由 async 返回的 future 都是有效的, 直到检索到异步任务的返回值, 
      * 这时共享状态的生命周期就结束了
      */
    std::cout << "[====]std::async returns valid future\n";
    using namespace std::literals::string_literals;

    auto the_future = std::async([] { return "female"s; });
    std::cout << "The future is valid: " << std::boolalpha << the_future.valid() << '\n';

    std::cout << "[====]std::future invalid by default\n";
    std::future<bool> default_future;
    std::cout << "The future is valid: " << std::boolalpha << default_future.valid() << '\n';

    /**
     * @brief 第二可以用 get 方法从有效的 future 中获取值,
     * *如果异步任务还没有完成, 对 get 的调用将阻塞当前执行的线程, 直到结果可用.
     * ?如果异步任务抛出异常,future 将收集该异常并在 get 被调用时抛出
     */
    std::cout << "[====]std::async returns the return value of the function object\n";
    auto the_future2 = std::async([] { return "female"s; });
    std::cout << "The result value of std::async: " << the_future2.get() << '\n';

    // !get may throw
    auto ghostrider = std::async([] { throw std::runtime_error{"The pattern is full."}; });
    try
    {
        ghostrider.get();
    }
    catch (const std::exception &exp)
    {
        std::cout << "The Exception: " << exp.what() << '\n';
    }

    /**
     * @brief 第三可以使用 std::wait_for 或 std::wait_until 检查异步任务是否已经完成,
     * 选择哪一种取决于想传递的 chrono 的类型,
     * 对于 duration 对象, 使用wait_for; 对于 time_point 对象, 使用 wait_until;
     * 两者都返回 std::future_status, 它有三个值:
     * 1. future_status::deferred 表明异步任务将被延迟评估, 所以一旦调用 get 任务就会执行;
     * 2. future_status::ready 表示任务已经完成, 结果已经就绪;
     * 3. future_status::timeout 表示该任务还没有准备好;
     * *如果任务在指定的等待期之前完成, std::async 将提前返回.
     */
    std::cout << "[====]wait_for indicates whether a task is ready\n";
    using namespace std::literals::chrono_literals;
    auto sleepy = std::async(std::launch::async, [] { std::this_thread::sleep_for(100ms); });
    // ?this_thread::sleep_for 它并不准确, 操作环境负责调度线程, 它可能会把正在睡眠的线程安排得比指定的时间晚

    const auto not_ready_yet = sleepy.wait_for(25ms);
    assert(not_ready_yet == std::future_status::timeout);

    const auto totally_ready = sleepy.wait_for(100ms);
    assert(totally_ready == std::future_status::ready);

    return 0;
}
