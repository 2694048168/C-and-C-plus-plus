/**
 * @file 03_mutex.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <future>
#include <iostream>
#include <mutex>

/**
 * @brief 互斥锁(mutex)是一种防止多个线程同时访问资源的机制.
 * 它是同步原语, 支持两种操作: 锁定和解锁.
 * 当线程需要访问共享数据时, 它就会锁住互斥锁; 这个操作可能会阻塞,
 * 这取决于互斥锁的性质以及其他线程是否拥有该锁.
 * 当线程不再需要访问时, 它就会解锁该互斥锁.
 * ====＜mutex＞ 头文件公开了几个互斥锁选项:
 * ?1. std::mutex 提供了基本的互斥锁;
 * ?2. std::timed_mutex 提供了带有超时机制的互斥锁;
 * ?3. std::recursive_mutex 提供了互斥锁, 允许同一线程进行递归锁定;
 * ?4. std::recursive_timed_mutex 提供了互斥锁, 允许同一线程进行递归锁定并有超时机制;
 * ====＜shared_mutex＞ 头文件提供了两个额外的选项:
 * ?1. std::shared_mutex 提供了共享互斥锁,这意味着几个线程可以同时拥有该互斥锁;
 *     这个选项通常用在多个读取操作访问共享数据, 但一个写入操作需要独占访问的场景;
 * ?2. std::shared_timed_mutex 提供了共享互斥锁, 并实现了有超时机制的锁定;
 * 
 */

// *使用 mutex 解决代码中的竞争条件问题
void goat_rodeo()
{
    const size_t iterations{1'000'000};
    int          tin_cans_available{};

    std::mutex tin_can_mutex;
    auto       eat_cans = std::async(std::launch::async,
                                     [&]
                                     {
                                   for (size_t i{}; i < iterations; i++)
                                   {
                                       tin_can_mutex.lock();
                                       tin_cans_available--;
                                       tin_can_mutex.unlock();
                                   }
                               });

    auto deposit_cans = std::async(std::launch::async,
                                   [&]
                                   {
                                       for (size_t i{}; i < iterations; i++)
                                       {
                                           tin_can_mutex.lock();
                                           tin_cans_available++;
                                           tin_can_mutex.unlock();
                                       }
                                   });
    eat_cans.get();
    deposit_cans.get();
    std::cout << "Tin cans: " << tin_cans_available << "\n";
}

/**
 * @brief 处理互斥锁对于 RAII 对象来说是一项完美的工作.
 * 假设忘记在互斥锁上调用unlock, 例如因为它引发了异常.
 * 当下一个线程出现并尝试通过 lock 获取互斥锁时, 程序将突然停止.
 * C++ 出于这个原因 stdlib 在＜mutex＞头文件中提供了 RAII 类来处理互斥锁.
 * *几个类模板, 它们都将互斥锁作为构造函数参数并接受一个对应于互斥锁类别的模板参数.
 * ?1. std::lock_guard 是一个不可复制,不可移动的 RAII 包装器,它在构造函数中接受一个互斥锁对象,
 *   在那里调用 lock, 然后在析构函数中调用 unlock;
 * ?2. std::scoped_lock 是一个避免死锁的 RAII 包装器, 用于多个互斥锁;
 * ?3. std::unique_lock 实现了一个可移动的互斥锁所有权包装器;
 * ?4. std::shared_lock 实现了一个可移动的共享互斥锁所有权包装器;
 * 
 */
void goat_rodeo_()
{
    const size_t iterations{1'000'000};
    int          tin_cans_available{};

    std::mutex tin_can_mutex;
    auto       eat_cans = std::async(std::launch::async,
                                     [&]
                                     {
                                   for (size_t i{}; i < iterations; i++)
                                   {
                                       std::lock_guard<std::mutex> guard{tin_can_mutex};
                                       tin_cans_available--;
                                   }
                               });

    auto deposit_cans = std::async(std::launch::async,
                                   [&]
                                   {
                                       for (size_t i{}; i < iterations; i++)
                                       {
                                           std::lock_guard<std::mutex> guard{tin_can_mutex};
                                           tin_cans_available++;
                                       }
                                   });
    eat_cans.get();
    deposit_cans.get();
    std::cout << "Tin cans: " << tin_cans_available << "\n";
}

// -----------------------------------
int main(int argc, const char **argv)
{
    std::cout << "\n========= std::mutex Coordinating ==========\n";
    goat_rodeo();
    goat_rodeo();
    goat_rodeo();

    std::cout << "\n========= std::lock_guard<mutex> Coordinating ==========\n";
    goat_rodeo_();
    goat_rodeo_();
    goat_rodeo_();

    /**
     * @brief 相对于递增或递减整数(任务操作),
     * *获取或释放锁要花费更多时间, 因此使用互斥锁来同步异步任务不是最佳选择.
     * 在某些情况下, 可以通过原子量来采取可能更高效的方法.
     */

    return 0;
}
