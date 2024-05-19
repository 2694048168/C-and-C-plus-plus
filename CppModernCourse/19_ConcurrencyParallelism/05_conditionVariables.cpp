/**
 * @file 05_conditionVariables.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>

/** C++ stdlib 在＜condition_variable＞头文件中提供了 std::condition_variable.
 * @brief 条件变量(condition variable)是一个同步原语,
 * 它可以阻塞多个线程, 直到收到通知;
 * 另一个线程可以通知该条件变量, 在收到通知之后, 条件变量可以解除对多个线程的锁定,从而使它们继续执行.
 * *流行的条件变量模式下的线程执行以下动作:
 * 1）获取与等待线程共享的互斥锁;
 * 2）修改共享状态;
 * 3）通知条件变量;
 * 4）释放互斥锁;
 * *任何在条件变量上等待的线程都会执行以下动作:
 * 1）获取互斥锁;
 * 2）在条件变量上等待(这会释放互斥锁);
 * 3）当另一个线程通知条件变量时,该线程被唤醒并执行一些工作(这会自动重新获取互斥锁);
 * 4）释放互斥锁;
 *
 * !=====由于现代操作系统的复杂性导致的复杂情况,有时线程会被虚假地唤醒, 
 * 因此一旦一个等待线程被唤醒, 验证条件变量是否真的发出了信号是很重要的.
 */

void goat_rodeo()
{
    std::mutex              mut;
    std::condition_variable cv;

    const size_t iterations{1'000'000};
    int          tin_cans_available{};

    auto eat_cans = std::async(std::launch::async,
                               [&]
                               {
                                   std::unique_lock<std::mutex> lock{mut};
                                   // !记得必须检查所有的 cans 是否可用,因为存在虚假唤醒
                                   cv.wait(lock, [&] { return tin_cans_available == 1'000'000; });

                                   for (size_t i{}; i < iterations; i++)
                                   {
                                       tin_cans_available--;
                                   }
                               });

    auto deposit_cans = std::async(std::launch::async,
                                   [&]
                                   {
                                       std::scoped_lock<std::mutex> lock{mut};

                                       for (size_t i{}; i < iterations; i++)
                                       {
                                           tin_cans_available++;
                                       }

                                       cv.notify_all();
                                   });
    eat_cans.get();
    deposit_cans.get();
    std::cout << "Tin cans: " << tin_cans_available << "\n";
}

// -----------------------------------
int main(int argc, const char **argv)
{
    std::cout << "\n========= condition_variable Operator ==========\n";
    goat_rodeo();
    goat_rodeo();
    goat_rodeo();
    goat_rodeo();
    goat_rodeo();
    goat_rodeo();
    goat_rodeo();

    return 0;
}
