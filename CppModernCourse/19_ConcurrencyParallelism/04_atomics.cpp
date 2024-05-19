/**
 * @file 04_atomics.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <atomic>
#include <future>
#include <iostream>

/**
 * @brief 原子量(atomic)来自希腊语 átomos, 意思是"不可分割的".
 * 如果一个操作发生在一个不可分割的单元中, 那么它就是原子的. 另一个线程无法在中途观察该操作.
 * !这种基于锁的解决方案时所经历的那样, 这种方法非常缓慢, 因为获取锁的成本很高.
 * 使用 ＜atomic＞ 头文件中的 std::atomic 类模板, 
 * 它提供了无锁并发编程中经常使用的原语(primitive),
 * 无锁并发编程在不涉及锁的情况下解决了数据竞争问题.
 * 在许多现代架构上, CPU 支持原子指令; 使用原子量, 可能能够依赖原子硬件指令来避免锁定.
 * *std::atomic 模板为所有基本类型都提供了特化
 */
void goat_rodeo()
{
    const size_t    iterations{1'000'000};
    std::atomic_int tin_cans_available{}; //!Atomic

    auto eat_cans = std::async(std::launch::async,
                               [&]
                               {
                                   for (size_t i{}; i < iterations; i++)
                                   {
                                       tin_cans_available--;
                                   }
                               });

    auto deposit_cans = std::async(std::launch::async,
                                   [&]
                                   {
                                       for (size_t i{}; i < iterations; i++)
                                       {
                                           tin_cans_available++;
                                       }
                                   });
    eat_cans.get();
    deposit_cans.get();
    std::cout << "Tin cans: " << tin_cans_available << "\n";
}

// ----------------------------------
int main(int argc, const char **argv)
{
    std::cout << "\n========= Atomic Operator ==========\n";
    goat_rodeo();
    goat_rodeo();
    goat_rodeo();
    goat_rodeo();

    return 0;
}
