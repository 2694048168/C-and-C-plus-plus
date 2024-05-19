/**
 * @file 02_sharingCoordinating.cpp
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

/**
  * @brief  共享和同步
  * 只要任务不需要同步, 并且不涉及易变数据共享, 用异步任务进行并发编程就很简单.
  * 例如, 考虑两个线程访问同一个整数的简单情况;
  * 一个线程将递增该整数, 而另一个线程将递减它;
  * 要修改一个变量, 每个线程必须读取该变量的当前值, 执行加减操作, 然后将该变量写入内存;
  * 如果没有同步, 两个线程将以未定义的、交错的顺序执行这些操作.
  * *这种情况有时被称为竞争条件, 因为其结果取决于哪个线程先执行, 这种情况具有的灾难性
  * 
  */
void goat_rodeo()
{
    const size_t iterations{1'000'000};
    int          tin_cans_available{};

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

int main(int argc, const char **argv)
{
    /**
     * @brief 这种数据竞争的根本问题是对可变共享数据的不同步访问.
     * 可能想知道为什么当线程计算 cans_available+1 或 cans_available-1 的时候 cans_available 不更新.
     * *答案在于的每一行都代表了某个指令完成执行的时间点, 而加、减、读、写的指令都是独立的.
     * 因为 cans_available 变量是共享变量, 而且两个线程在没有同步动作的情况下向其写入,
     * 所以这些指令在运行时以一种未定义的方式交错进行(造成灾难性的结果)
     */
    std::cout << "\n========= Sharing and Coordinating ==========\n";
    goat_rodeo();
    goat_rodeo();
    goat_rodeo();

    // ?处理这种"可变共享数据的同步访问"的三种工具：互斥锁、原子量和条件变量

    return 0;
}
