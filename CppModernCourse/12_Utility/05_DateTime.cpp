/**
 * @file 05_DateTime.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>

/**
 * @brief Date and Time 日期和时间
 * stdlib 和 Boost 有许多可以处理日期和时间的库.
 * !在处理日历日期和时间时, 请查看Boost 的 DateTime 库;
 * *当尝试获取当前时间或测量经过的时间时, 请查看stdlib的 Chrono 库以及Boost的Timer 库.
 * 
 * Boost DateTime 库支持基于公历的丰富系统的日期编程, 公历是国际上使用最广泛的民用日历.
 * 1. 构建日期
 * 2. 访问日期成员
 * 3. 日历运算
 * 4. 日期区间
 * 5. 其他 DateTime 功能
 * 
 * ==== stdlib Chrono库在＜chrono＞头文件中提供了各种时钟
 * *当需要编写依赖于时间或为代码计时的东西时
 *
 * ======时钟 Clocks
 * 1. std::chrono::system_clock 是系统范围的实时时钟(wall clock),自特定实现开始日期以来经过的实时时间;
 *    大多数实现都将 Unix 的开始日期指定为 1970 年 1 月 1 日午夜.
 * 2. std::chrono::steady_clock 保证它的值永远不会减少,这似乎很荒谬,但时间测量比看起来的更复杂.
 *    例如系统可能不得不应对闰秒或不准确的时钟.
 * 3. std::chrono::high_resolution_clock 具有最短的可用 tick 周期,
 *    tick 是时钟可以测量的最小原子改变.
 * *这三个时钟都支持静态成员函数 now, 它返回一个与时钟当前值对应的时间点.
 * 
 * ======时间点 Time Points
 * 时间点表示某个具体时间, Chrono 使用std::chrono::time_point 类型对时间点进行编码.
 * 从用户的角度来看, time_point 对象非常简单, 它们提供了一个 time_since_epoch 方法,
 * 该方法返回时间点和时钟纪元(epoch)之间的时间量, 这个时间量称为 duration(持续时间).
 * *纪元是实现定义的参考时间点, 表示时钟的开始时间;
 * *Unix 纪元(或 POSIX 时间)开始于 1970 年 1 月 1 日;
 * Windows 纪元开始于 1601 年 1 月 1 日(对应于 400 年公历周期的开始);
 * time_since_epoch 方法不是从 time_point 获取持续时间的唯一方法;
 * 通过将两个 time_point 对象相减即可获得它们之间的持续时间.
 * 
 * ======持续时间 Durations
 * std::chrono::duration 表示两个 time_point 对象之间的时间.
 * *持续时间暴露了一个计数方法(count), 该方法返回持续时间内的时钟 tick 数.
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("[=====]std::chrono supports several clocks\n");

    auto sys_now    = std::chrono::system_clock::now();
    auto hires_now  = std::chrono::high_resolution_clock::now();
    auto steady_now = std::chrono::steady_clock::now();

    assert(sys_now.time_since_epoch().count() > 0);
    assert(hires_now.time_since_epoch().count() > 0);
    assert(steady_now.time_since_epoch().count() > 0);

    printf("sys_now.time_since_epoch().count(): %lld\n", sys_now.time_since_epoch().count());
    printf("hires_now.time_since_epoch().count(): %lld\n", hires_now.time_since_epoch().count());
    printf("steady_now.time_since_epoch().count(): %lld\n", steady_now.time_since_epoch().count());

    /**
     * *std::chrono 命名空间包含生成持续时间的辅助函数(helper function).
     * *Chrono 在 std::literals::chrono_literals 命名空间中提供了许多用户自定义持续时间字面量.
     * std::chrono::milliseconds(3600000) == 3600000ms
     * 
     */
    printf("\n[=====]std::chrono supports several units of measurement\n");
    auto one_sec = std::chrono::seconds(1);

    using namespace std::literals::chrono_literals;
    auto thousand_ms = 1000ms;
    assert(one_sec == thousand_ms);
    printf("the helper function: %lld, and literals: %lld\n", one_sec.count(), thousand_ms.count());

    /**
     * @brief Chrono 提供了函数模板 std::chrono::duration_cast 
     * 以将持续时间从一种单位转换为另一种单位.
     * std::chrono::duration_cast 也接受与目标持续时间相对应的单个模板参数和与要转换的持续时间相对应的单个参数.
     * 
     */
    printf("\n[=====]std::chrono supports duration_cast\n");
    auto billion_ns_as_s = std::chrono::duration_cast<std::chrono::seconds>(1'000'000'000ns);
    assert(billion_ns_as_s.count() == 1);
    printf("the duration_cast: %lld s\n", billion_ns_as_s.count());

    /**
     * @brief 等待 Waiting
     * *有时使用持续时间对象来指定程序等待的时间段;
     * stdlib 在＜thread＞头文件中提供了并发原语,
     *  其中包含非成员函数 std::this_thread::sleep_for,
     * sleep_for 函数接受一个持续时间参数, 该参数对应于希望当前执行线程等待或"休眠"的时间.
     * 
     */
    printf("\n[=====]std::chrono used to sleep\n");
    auto start = std::chrono::system_clock::now();
    std::this_thread::sleep_for(100ms);
    auto end = std::chrono::system_clock::now();
    assert(end - start >= 100ms);
    printf("the waiting time: %lld\n", (end - start).count());

    return 0;
}
