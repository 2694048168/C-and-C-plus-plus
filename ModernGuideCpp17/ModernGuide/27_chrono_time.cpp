/**
 * @file 27_chrono_time.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** C++11中提供了日期和时间相关的库chrono,通过chrono库可以很方便地处理日期和时间,
 * 为程序的开发提供了便利, chrono库主要包含三种类型的类:
 * *1. 时间间隔duration
 * ---duration表示一段时间间隔,用来记录时间长度,可以表示几秒、几分钟、几个小时的时间间隔;
 * ---方便使用在标准库中定义了一些常用的时间间隔,时、分、秒、毫秒、微秒、纳秒,都位于chrono命名空间;
 *   ====纳秒: std::chrono::nanoseconds;
 *   ====微秒: std::chrono::microseconds;
 *   ====毫秒: std::chrono::milliseconds;
 *   ====秒: std::chrono::seconds;
 *   ====分钟: std::chrono::minutes;
 *   ====小时: std::chrono::hours;
 * ---duration类还提供了获取时间间隔的时钟周期数的方法count();
 * *2. 时钟clocks
 * ---chrono库中提供了获取当前的系统时间的时钟类,包含的时钟一共有三种:
 *   ====system_clock:系统的时钟,系统的时钟可以修改,甚至可以网络对时,因此使用系统时间计算时间差可能不准;
 *   ====steady_clock:是固定的时钟,相当于秒表. 开始计时后,时间只会增长并且不能修改,适合用于记录程序耗时;
 *   ====high_resolution_clock: 和时钟类 steady_clock 是等价的(别名);
 * ----时钟类成员类型	描述
 * ==========================================================
 *     rep	         表示时钟周期次数的有符号算术类型
 *     period	     表示时钟计次周期的 std::ratio 类型
 *     duration	     时间间隔，可以表示负时长
 *     time_point	 表示在当前时钟里边记录的时间点
 * ==========================================================
 * *3. 时间点time point
 * ---chrono库中提供了一个表示时间点的类time_point;
 * ---被实现成如同存储一个 Duration 类型的自 Clock 的纪元起始开始的时间间隔的值;
 * ---time_since_epoch()函数,用来获得1970年1月1日到time_point对象中记录的时间经过的时间间隔(duration);
 * ---时间点time_point对象和时间段对象duration之间还支持直接进行算术运算(加减运算),时间点对象之间可以进行逻辑运算;
 * *4. 转换函数
 * ---duration_cast
 *   ====duration_cast是chrono库提供的一个模板函数, 这个函数不属于duration类.
 *   ====通过这个函数可以对duration类对象内部的时钟周期Period, 和周期次数的类型Rep进行修改;
 * ---time_point_cast
 *   ====time_point_cast也是chrono库提供的一个模板函数,这个函数不属于time_point类.
 *   ====函数的作用是对时间点进行转换,因为不同的时间点对象内部的时钟周期Period, 和周期次数的类型Rep可能也是不同的,
 *   ====一般情况下它们之间可以进行隐式类型转换,也可以通过该函数显示的进行转换;
 * 
 */

#include <chrono>
#include <ctime>
#include <iostream>

void func_tasks()
{
    std::cout << "print 1000 stars ....\n";
    for (int i = 0; i < 1000; ++i)
    {
        std::cout << "*";
    }
    std::cout << std::endl;
}

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::milliseconds;
using Sec   = std::chrono::seconds;
template<class Duration>
using TimePoint = std::chrono::time_point<Clock, Duration>;

void print_ms(const TimePoint<Ms> &time_point)
{
    std::cout << time_point.time_since_epoch().count() << " ms\n";
}

// --------------------------------------
int main(int argc, const char **argv)
{
    // 时钟周期为1小时，共有1个时钟周期，所以h表示的时间间隔为1小时
    std::chrono::hours h(1); // 一小时

    // 时钟周期为1毫秒，共有3个时钟周期，所以ms表示的时间间隔为3毫秒
    std::chrono::milliseconds ms{3}; // 3 毫秒

    // 时钟周期为1000秒，一共有三个时钟周期，所以ks表示的时间间隔为3000秒
    std::chrono::duration<int, std::ratio<1000>> ks(3); // 3000 秒

    // 时钟周期为1000秒，时钟周期数量只能用整形来表示，但是此处指定的是浮点数，因此语法错误
    // std::chrono::duration<int, std::ratio<1000>> d3(3.5); // !error

    // 时钟周期为默认的1秒，共有6.6个时钟周期，所以dd表示的时间间隔为6.6秒
    std::chrono::duration<double> dd(6.6); // 6.6 秒

    // 使用小数表示时钟周期的次数
    // 时钟周期为1/30秒，共有3.5个时钟周期，所以hz表示的时间间隔为1/30*3.5秒
    std::chrono::duration<double, std::ratio<1, 30>> hz(3.5);

    // ?chrono库中根据duration类封装了不同长度的时钟周期(也可以自定义),
    // 基于这个时钟周期再进行周期次数的设置就可以得到总的时间间隔了
    // *时钟周期 * 周期次数 = 总的时间间隔
    std::chrono::milliseconds                        ms_{3};        // 3 毫秒
    std::chrono::microseconds                        us_ = 2 * ms_; // 6000 微秒
    // 时间间隔周期为 1/30 秒
    std::chrono::duration<double, std::ratio<1, 30>> hz_(3.5);

    std::cout << "3 ms duration has " << ms_.count() << " ticks\n"
              << "6000 us duration has " << us_.count() << " ticks\n"
              << "3.5 hz duration has " << hz_.count() << " ticks\n";

    // 由于在duration类内部做了操作符重载,因此时间间隔之间可以直接进行算术运算,比如计算两个时间间隔的差值
    // ?注意事项:duration的加减运算有一定的规则,当两个duration时钟周期不相同的时候,会先统一成一种时钟，然后再进行算术运算
    std::chrono::minutes t1(10);
    std::chrono::seconds t2(60);
    std::chrono::seconds t3 = t1 - t2;
    std::cout << t3.count() << " second" << std::endl;

    // =====================================================
    // 新纪元1970.1.1时间
    std::chrono::system_clock::time_point epoch;

    std::chrono::duration<int, std::ratio<60 * 60 * 24>> day(1);
    // 新纪元1970.1.1时间 + 1天
    std::chrono::system_clock::time_point                ppt(day);

    using dday = std::chrono::duration<int, std::ratio<60 * 60 * 24>>;
    // 新纪元1970.1.1时间 + 10天
    std::chrono::time_point<std::chrono::system_clock, dday> t(dday(10));

    // 系统当前时间
    std::chrono::system_clock::time_point today = std::chrono::system_clock::now();

    // 转换为time_t时间类型
    time_t tm = std::chrono::system_clock::to_time_t(today);
    std::cout << "今天的日期是:    " << std::ctime(&tm);

    time_t tm1 = std::chrono::system_clock::to_time_t(today + day);
    std::cout << "明天的日期是:    " << std::ctime(&tm1);

    time_t tm2 = std::chrono::system_clock::to_time_t(epoch);
    std::cout << "新纪元时间:      " << std::ctime(&tm2);

    time_t tm3 = std::chrono::system_clock::to_time_t(ppt);
    std::cout << "新纪元时间+1天:  " << std::ctime(&tm3);

    time_t tm4 = std::chrono::system_clock::to_time_t(t);
    std::cout << "新纪元时间+10天: " << std::ctime(&tm4);

    // ========================================================
    // high_resolution_clock 提供的时钟精度比 system_clock 要高
    // 获取开始时间点
    std::chrono::steady_clock::time_point          start      = std::chrono::steady_clock::now();
    std::chrono::high_resolution_clock::time_point start_high = std::chrono::high_resolution_clock::now();
    // 执行业务流程
    std::cout << "print 1000 stars ...." << std::endl;
    for (int i = 0; i < 1000; ++i)
    {
        std::cout << "*";
    }
    std::cout << std::endl;

    // 获取结束时间点
    std::chrono::steady_clock::time_point          last      = std::chrono::steady_clock::now();
    std::chrono::high_resolution_clock::time_point last_high = std::chrono::high_resolution_clock::now();

    // 计算差值
    auto dt       = last - start;
    auto dt_hight = last_high - start_high;
    std::cout << "总共耗时: " << dt.count() << " 纳秒\n";
    std::cout << "总共耗时: " << dt_hight.count() << " 纳秒\n";

    // =================================
    auto time_1 = std::chrono::steady_clock::now();
    func_tasks();
    auto time_2 = std::chrono::steady_clock::now();

    // 整数时长：时钟周期纳秒转毫秒，要求 duration_cast
    auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_2 - time_1);

    // 小数时长：不要求 duration_cast
    std::chrono::duration<double, std::ratio<1, 1000>> fp_ms = time_2 - time_1;

    std::cout << "f() took " << fp_ms.count() << " ms, " << "or " << int_ms.count() << " whole milliseconds\n";

    // ==========================================================================
    // 注意事项:关于时间点的转换如果没有没有精度的损失可以直接进行隐式类型转换,
    // 如果会损失精度只能通过显示类型转换，也就是调用time_point_cast函数来完成该操作
    TimePoint<Sec> time_point_sec(Sec(6));
    // 无精度损失, 可以进行隐式类型转换
    TimePoint<Ms>  time_point_ms(time_point_sec);
    print_ms(time_point_ms); // 6000 ms

    time_point_ms = TimePoint<Ms>(Ms(6789));
    // error，会损失精度，不允许进行隐式的类型转换
    // TimePoint<Sec> sec(time_point_ms);

    // 显示类型转换,会损失精度。6789 truncated to 6000
    time_point_sec = std::chrono::time_point_cast<Sec>(time_point_ms);
    print_ms(time_point_sec); // 6000 ms

    return 0;
}
