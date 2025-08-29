/**
 * @file 33_thread_synchronization_atomic.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** C++11提供了一个原子类型std::atomic<T>,
 * 通过这个原子类型管理的内部变量就可以称之为原子变量,
 * *可以给原子类型指定bool、char、int、long、指针等类型作为模板参数(不支持浮点类型和复合类型).
 * ======================================================================
 * ?原子指的是一系列不可被CPU上下文交换的机器指令,这些指令组合在一起就形成了原子操作.
 * 在多核CPU下,当某个CPU核心开始运行原子操作时,会先暂停其它CPU内核对内存的操作,以保证原子操作不会被其它CPU内核所干扰.
 * 由于原子操作是通过指令提供的支持,因此它的性能相比锁和消息传递会好很多.
 * 相比较于锁而言,原子类型不需要开发者处理加锁和释放锁的问题,同时支持修改,读取等操作,还具备较高的并发性能,几乎所有的语言都支持原子类型.
 * *可以看出原子类型是无锁类型,但是无锁不代表无需等待,因为原子类型内部使用了CAS循环,
 * 当大量的冲突发生时,该等待还是得等待！但是总归比锁要好.
 * C++11内置了整形的原子变量,可以更方便的使用原子变量了.
 * 在多线程操作中, 使用原子变量之后就不需要再使用互斥量来保护该变量了, 用起来更简洁.
 * !因为对原子变量进行的操作只能是一个原子操作（atomic operation）,
 * 原子操作指的是不会被线程调度机制打断的操作,这种操作一旦开始,就一直运行到结束,中间不会有任何的上下文切换.
 * 多线程同时访问共享资源造成数据混乱的原因就是因为CPU的上下文切换导致的,使用原子变量解决了这个问题,因此互斥锁的使用也就不再需要了.
 * *CAS全称是Compare and swap, 它通过一条指令读取指定的内存地址,
 * *然后判断其中的值是否等于给定的前置值，如果相等，则将其修改为新的值.
 * ======================================================================
 * 1. atomic 通过定义可得知:在使用这个模板类的时候，一定要指定模板类型;
 * 2. atomic() noexcept = default; 构造函数--->默认无参构造函数;
 * 3. constexpr atomic( T desired ) noexcept; 构造函数--->使用 desired 初始化原子变量的值;
 * 4. atomic( const atomic& ) = delete; 构造函数--->使用=delete显示删除拷贝构造函数,不允许进行对象之间的拷贝;
 * 5. 公共成员函数(operator=): 原子类型在类内部重载了=操作符，并且不允许在类的外部使用 =进行对象的拷贝;
 * 6. 公共成员函数(store): 原子地以 desired 替换当前值, 按照 order 的值影响内存;
 * 7. 公共成员函数(load): 原子地加载并返回原子变量的当前值, 按照 order 的值影响内存,直接访问原子对象也可以得到原子变量的当前值;
 * 8. 特化成员函数, 复合赋值运算符重载(+=, -=, /=, *=, ....)
 * 9. 内存顺序约束, API 函数可以看出,在调用 atomic类提供的 API 函数的时候,需要指定原子顺序.
 * *在C++11提供的 API 中使用枚举用作执行原子操作的函数的实参,以指定如何同步不同线程上的其他操作.
// ======================================================
typedef enum memory_order {
    memory_order_relaxed,   // relaxed
    memory_order_consume,   // consume
    memory_order_acquire,   // acquire
    memory_order_release,   // release
    memory_order_acq_rel,   // acquire/release
    memory_order_seq_cst    // sequentially consistent
} memory_order;
// ======================================================
 * *memory_order_relaxed,这是最宽松的规则,它对编译器和CPU不做任何限制,可以乱序;
 * *memory_order_release 释放,设定内存屏障(Memory barrier),保证它之前的操作永远在它之前,但是它后面的操作可能被重排到它前面;
 * *memory_order_acquire 获取,设定内存屏障,保证在它之后的访问永远在它之后,但是它之前的操作却有可能被重排到它后面,往往和Release在不同线程中联合使用;
 * *memory_order_consume, 改进版的memory_order_acquire,开销更小;
 * *memory_order_acq_rel, 它是Acquire 和 Release 的结合,同时拥有它们俩提供的保证.
 *   比如要对一个 atomic 自增 1,同时希望该操作之前和之后的读取或写入操作不会被重新排序;
 * *memory_order_seq_cst 顺序一致性, memory_order_seq_cst 就像是memory_order_acq_rel的加强版,
 *  它不管原子操作是属于读取还是写入的操作,只要某个线程有用到memory_order_seq_cst 的原子操作,
 *  线程中该memory_order_seq_cst 操作前的数据操作绝对不会被重新排在该memory_order_seq_cst 操作之后,
 *  且该memory_order_seq_cst 操作后的数据操作也绝对不会被重新排在memory_order_seq_cst 操作前.
 * 
 * 在C++20版本中添加了新的功能函数,可以通过原子类型来阻塞线程,和条件变量中的等待/通知函数是一样的
 * ?1. wait(C++20)	阻塞线程直至被提醒且原子值更改;
 * ?2. notify_one(C++20) 通知（唤醒）至少一个在原子对象上阻塞的线程;
 * ?3. notify_all(C++20) 通知（唤醒）所有在原子对象上阻塞的线程;
 * 
 * 类型别名
 * ?atomic_bool(C++11)	std::atomic<bool>
 * ?atomic_int(C++11)	std::atomic<int>
 * ?atomic_size_t(C++11) std::atomic<std::size_t>
 * 
 * !原子类型atomic<T> 可以封装原始数据最终得到一个原子变量对象,
 * *操作原子对象能够得到和操作原始数据一样的效果,
 * *当然也可以通过store()和load()来读写原子对象内部的原始数据.
 * 
 */

#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

// 多线程交替数数的计数器，我们使用互斥锁和原子变量的方式分别进行实现
class CounterMutex
{
public:
    void increment()
    {
        for (int i = 0; i < 100; ++i)
        {
            std::lock_guard<std::mutex> locker(m_mutex);
            m_value++;
            // std::cout << "increment number: " << m_value << ", theadID: " << std::this_thread::get_id() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    void decrement()
    {
        for (int i = 0; i < 100; ++i)
        {
            std::lock_guard<std::mutex> locker(m_mutex);
            m_value--;
            // std::cout << "decrement number: " << m_value << ", theadID: " << std::this_thread::get_id() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

private:
    int        m_value = 0;
    std::mutex m_mutex;
};

class CounterAtomic
{
public:
    void increment()
    {
        for (int i = 0; i < 100; ++i)
        {
            m_value++;
            // std::cout << "increment number: " << m_value << ", theadID: " << std::this_thread::get_id() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    void decrement()
    {
        for (int i = 0; i < 100; ++i)
        {
            m_value--;
            // std::cout << "decrement number: " << m_value << ", theadID: " << std::this_thread::get_id() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

private:
    // atomic<int> == atomic_int
    std::atomic_int m_value = 0;
};

struct TimeAutoLog
{
    TimeAutoLog(const char *msg)
        : m_msg{msg}
    {
        m_start_time = std::chrono::high_resolution_clock::now();
    }

    ~TimeAutoLog()
    {
        m_end_time = std::chrono::high_resolution_clock::now();

        m_time_comsume = std::chrono::duration_cast<std::chrono::milliseconds>(m_end_time - m_start_time).count();

        std::cout << m_msg << " The time: " << m_time_comsume << " ms\n\n";
    }

private:
    std::chrono::high_resolution_clock::time_point m_start_time;
    std::chrono::high_resolution_clock::time_point m_end_time;

    double      m_time_comsume; // ms
    const char *m_msg;
};

// ----------------------------------
int main(int argc, const char **argv)
{
    {
        TimeAutoLog  time_auto_log("CounterMutex");
        CounterMutex obj_mutex;

        auto increment_mutex = std::bind(&CounterMutex::increment, &obj_mutex);
        auto decrement_mutex = std::bind(&CounterMutex::decrement, &obj_mutex);

        std::thread t1(increment_mutex);
        std::thread t2(decrement_mutex);

        t1.join();
        t2.join();
    }

    {
        TimeAutoLog   time_auto_log("CounterAtomic");
        CounterAtomic obj_atomic;

        auto increment_atomic = std::bind(&CounterAtomic::increment, &obj_atomic);
        auto decrement_atomic = std::bind(&CounterAtomic::decrement, &obj_atomic);

        std::thread t1(increment_atomic);
        std::thread t2(decrement_atomic);

        t1.join();
        t2.join();
    }

    // CounterMutex The time: 3273 ms
    // CounterAtomic The time: 1586 ms
    // ?从记录的时间耗时推断可知, 使用原子变量atomic比加锁效率高

    return 0;
}
