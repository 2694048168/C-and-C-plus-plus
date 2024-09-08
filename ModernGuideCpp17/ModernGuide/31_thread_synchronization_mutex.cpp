/**
 * @file 31_thread_synchronization_mutex.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 进行多线程编程,如果多个线程需要对同一块内存进行操作,比如：同时读、同时写、同时读写;
 * 对于后两种情况来说,如果不做任何的人为干涉就会出现各种各样的错误数据.
 * *这是因为线程在运行的时候需要先得到CPU时间片,时间片用完之后需要放弃已获得的CPU资源,
 * 就这样线程频繁地在就绪态和运行态之间切换,更复杂一点还可以在就绪态、运行态、挂起态之间切换,
 * 这样就会导致线程的执行顺序(不一定在同一个CPU core中)并不是有序的,而是随机的混乱的.
 * ?同时由于多核CPU架构的cache模式, 数据在L1/L2cache 以及L3cache和内存中是否多线程同步？
 * 
 * 解决多线程数据混乱的方案就是进行线程同步, 最常用的就是互斥锁,在C++11中一共提供了四种互斥锁:
 * 互斥锁在有些资料中也被称之为互斥量。
 * *1. std::mutex: 独占的互斥锁,不能递归使用;
 * *2. std::timed_mutex: 带超时的独占互斥锁,不能递归使用;
 * *3. std::recursive_mutex: 递归互斥锁,不带超时功能;
 * *4. std::recursive_timed_mutex: 带超时的递归互斥锁;
 * 
 * 1.1 成员函数
 * *lock()函数用于给临界区加锁,并且只能有一个线程获得锁的所有权,它有阻塞线程的作用;
 * 独占互斥锁对象有两种状态: 锁定和未锁定. 如果互斥锁是打开的, 调用lock()函数的线程会得到互斥锁的所有权,
 * 并将其上锁, 其它线程再调用该函数的时候由于得不到互斥锁的所有权, 就会被lock()函数阻塞.
 * 当拥有互斥锁所有权的线程将互斥锁解锁, 此时被lock()阻塞的线程解除阻塞, 
 * 抢到互斥锁所有权的线程加锁并继续运行, 没抢到互斥锁所有权的线程继续阻塞.
 * ?除了使用lock()还可以使用try_lock()获取互斥锁的所有权并对互斥锁加锁.
 * ?二者的区别在于try_lock()不会阻塞线程，lock()会阻塞线程.
 * 1. try_lock(), 如果互斥锁是未锁定状态，得到了互斥锁所有权并加锁成功，函数返回true;
 * 2. try_lock(), 如果互斥锁是锁定状态，无法得到互斥锁所有权加锁失败，函数返回false;
 * 当互斥锁被锁定之后可以通过unlock()进行解锁,
 * 但是需要注意的是只有拥有互斥锁所有权的线程也就是对互斥锁上锁的线程才能将其解锁,其它线程是没有权限做这件事情的.
 * 
 * =====使用互斥锁进行线程同步的大致思路差不多就能搞清楚了,主要分为以下几步:
 * 1. 找到多个线程操作的共享资源(全局变量、堆内存、类成员变量等), 也可以称之为临界资源;
 * 2. 找到和共享资源有关的上下文代码, 也就是临界区;
 * 3. 在临界区的上边调用互斥锁类的lock()方法;
 * 4. 在临界区的下边调用互斥锁的unlock()方法;
 * *线程同步的目的是让多线程按照顺序依次执行临界区代码,这样做线程对共享资源的访问就从并行访问变为了线性访问,
 * *访问效率降低了, 但是保证了数据的正确性; 尽可能的做到 lock-free op.
 * !当线程对互斥锁对象加锁,并且执行完临界区代码之后,一定要使用这个线程对互斥锁解锁,
 * !否则最终会造成线程的死锁. 死锁之后当前应用程序中的所有线程都会被阻塞,并且阻塞无法解除,应用程序也无法继续运行.
 * 
 * 2. std::lock_guard
 * lock_guard是C++11新增的一个模板类,使用这个类(RAII),可以简化互斥锁lock()和unlock()的写法,同时也更安全.
 * 
 * 3. std::recursive_mutex
 * *递归互斥锁std::recursive_mutex允许同一线程多次获得互斥锁,
 * *可以用来解决同一线程需要多次获取互斥量时死锁的问题.
 * 虽然递归互斥锁可以解决同一个互斥锁频繁获取互斥锁资源的问题,但是还是建议少用,主要原因如下:
 * 1. 使用递归互斥锁的场景往往都是可以简化的,使用递归互斥锁很容易放纵复杂逻辑的产生,从而导致bug的产生;
 * 2. 递归互斥锁比非递归互斥锁效率要低一些;
 * 3. 递归互斥锁虽然允许同一个线程多次获得同一个互斥锁的所有权,但最大次数并未具体说明,一旦超过一定的次数,就会抛出std::system错误;
 * 
 * 4. std::timed_mutex
 * ?std::timed_mutex是超时独占互斥锁,主要是在获取互斥锁资源时增加了超时等待功能,
 * 因为不知道获取锁资源需要等待多长时间,为了保证不一直等待下去,设置了一个超时时长,超时后线程就可以解除阻塞去做其他事情了.
 * std::timed_mutex比std::_mutex多了两个成员函数：try_lock_for()和try_lock_until():
 * 1. try_lock_for函数是当线程获取不到互斥锁资源的时候，让线程阻塞一定的时间长度;
 * 2. try_lock_until函数是当线程获取不到互斥锁资源的时候，让线程阻塞到某一个指定的时间点;
 * 关于两个函数的返回值:当得到互斥锁的所有权之后,函数会马上解除阻塞,返回true;
 * ?如果阻塞的时长用完或者到达指定的时间点之后,函数也会解除阻塞,返回false.
 * 
 */

#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

// 在所有线程的任务函数执行完毕之前,互斥锁对象是不能被析构的,一定要在程序中保证这个对象的可用性。
// 互斥锁的个数和共享资源的个数相等,也就是说每一个共享资源都应该对应一个互斥锁对象,互斥锁对象的个数和线程的个数没有关系
int        g_num = 0; // 为 g_num_mutex 所保护
std::mutex g_num_mutex;

void slow_increment(int id)
{
    for (int i = 0; i < 3; ++i)
    {
        g_num_mutex.lock();
        ++g_num;
        std::cout << id << " => " << g_num << std::endl;
        g_num_mutex.unlock();

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void slow_increment_guard_lock(int id)
{
    for (int i = 0; i < 3; ++i)
    {
        // std::lock_guard 构造函数构造对象时,会自动锁定互斥量,
        // 而在退出作用域后进行析构时就会自动解锁,从而保证了互斥量的正确操作,
        // 避免忘记unlock()操作而导致线程死锁.
        // std::lock_guard使用了RAII技术.

        // 这种方式也有弊端,可能整个for循环的体都被当做了临界区,
        // 多个线程是线性的执行临界区代码的,因此临界区越大程序效率越低,
        // 所以需要善用 {} 界定临界区, 灵巧使用RAII技术
        {
            // 使用哨兵锁管理互斥锁
            std::lock_guard<std::mutex> lock(g_num_mutex);
            ++g_num;
            std::cout << id << " => " << g_num << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

// ========= 递归互斥锁 =========
class Calculate
{
public:
    Calculate()
        : m_i(6)
    {
    }

    void mul(int x)
    {
        // std::lock_guard<std::mutex> locker(m_mutex);
        std::lock_guard<std::recursive_mutex> locker(m_mutex);
        m_i *= x;
        std::cout << "The value is: " << m_i << std::endl;
    }

    void div(int x)
    {
        // std::lock_guard<std::mutex> locker(m_mutex);
        std::lock_guard<std::recursive_mutex> locker(m_mutex);
        m_i /= x;
        std::cout << "The value is: " << m_i << std::endl;
    }

    void both(int x, int y)
    {
        // std::lock_guard<std::mutex> locker(m_mutex);
        std::lock_guard<std::recursive_mutex> locker(m_mutex);
        mul(x);
        div(y);
    }

private:
    int                  m_i;
    // 要解决这个死锁的问题,一个简单的办法就是使用递归互斥锁std::recursive_mutex
    // 它允许一个线程多次获得互斥锁的所有权.
    // std::mutex m_mutex;
    std::recursive_mutex m_mutex;
};

// ========= 超时机制的互斥锁 =========
std::timed_mutex g_mutex;

void work()
{
    std::chrono::seconds timeout(1);
    while (true)
    {
        // 通过阻塞一定的时长来争取得到互斥锁所有权
        if (g_mutex.try_lock_for(timeout))
        {
            std::cout << "当前线程ID: " << std::this_thread::get_id() << ", 得到互斥锁所有权, 解除阻塞...\n";
            // 模拟处理任务用了一定的时长
            std::this_thread::sleep_for(std::chrono::seconds(10));
            // 互斥锁解锁
            g_mutex.unlock();
            break;
        }
        else
        {
            std::cout << "当前线程ID: " << std::this_thread::get_id() << ", 超时没有得到互斥锁所有权, 解除阻塞...\n";
            // 模拟处理其他任务用了一定的时长
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
}

// -------------------------------------
int main(int argc, const char **argv)
{
    std::thread t1(slow_increment, 0);
    std::thread t2(slow_increment, 1);
    t1.join();
    t2.join();

    std::thread task1(slow_increment_guard_lock, 2);
    std::thread task2(slow_increment_guard_lock, 3);
    task1.join();
    task2.join();

    // =======================
    Calculate cal;
    // 调用之后, 程序就会发生死锁, 在both()中已经对互斥锁加锁了,
    // 继续调用mul()函数,已经得到互斥锁所有权的线程再次获取这个互斥锁的所有权就会造成死锁
    // (在C++中程序会异常退出,使用C库函数会导致这个互斥锁永远无法被解锁,最终阻塞所有的线程)
    cal.both(6, 3);

    std::thread thread1(work);
    std::thread thread2(work);

    thread1.join();
    thread2.join();

    return 0;
}
