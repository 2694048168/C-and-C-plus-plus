/**
 * @file 04_mutex_shared_data.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-06-15
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <algorithm>
#include <iostream>
#include <list>
#include <mutex>
#include <thread>
#include <vector>

// shared-memory global data
std::list<int> g_some_list;
std::mutex     g_some_mutex;

void add_to_list(int value)
{
    // C++标准库为互斥量提供了RAII模板类 std::lock_guard,
    // 在构造时就能提供已锁的互斥量，并在析构时进行解锁，从而保证了互斥量能被正确解锁。
    // std::mutex.lock() || std::mutex.unlock()
    // std::lock_guard<std::mutex> guard(some_mutex);

    // C++17 添加了一个新特性，称为模板类参数推导
    // std::lock_guard guard(some_mutex);

    // C++17中的一种加强版数据保护机制: std::scoped_lock
    // std::scoped_lock<std::mutex> guard(some_mutex);
    std::scoped_lock guard(g_some_mutex);

    g_some_list.push_back(value);
}

bool list_contains(int value_to_find)
{
    // std::lock_guard<std::mutex> guard(some_mutex);
    // std::lock_guard guard(some_mutex);

    // std::scoped_lock<std::mutex> guard(some_mutex);
    std::scoped_lock guard(g_some_mutex);

    return std::find(g_some_list.begin(), g_some_list.end(), value_to_find) != g_some_list.end();
}

/* 某些情况下使用全局变量没问题，但大多数情况下，互斥量通常会与需要保护的数据放在同一类中,
    而不是定义成全局变量。这是面向对象设计的准则：将其放在一个类中，就可让他们联系在一起，
    也可对类的功能进行封装，并进行数据保护。这种情况下，
    函数add_to_list和list_contains可以作为这个类的成员函数。互斥量和需要保护的数据，
    在类中都定义为private成员，这会让代码更清晰，并且方便了解什么时候对互斥量上锁。
    所有成员函数都会在调用时对数据上锁，结束时对数据解锁，这就保证了访问时数据不变量的状态稳定。
    当然，也不是总能那么理想：当其中一个成员函数返回的是保护数据的指针或引用时，也会破坏数据。
    具有访问能力的指针或引用可以访问(并可能修改)保护数据，而不会被互斥锁限制。
    这就需要对接口谨慎设计，要确保互斥量能锁住数据访问，并且不留后门。
------------------------------------------------------------ */
class MutexSharedData
{
public:
    void add_to_list(int value);
    bool list_contains(int value_to_find);

private:
    std::list<int> m_some_list;
    std::mutex     m_some_mutex;
};

template<typename T>
void print_list_container(T &container, const char *msg)
{
    std::cout << msg << "[";
    for (const auto &element : container)
    {
        std::cout << element << ", ";
    }
    std::cout << "]" << std::endl;
}

/**
 * @brief 保护共享数据, 特别在多线程对该共享数据都有写操作的时候
 * C++ 中 Race Condition 情况很多种类型, 
 * 特别考虑到指针或引用可以对数据进行修改的情况下，情况很复杂, 需要仔细设计接口API
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::vector<std::thread> threads;

    const unsigned int NUM_THREAD = 16;
    for (int i = 0; i < NUM_THREAD; ++i)
    {
        threads.push_back(std::thread{add_to_list, i});
    }

    for (auto &t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }
    print_list_container(g_some_list, "the list elements: ");

    if (list_contains(6))
    {
        std::cout << "find the number 6 in the list container\n";
    }
    else
    {
        std::cout << "NOT find the number 6 in the list container\n";
    }

    if (list_contains(42))
    {
        std::cout << "find the number 42 in the list container\n";
    }
    else
    {
        std::cout << "NOT find the number 42 in the list container\n";
    }

    std::cout << "the main thread of program ending" << std::endl;

    return 0;
}