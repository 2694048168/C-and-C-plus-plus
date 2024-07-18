/**
 * @file singleton_mode.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 单例模式之线程安全的三种写法
 * @version 0.1
 * @date 2024-07-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <mutex>

// ======== 第一种方式, 互斥锁，双重锁定
class Singleton
{
private:
    Singleton() = default;

    static Singleton *m_instance;
    static std::mutex m_mutex;

public:
    static Singleton *getInstance()
    {
        // 双重检查锁定, 多线程执行时候存在内存一致性问题
        // 机器指令角度多条指定的顺序存在重新排序, 不一定按照我们预期的执行, 原子操作
        // 但是实际上 m_taskQ = new Singleton; 在执行过程中对应的机器指令可能会被重新排序
        // std::lock_guard<std::mutex> lock(m_mutex);
        if (nullptr == m_instance)
        {
            m_mutex.lock();
            if (m_instance == nullptr)
                m_instance = new Singleton(); // 不加锁存在线程不安全问题
            m_mutex.unlock();                 // 加锁后导致多线程访问该单例对象的效率低
        }

        return m_instance;
    }

    Singleton(const Singleton &)            = delete;
    Singleton &operator=(const Singleton &) = delete;
};

Singleton *Singleton::m_instance = nullptr;
std::mutex Singleton::m_mutex{};

// ======== 第二种方式, C11 std::once_flag and std::call_once
class Singleton
{
private:
    Singleton() = default;

    static Singleton    *m_instance;
    static std::once_fla m_initInstanceFlag;

    static void init()
    {
        m_instance = new Singleton;
    }

public:
    static Singleton *getInstance()
    {
        // C11 标准支持多线程安全
        std::call_once(m_initInstanceFlag, &Singleton::init);

        return m_instance;
    }

    Singleton(const Singleton &)            = delete;
    Singleton &operator=(const Singleton &) = delete;
};

Singleton *Singleton::m_instance = nullptr;
std::mutex Singleton::m_initInstanceFlag{};

// ======== 第三种方式, C11 静态局部变量的初始化是线程安全的
class Singleton
{
private:
    Singleton() = default;

public:
    static Singleton *getInstance()
    {
        // C11 静态局部变量的初始化是线程安全的,编译器支持
        static Singleton m_instance;

        return m_instance;
    }

    Singleton(const Singleton &)            = delete;
    Singleton &operator=(const Singleton &) = delete;
};

// ======== 单例模板：异递归模版模式
#include <iostream>

template<typename T>
class Singleton
{
protected:
    Singleton() = default;

public:
    Singleton(const Singleton &)            = delete;
    Singleton &operator=(const Singleton &) = delete;

    static T &getInstance()
    {
        static T m_instance;
        return m_instance;
    }
};

class Single : public Singleton<Single>
{
private:
    Single() = default;

    friend class Singleton<Single>;

public:
    void printInfo()
    {
        std::cout << "Single printInfo\n";
    }
};
