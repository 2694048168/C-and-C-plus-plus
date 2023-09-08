/**
 * @file weak_ptr.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-08
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __WEAK_PTR_HPP__
#define __WEAK_PTR_HPP__

#include "utility/shared_ptr.hpp"

namespace WeiLi { namespace utility {

// 解决 shared_ptr 存在的循环引用的问题，复杂的业务逻辑场景
template<typename T>
class WeakPtr
{
public:
    WeakPtr()
        : m_data(nullptr)
        , m_count(nullptr)
    {
    }

    WeakPtr(const SharedPtr<T> &sp)
        : m_data(sp.m_data)
        , m_count(sp.m_count)
    {
    }

    // copy constructor
    WeakPtr(const WeakPtr<T> &other)
        : m_data(other.m_data)
        , m_count(other.m_count)
    {
    }

    // move constructor
    WeakPtr(const WeakPtr<T> &&other)
        : m_data(other.m_data)
        , m_count(other.m_count)
    {
        other.m_data  = nullptr;
        other.m_count = nullptr;
    }

    ~WeakPtr()
    {
        m_data  = nullptr;
        m_count = nullptr;
    }

    void reset()
    {
        m_data  = nullptr;
        m_count = nullptr;
    }

    bool expired() const
    {
        return !m_count || (*m_data) <= 0;
    }

    SharedPtr<T> lock() const
    {
        if (expired())
        {
            return SharedPtr<T>();
        }
        SharedPtr<T> sp;
        sp.m_data  = m_data;
        sp.m_count = m_count;
        if (m_count != nullptr)
        {
            (*m_count)++;
        }
        return sp;
    }

    int use_count() const

    {
        if (m_data == nullptr)
        {
            return 0;
        }
        return *m_count;
    }

    void swap(WeakPtr<T> &other)
    {
        auto data     = other.m_data;
        auto count    = other.m_count;
        other.m_data  = m_data;
        other.m_count = m_count;
        m_data        = data;
        m_count       = count;
    }

    // copy assignment operator
    // copy assignment operator
    WeakPtr<T> &operator=(const WeakPtr<T> &other)
    {
        if (this == &other)
        {
            return *this;
        }
        m_data  = other.m_data;
        m_count = other.m_count;
        return *this;
    }

    WeakPtr<T> &operator=(const SharedPtr<T> &sp)
    {
        m_data  = sp.m_data;
        m_count = sp.m_count;
        return *this;
    }

    // move assignment operator
    WeakPtr<T> &operator=(const WeakPtr<T> &&other)
    {
        if (this == &other)
        {
            return *this;
        }
        m_data        = other.m_data;
        m_count       = other.m_count;
        other.m_count = nullptr;
        other.m_data  = nullptr;
        return *this;
    }

private:
    T   *m_data;
    int *m_count;
};
}} // namespace WeiLi::utility

#endif // !__WEAK_PTR_HPP__