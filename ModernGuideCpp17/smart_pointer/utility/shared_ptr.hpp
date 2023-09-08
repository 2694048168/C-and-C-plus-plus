/**
 * @file shared_ptr.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-08
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __SHARED_PTR_HPP__
#define __SHARED_PTR_HPP__

namespace WeiLi { namespace utility {

template<typename T>
class SharedPtr
{
    template<typename Y>
    friend class WeakPtr;

public:
    SharedPtr()
        : m_data(nullptr)
        , m_count(nullptr)
    {
    }

    SharedPtr(T *data)
        : m_data(data)
    {
        m_count = new int(1);
    }

    // copy constructor
    SharedPtr(const SharedPtr<T> &other)
        : m_data(other.m_data)
        , m_count(other.m_count)
    {
        if (m_data != nullptr)
        {
            (*m_count)++;
        }
    }

    // move constructor
    SharedPtr(const SharedPtr<T> &&other)
        : m_data(other.m_data)
        , m_count(other.m_count)
    {
        other.m_data  = nullptr;
        other.m_count = nullptr;
    }

    ~SharedPtr()
    {
        if (m_data != nullptr)
        {
            (*m_count)--;
            if ((*m_count) <= 0)
            {
                delete m_data;
                m_data = nullptr;
                delete m_count;
                m_count = nullptr;
            }
        }
    }

    T *get() const
    {
        return m_data;
    }

    void reset(T *data = nullptr)
    {
        if (m_data == data)
        {
            return;
        }

        if (m_data == nullptr)
        {
            if (data != nullptr)
            {
                m_data  = data;
                m_count = new int(1);
            }
            return;
        }

        (*m_count)--;
        if ((*m_count) <= 0)
        {
            delete m_data;
            m_data = nullptr;
            delete m_count;
            m_count = nullptr;
        }
        m_data = data;
        if (data != nullptr)
        {
            m_count = new int(1);
        }
    }

    int use_count() const
    {
        if (m_data == nullptr)
        {
            return 0;
        }
        return *m_count;
    }

    bool unique() const
    {
        if (m_data == nullptr)
        {
            return false;
        }
        return *m_count == 1;
    }

    void swap(SharedPtr<T> &other)
    {
        auto data     = other.m_data;
        auto count    = other.m_count;
        other.m_data  = m_data;
        other.m_count = m_count;
        m_data        = data;
        m_count       = count;
    }

    T *operator->() const
    {
        return m_data;
    }

    T &operator*() const
    {
        return *m_data;
    }

    explicit operator bool() const noexcept
    {
        return m_data != nullptr;
    }

    // copy assignment operator
    SharedPtr<T> &operator=(const SharedPtr<T> &other)
    {
        if (this == &other)
        {
            return *this;
        }
        m_data  = other.m_data;
        m_count = other.m_count;
        (*m_count)++;
        return *this;
    }

    // move assignment operator
    SharedPtr<T> &operator=(const SharedPtr<T> &&other)
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

    // 考虑到 weak_ptr, 这里的引用计数不能是简单的一个引用计数;
    // TODO
    // 而是采用一个结构体的形式, 分别处理 shared_ptr and weak_ptr 的引用计数
    int *m_count;
};

}} // namespace WeiLi::utility

#endif // !__SHARED_PTR_HPP__
