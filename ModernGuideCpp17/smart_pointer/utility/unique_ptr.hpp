/**
 * @file unique_ptr.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-07
 * 
 * @copyright Copyright (c) 2023
 * 
 */

// C++ 提出的智能指针
#ifndef __UNIQUE_PTR_HPP__
#define __UNIQUE_PTR_HPP__

namespace WeiLi { namespace utility {

template<typename T>
class UniquePtr
{
public:
    UniquePtr()
        : m_data(nullptr)
    {
    }

    UniquePtr(T *data)
        : m_data(data)
    {
    }

    // 禁止拷贝构造函数, 避免对象的所有权在多个智能指针进行转移
    UniquePtr(const UniquePtr<T> &other) = delete;

    // 但是允许移动构造函数, ^_^
    UniquePtr(UniquePtr<T> &&other)
        : m_data(other.release())
    {
    }

    ~UniquePtr()
    {
        if (m_data != nullptr)
        {
            delete m_data;
            m_data = nullptr;
        }
    }

    // https://cplusplus.com/reference/memory/auto_ptr/
    T *get() const
    {
        return m_data;
    }

    T *release()
    {
        auto data = m_data;
        m_data    = nullptr;
        // 对象所有权的转移，对比 get 的区别
        return data;
    }

    void reset(T *data = nullptr)
    {
        if (m_data != data)
        {
            delete m_data;
            m_data = data;
        }
    }

    void swap(const UniquePtr<T> &other)
    {
        auto data    = other.m_data;
        other.m_data = this->m_data;
        this->m_data = data;
    }

    // 重载运算符，类似普通指针一样使用
    T *operator->()
    {
        return m_data;
    }

    T &operator*()
    {
        return *m_data;
    }

    // 赋值运算符也被禁止
    UniquePtr<T> &operator=(const UniquePtr<T> &other) = delete;

    // 移动赋值是允许的, ^_^
    UniquePtr<T> &operator=(UniquePtr<T> &&other)
    {
        if (this == &other)
        {
            return *this;
        }
        reset(other.release());
        return *this;
    }

    // 智能指针允许以数组下标形式, 需要重载
    T &operator[](int i) const
    {
        return m_data[i];
    }

    // 判断智能指针是否为空，指向有效对象
    explicit operator bool() const noexcept
    {
        return m_data != nullptr;
    }

private:
    T *m_data;
};

}} // namespace WeiLi::utility

#endif // !__UNIQUE_PTR_HPP__