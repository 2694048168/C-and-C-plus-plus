/**
 * @file auto_ptr.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-07
 * 
 * @copyright Copyright (c) 2023
 * 
 */

// C++98, 已经废弃了
#ifndef __AUTO_PTR_HPP__
#define __AUTO_PTR_HPP__

namespace WeiLi { namespace utility {

template<typename T>
class AutoPtr
{
public:
    AutoPtr()
        : m_data(nullptr)
    {
    }

    AutoPtr(T *data)
        : m_data(data)
    {
    }

    AutoPtr(AutoPtr<T> &other)
        : m_data(other.release())
    {
    }

    ~AutoPtr()
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

    // 重载运算符，类似普通指针一样使用
    T *operator->()
    {
        return m_data;
    }

    T &operator*()
    {
        return *m_data;
    }

    AutoPtr<T> &operator=(AutoPtr<T> other)
    {
        if (this == &other)
        {
            return *this;
        }
        // 对象所拥有的所有权转移，Rust 语言设计时考虑所有权
        // m_data = other.release();
        reset(other.release());
        return *this;
    }

private:
    T *m_data;
};

}} // namespace WeiLi::utility

#endif // !__AUTO_PTR_HPP__