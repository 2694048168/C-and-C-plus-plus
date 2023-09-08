/**
 * @file smart_ptr.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-07
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __SMART_PTR_HPP__
#define __SMART_PTR_HPP__

namespace WeiLi { namespace utility {

// 模板类，利用 RAII 技术实现对象的生命周期内资源有效性，
template<typename T>
class SmartPtr
{
public:
    SmartPtr()
        : m_data(nullptr)
    {
    }

    SmartPtr(T *data)
        : m_data(data)
    {
    }

    ~SmartPtr()
    {
        if (m_data != nullptr)
        {
            delete m_data;
            m_data = nullptr;
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

private:
    T *m_data;
};

}} // namespace WeiLi::utility

#endif // !__SMART_PTR_HPP__