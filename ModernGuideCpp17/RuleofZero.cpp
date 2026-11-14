/**
 * @file RuleofZero.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-11-13
 * 
 * @copyright Copyright (c) 2026
 * 
 */

/**
 * @brief 三/五/零 法则: 避免浅拷贝与双重释放
 * 
 * 核心判断: 类一旦开始自己管理资源(裸指针、堆内存), 那么它的
 *     拷贝、移动、析构 就必须成套考虑
 * 要不按照法则将 特殊成员函数 配齐, 要不采用 RAII 成员将资源托管出去;
 * 如果只写析构, 放任编译器生成默认拷贝, 是双重释放和浅拷贝悬挂的头号来源.
 * 
 * Rule of Three 三法则 (std==C++03)
 * ---> 析构、拷贝构造、拷贝赋值 三者都要写, 针对裸指针的深拷贝.
 * 
 * Rule of Five 五法则 (std=C++11)
 * ---> 移动语义加入后, 再加入移动构造、移动赋值, 五个函数配套.
 * 
 * Rule of Zero 零法则 (std=Modern C++)
 * ---> 成员全用 RAII 容器, 五个特殊函数全靠编译器默认生成.
 * 
 */

#include <iostream>
#include <vector>

class Buffer
{
public:
    explicit Buffer(size_t n)
        : mp_data(new char[n])
        , m_size(n)
    {
    }

    ~Buffer()
    {
        if (mp_data)
        {
            delete[] mp_data;
            mp_data = nullptr;
        }
    }

    // copy-constructor
    Buffer(const Buffer &obj)
        : mp_data(new char[obj.m_size])
        , m_size(obj.m_size)
    {
        std::copy(obj.mp_data, obj.mp_data + m_size, mp_data);
    }

    // copy-assignment
    Buffer &operator=(const Buffer &obj)
    {
        if (this != &obj)
        {
            Buffer tmp(obj);
            std::swap(*this, tmp);
        }

        return *this;
    }

    // move-constructor
    Buffer(Buffer &&obj) noexcept
        : mp_data(obj.mp_data)
        , m_size(obj.m_size)
    {
        obj.mp_data = nullptr;
        obj.m_size  = 0;
    }

    // move-assignment
    Buffer &operator=(Buffer &&obj) noexcept
    {
        if (this != &obj)
        {
            delete[] mp_data;
            mp_data = obj.mp_data;
            m_size  = obj.m_size;

            obj.mp_data = nullptr;
            obj.m_size  = 0;
        }

        return *this;
    }

private:
    char  *mp_data;
    size_t m_size;
};

class BufferModern
{
public:
    explicit BufferModern(size_t n)
        : m_data(n)
    {
    }

    ~BufferModern() = default;

    // copy-constructor
    BufferModern(const BufferModern &obj) = default; // delete
    // copy-assignment
    BufferModern &operator=(const BufferModern &obj) = default; // delete
    // move-constructor
    BufferModern(BufferModern &&obj) = default; // delete
    // move-assignment
    BufferModern &operator=(BufferModern &&obj) = default; // delete

private:
    // RAII container
    std::vector<char> m_data;
};

//  -----------------------------------
int main(int argc, const char *argv[])
{
    std::cout << "Rule of Three (std==C++03)\n";
    std::cout << "Rule of Five (std=C++11)\n";
    std::cout << "Rule of Zero (std=Modern C++)\n";

    return 0;
}

// g++ RuleofZero.cpp -std=c++23