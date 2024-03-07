/**
 * @file 15_classTemplate.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 现代C++编程学习之类模板
 * @version 0.1
 * @date 2024-03-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>

template<typename T>
class Array
{
private:
    int m_length{};
    T  *m_data{};

public:
    Array(int length)
    {
        assert(length > 0);
        m_data   = new T[length]{}; // allocated an array of objects of type T
        m_length = length;
    }

    Array(const Array &)            = delete;
    Array &operator=(const Array &) = delete;

    ~Array()
    {
        delete[] m_data;
    }

    void erase()
    {
        delete[] m_data;
        // We need to make sure we set m_data to 0 here, otherwise it will
        // be left pointing at deallocated memory!
        m_data   = nullptr;
        m_length = 0;
    }

    // templated operator[] function defined below
    T &operator[](int index); // now returns a T&

    int size() const
    {
        return m_length;
    }
};

// member functions defined outside the class need their own template declaration
template<typename T>
T &Array<T>::operator[](int index) // now returns a T&
{
    assert(index >= 0 && index < m_length);
    return m_data[index];
}

template<typename T>
void printContainer(T &container)
{
    for (size_t idx = 0; idx < container.size(); ++idx)
    {
        std::cout << container[idx] << ' ';
    }
    std::cout << std::endl;
}

// ====================================
int main(int argc, const char **argv)
{
    Array array_int = Array<int>(6);
    for (size_t idx = 0; idx < array_int.size(); ++idx)
    {
        array_int[idx] = idx + 12;
    }
    printContainer(array_int);

    Array array_char = Array<std::string>(6);
    for (size_t idx = 0; idx < array_int.size(); ++idx)
    {
        array_char[idx] = std::to_string(idx + 12);
    }
    printContainer(array_char);

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\15_classTemplate.cpp -std=c++23
// g++ .\15_classTemplate.cpp -std=c++23
