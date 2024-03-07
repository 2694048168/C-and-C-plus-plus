/**
 * @file 12_inheritance.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 现代C++编程学习之继承
 * @version 0.1
 * @date 2024-03-06
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <string>

class Shape
{
public:
    Shape()  = default;
    ~Shape() = default;

    inline void setName(const std::string &name)
    {
        m_name = name;
    }

    inline std::string getName() const
    {
        return m_name;
    }

private:
    std::string m_name;
};

class Triangle : public Shape
{
public:
    Triangle()  = default;
    ~Triangle() = default;

    inline float computeArea()
    {
        return m_width * m_height * 0.5;
    }

    inline void setWidth(const float &width)
    {
        m_width = width;
    }

    inline void setHeight(const float &height)
    {
        m_height = height;
    }

private:
    float m_width;
    float m_height;
};

// ====================================
int main(int argc, const char **argv)
{
    Triangle triangle;
    triangle.setName("三角形");
    triangle.setWidth(12.0f);
    triangle.setHeight(4.f);

    std::cout << triangle.getName() << " 的面积是: " << triangle.computeArea() << std::endl;

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\12_inheritance.cpp -std=c++23
// g++ .\12_inheritance.cpp -std=c++23
