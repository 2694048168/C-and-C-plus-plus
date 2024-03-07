/**
 * @file 13_polymorphic.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 现代C++编程学习之多态
 * @version 0.1
 * @date 2024-03-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <string>
#include <string_view>

class Shape
{
public:
    Shape()
    {
        std::cout << "Shape 构造函数被调用\n";
    }

    virtual ~Shape()
    {
        std::cout << "Shape 析构函数被调用\n";
    }

    virtual inline void setName(const std::string_view &name) = 0;

    virtual inline std::string getName() const = 0;

    virtual inline void setWidthHeight(const float &width, const float &height) = 0;

    virtual inline float computeArea() = 0;

    virtual inline void printInfo() = 0;
};

class Triangle : public Shape
{
public:
    Triangle()
    {
        std::cout << "Triangle 构造函数被调用\n";
    }

    ~Triangle()
    {
        std::cout << "Triangle 析构函数被调用\n";
    }

    inline void setName(const std::string_view &name) override
    {
        m_name = name;
    }

    inline std::string getName() const override
    {
        return m_name;
    }

    inline void setWidthHeight(const float &width, const float &height) override
    {
        m_width  = width;
        m_height = height;
    }

    inline float computeArea() override
    {
        m_area = m_width * m_height * 0.5;
        return m_area;
    }

    inline void printInfo() override
    {
        std::cout << "The Triangle named: " << m_name;
        std::cout << " with width: " << m_width;
        std::cout << " and height: " << m_height;
        std::cout << " and the area is: " << m_area << std::endl;
    }

private:
    std::string m_name;
    float       m_width;
    float       m_height;
    float       m_area;
};

class Rectangle : public Shape
{
public:
    Rectangle()
    {
        std::cout << "Rectangle 构造函数被调用\n";
    }

    ~Rectangle()
    {
        std::cout << "Rectangle 析构函数被调用\n";
    }

    inline void setName(const std::string_view &name) override
    {
        m_name = name;
    }

    inline std::string getName() const override
    {
        return m_name;
    }

    inline void setWidthHeight(const float &width, const float &height) override
    {
        m_width  = width;
        m_height = height;
    }

    inline float computeArea() override
    {
        m_area = m_width * m_height;
        return m_area;
    }

    inline void printInfo() override
    {
        std::cout << "The Rectangle named: " << m_name;
        std::cout << " with width: " << m_width;
        std::cout << " and height: " << m_height;
        std::cout << " and the area is: " << m_area << std::endl;
    }

private:
    std::string m_name;
    float       m_width;
    float       m_height;
    float       m_area;
};

// ====================================
int main(int argc, const char **argv)
{
    std::cout << "============ 三角形 ============\n";
    Shape *p_triangle = new Triangle;
    p_triangle->setName("三角形");
    p_triangle->setWidthHeight(12.0f, 6.f);
    p_triangle->computeArea();
    p_triangle->printInfo();

    std::cout << "============ 矩形 ============\n";
    Shape *p_rectangle = new Rectangle;
    p_rectangle->setName("三角形");
    p_rectangle->setWidthHeight(12.0f, 6.f);
    p_rectangle->computeArea();
    p_rectangle->printInfo();

    std::cout << "\n=========== 析构函数, 释放申请的内存 ===========\n";
    if (p_triangle != nullptr)
    {
        delete p_triangle;
        p_triangle = nullptr;
    }
    if (p_rectangle != nullptr)
    {
        delete p_rectangle;
        p_rectangle = nullptr;
    }

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\13_polymorphic.cpp -std=c++23
// g++ .\13_polymorphic.cpp -std=c++23
