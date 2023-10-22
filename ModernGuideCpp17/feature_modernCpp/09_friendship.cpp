/**
 * @file 09_friendship.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

// https://cplusplus.com/doc/tutorial/inheritance/
class Square
{
    friend class Rectangle;

private:
    int side;

public:
    Square(int a)
        : side(a)
    {
    }
};

class Rectangle
{
public:
    Rectangle() = default;

    Rectangle(int x, int y)
        : width(x)
        , height(y)
    {
    }

    int getArea()
    {
        return width * height;
    }

    // friend function
    friend Rectangle duplicate(const Rectangle &);

    void convert(Square a);

private:
    int width;
    int height;
};

Rectangle duplicate(const Rectangle &rect)
{
    Rectangle rectangle;

    rectangle.width  = rect.width * 2;
    rectangle.height = rect.height * 2;

    return rectangle;
}

void Rectangle::convert(Square a)
{
    width  = a.side;
    height = a.side;
}

// -------------Inheritance between classes
class Polygon
{
protected:
    int width;
    int height;

public:
    void setValues(int a, int b)
    {
        width  = a;
        height = b;
    }
};

class Rectangle2 : public Polygon
{
public:
    int area()
    {
        return width * height;
    }
};

class Triangle : public Polygon
{
public:
    int area()
    {
        return width * height / 2;
    }
};

// Multiple inheritance
class PolygonMulti
{
protected:
    int width;
    int height;

public:
    PolygonMulti(int a, int b)
        : width(a)
        , height(b)
    {
    }
};

class Output
{
public:
    static void print(int i);
};

void Output::print(int i)
{
    std::cout << "[the area]: " << i << '\n';
}

class RectangleMulti
    : public PolygonMulti
    , public Output
{
public:
    RectangleMulti(int a, int b)
        : PolygonMulti(a, b)
    {
    }

    int area()
    {
        return width * height;
    }
};

class TriangleMulti
    : public PolygonMulti
    , public Output
{
public:
    TriangleMulti(int a, int b)
        : PolygonMulti(a, b)
    {
    }

    int area()
    {
        return width * height / 2;
    }
};

// -----------------------------
int main(int argc, char **argv)

{
    // ======== friend function ========
    Rectangle foo;
    Rectangle bar(2, 3);

    foo = duplicate(bar);
    std::cout << "The area of Rectangle is: " << foo.getArea() << '\n';

    // ======== friend class ========
    Rectangle rect;
    Square    sqr(4);

    rect.convert(sqr);
    std::cout << "The area of Square is: " << rect.getArea() << '\n';

    // ======== Inheritance between class ========
    Rectangle2 rect2;
    Triangle   triangle;
    rect2.setValues(4, 5);
    triangle.setValues(4, 5);

    std::cout << rect2.area() << '\n';
    std::cout << triangle.area() << '\n';

    // ======== Multiple inheritance between class ========
    RectangleMulti rect_multi(8, 9);
    TriangleMulti  triangle_multi(8, 9);

    rect_multi.print(rect_multi.area());
    TriangleMulti::print(triangle_multi.area());

    return 0;
}
