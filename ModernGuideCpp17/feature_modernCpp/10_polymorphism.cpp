/**
 * @file 10_polymorphism.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

// https://cplusplus.com/doc/tutorial/polymorphism/
class Polygon
{
public:
    void setValues(int width, int height)
    {
        this->width  = width;
        this->height = height;
    }

    virtual void getArea() = 0;

    // private:
protected:
    unsigned width;
    unsigned height;
};

class Rectangle : public Polygon
{
public:
    void getArea() override
    {
        std::cout << "[Rectangle Area from Virtual]: " << width * height << '\n';
    }

    void area()
    {
        // return width * height;
        std::cout << "[Rectangle Area]: " << width * height << '\n';
    }
};

class Triangle : public Polygon
{
public:
    void getArea() override
    {
        std::cout << "[Triangle Area from Virtual]: " << width * height / 2 << '\n';
    }

    void area()
    {
        std::cout << "[Triangle Area]: " << width * height / 2 << '\n';
        // return width * height / 2;
    }
};

// ------------------------
int main(int argc, char **argv)
{
    Rectangle rect;
    Triangle  trgl;
    Polygon  *ppoly1 = &rect;
    Polygon  *ppoly2 = &trgl;

    ppoly1->setValues(4, 9);
    ppoly1->getArea();

    ppoly2->setValues(4, 9);
    ppoly2->getArea();

    rect.area();
    trgl.area();

    return 0;
}
