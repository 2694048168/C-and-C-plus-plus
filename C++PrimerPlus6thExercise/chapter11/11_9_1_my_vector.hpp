/**
 * @file 11_9_1_my_vector.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-07
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __MY_VECTOR_HPP__
#define __MY_VECTOR_HPP__

#include <iostream>

namespace VECTOR {

class Vector
{
public:
    enum Mode
    {
        RECT, /* RECT for rectangular, */
        POL   /* POL for Polar modes */
    };

private:
    double x;    /* horizontal value */
    double y;    /* vertical value */
    double mag;  /* length of vector */
    double ang;  /* direction of vector in degrees */
    Mode   mode; /* RECT or POL */

private:
    void set_x();
    void set_y();
    void set_mag();
    void set_ang();

public:
    // default constructor
    Vector();
    
    Vector(double n1, double n2, Mode form = RECT);

    void reset(double n1, double n2, Mode form = RECT);
    ~Vector();

    double get_x_val() const;
    double get_y_val() const;
    double get_mag_val() const;
    double get_ang_val() const;

    void polar_mode(); /* set mode to POL */
    void rect_mode();  /* set mode to RECT */

    // operator overloading
    Vector operator+(const Vector &b) const;
    Vector operator-(const Vector &b) const;
    Vector operator-() const;
    Vector operator*(double n) const;

    // friends
    friend Vector operator*(double n, const Vector &a);

    // 针对输出流以友元函数方式重载, 便于 Vector 类型直接使用 C++ 的输入和输出
    friend std::ostream &operator<<(std::ostream &os, const Vector &v);
};

} // namespace VECTOR

#endif // !__MY_VECTOR_HPP__