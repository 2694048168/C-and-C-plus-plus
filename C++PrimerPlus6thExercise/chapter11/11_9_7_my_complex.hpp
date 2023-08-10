/**
 * @file 11_9_7_my_complex.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-10
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __MY_COMPLEX_HPP__
#define __MY_COMPLEX_HPP__

#include <ostream>

class MyComplex
{
private:
    float real      = 0.f;
    float imaginary = 0.f;

public:
    MyComplex() = default;

    MyComplex(const float real, const float imaginary);

    ~MyComplex();

    // operator overloading
    MyComplex operator+(const MyComplex &c);
    MyComplex operator-(const MyComplex &c);
    MyComplex operator*(const float &val);

    MyComplex operator~();

    friend MyComplex operator*(const float &val, MyComplex &c)
    {
        return c * val;
    }

    friend MyComplex operator*(const MyComplex &c1, const MyComplex &c2);

    friend std::ostream &operator<<(std::ostream &os, const MyComplex &c);
    friend std::istream &operator>>(std::istream &is, MyComplex &c);
};

#endif // !__MY_COMPLEX_HPP__