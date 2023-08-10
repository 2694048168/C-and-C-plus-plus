#include "11_9_7_my_complex.hpp"

#include <iostream>

MyComplex::MyComplex(const float real, const float imaginary)
{
    this->real      = real;
    this->imaginary = imaginary;
}

MyComplex::~MyComplex() {}

// operator overloading
MyComplex MyComplex::operator+(const MyComplex &c)
{
    this->real += c.real;
    this->imaginary += c.imaginary;

    return *this;
}

MyComplex MyComplex::operator-(const MyComplex &c)
{
    this->real -= c.real;
    this->imaginary -= c.imaginary;

    return *this;
}

MyComplex MyComplex::operator*(const float &val)
{
    this->real *= val;
    this->imaginary *= val;

    return *this;
}

MyComplex MyComplex::operator~()
{
    MyComplex temp;
    temp.real      = this->real;
    temp.imaginary = -this->imaginary;

    return temp;
}

MyComplex operator*(const MyComplex &c1, const MyComplex &c2)
{
    MyComplex temp;
    temp.real      = c1.real * c2.real - c1.imaginary * c2.imaginary;
    temp.imaginary = c1.real * c2.imaginary + c1.imaginary * c2.real;

    return temp;
}

std::ostream &operator<<(std::ostream &os, const MyComplex &c)
{
    std::cout << "(" << c.real << ", " << c.imaginary << "i)\n";

    return os;
}

std::istream &operator>>(std::istream &is, MyComplex &c)
{
    std::cout << "real: ";
    is >> c.real;

    std::cout << "imaginary: ";
    is >> c.imaginary;

    return is;
}
