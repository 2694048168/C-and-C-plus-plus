#include "11_9_1_my_vector.hpp"

#include <cmath>

namespace VECTOR {

// compute degrees in one radian
const double Rad_to_deg = 45.0 / std::atan(1.0);

void Vector::set_x()
{
    x = mag * std::cos(ang);
}

void Vector::set_y()
{
    this->y = mag * std::sin(ang);
}

void Vector::set_mag()
{
    this->mag = std::sqrt(x * x + y * y);
}

void Vector::set_ang()
{
    if (x == 0.0 && y == 0.0)
        this->ang = 0.0;
    else
        ang = std::atan2(y, x);
}

Vector::Vector()
{
    x    = 0.0;
    y    = 0.0;
    mag  = 0.0;
    ang  = 0.0;
    mode = RECT;
}

Vector::Vector(double n1, double n2, Mode form)
{
    mode = form;
    if (form == RECT)
    {
        x = n1;
        y = n2;
        set_mag();
        set_ang();
    }
    else if (form == POL)
    {
        mag = n1;
        ang = n2 / Rad_to_deg;
        set_x();
        set_y();
    }
    else
    {
        std::cout << "Incorrect 3rd argument to Vector() -- ";
        std::cout << "vector set to 0\n";

        x    = 0.0;
        y    = 0.0;
        mag  = 0.0;
        ang  = 0.0;
        mode = RECT;
    }
}

void Vector::reset(double n1, double n2, Mode form)
{
    mode = form;
    if (form == RECT)
    {
        x = n1;
        y = n2;
        set_mag();
        set_ang();
    }
    else if (form == POL)
    {
        mag = n1;
        ang = n2 / Rad_to_deg;
        set_x();
        set_y();
    }
    else
    {
        std::cout << "Incorrect 3rd argument to Vector() -- ";
        std::cout << "vector set to 0\n";

        x    = 0.0;
        y    = 0.0;
        mag  = 0.0;
        ang  = 0.0;
        mode = RECT;
    }
}

Vector::~Vector() {}

double Vector::get_x_val() const
{
    return this->x;
}

double Vector::get_y_val() const
{
    return y;
}

double Vector::get_mag_val() const
{
    return mag;
} // report magnitude

double Vector::get_ang_val() const
{
    return this->ang;
} // report angle

void Vector::polar_mode()
{
    this->mode = POL;
}

void Vector::rect_mode()
{
    this->mode = RECT;
}

// operator overloading
Vector Vector::operator+(const Vector &b) const
{
    return Vector(this->x + b.x, this->y + b.y);
}

Vector Vector::operator-(const Vector &b) const
{
    return Vector(this->x - b.x, this->y - b.y);
}

Vector Vector::operator-() const
{
    return Vector(-x, -y);
}

// multiply vector by n
Vector Vector::operator*(double n) const
{
    return Vector(n * x, n * y);
}

// friends method ---> multiply n by Vector a
// "Vector * n" and "n * Vector"
// 两者不同的操作, 类比矩阵乘法的不可交换性质
Vector operator*(double n, const Vector &a)
{
    return a * n;
}

// 针对输出流以友元函数方式重载, 便于 Vector 类型直接使用 C++ 的输入和输出
std::ostream &operator<<(std::ostream &os, const Vector &v)
{
    if (v.mode == Vector::RECT)
        os << "(x,y) = (" << v.x << ", " << v.y << ")";
    else if (v.mode == Vector::POL)
    {
        os << "(m,a) = (" << v.mag << ", " << v.ang * Rad_to_deg << ")";
    }
    else
        os << "Vector object mode is invalid";

    return os;
}

} // namespace VECTOR
