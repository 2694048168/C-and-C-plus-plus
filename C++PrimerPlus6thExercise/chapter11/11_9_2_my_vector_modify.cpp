#include "11_9_2_my_vector_modify.hpp"

#include <cmath>

namespace VECTOR {

// compute degrees in one radian
const double Rad_to_deg = 45.0 / std::atan(1.0);

void Vector::set_x(double mag, double ang)
{
    this->x = mag * std::cos(ang);
}

void Vector::set_y(double mag, double ang)
{
    this->y = mag * std::sin(ang);
}

Vector::Vector()
{
    x    = 0.0;
    y    = 0.0;
    mode = RECT;
}

Vector::Vector(double n1, double n2, Mode form)
{
    mode = form;
    if (form == RECT)
    {
        x = n1;
        y = n2;
    }
    else if (form == POL)
    {
        set_x(n1, n2);
        set_y(n1, n2);
    }
    else
    {
        std::cout << "Incorrect 3rd argument to Vector() -- ";
        std::cout << "vector set to 0\n";

        x    = 0.0;
        y    = 0.0;
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
    }
    else if (form == POL)
    {
        set_x(n1, n2);
        set_y(n1, n2);
    }
    else
    {
        std::cout << "Incorrect 3rd argument to Vector() -- ";
        std::cout << "vector set to 0\n";

        x    = 0.0;
        y    = 0.0;
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
    return std::sqrt(x * x + y * y);
} // report magnitude

double Vector::get_ang_val() const
{
    if (x == 0.0 && y == 0.0)
        return 0.0;
    else
        return std::atan2(y, x);
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
        os << "(m,a) = (" << v.get_mag_val() << ", " << v.get_ang_val() * Rad_to_deg << ")";
    }
    else
        os << "Vector object mode is invalid";

    return os;
}

} // namespace VECTOR
