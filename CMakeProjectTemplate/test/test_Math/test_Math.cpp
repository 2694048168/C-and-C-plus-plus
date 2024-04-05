#include "MathLib/Math.hpp"

#include <iostream>

void test_add(MyMath &math_obj)
{
    int val1 = 12;
    int val2 = 24;
    int res  = 0;
    math_obj.addNumer(val1, val2, res);

    std::cout << val1 << " + " << val2 << " = " << res << '\n';
}

void test_sub(MyMath &math_obj)
{
    int val1 = 12;
    int val2 = 24;
    int res  = 0;
    math_obj.subNumer(val2, val1, res);

    std::cout << val2 << " - " << val1 << " = " << res << '\n';
}

void test_mul(MyMath &math_obj)
{
    int val1 = 12;
    int val2 = 24;
    int res  = 0;
    math_obj.mulNumer(val1, val2, res);

    std::cout << val1 << " * " << val2 << " = " << res << '\n';
}

void test_div(MyMath &math_obj)
{
    int val1 = 12;
    int val2 = 24;
    int res  = 0;
    math_obj.addNumer(val2, val1, res);

    std::cout << val2 << " / " << val1 << " = " << res << '\n';
}

// ====================================
int main(int argc, const char **argv)
{
    MyMath math_obj;

    test_add(math_obj);
    test_sub(math_obj);
    test_mul(math_obj);
    test_div(math_obj);

    return 0;
}