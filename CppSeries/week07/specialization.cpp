/**
 * @file specialization.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-12
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief the function template specialization in C++
 * @attention
 *
 */

#include <iostream>
#include <typeinfo>


template <typename T>
T mysum(T x, T y)
{
    std::cout << "The input type is "
              << typeid(T).name() << "\n";
    return x + y;
}

struct Point
{
    int x;
    int y;
};

// Specialization for Point + Point operation
template <>
Point mysum<Point>(Point pt1, Point pt2)
{
    std::cout << "The input type is " << typeid(pt1).name() << std::endl;
    Point pt;
    pt.x = pt1.x + pt2.x;
    pt.y = pt1.y + pt2.y;
    return pt;
}

// Explicitly instantiate
template double mysum<double>(double, double);

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    // Explicit instantiated functions
    std::cout << "sum = " << mysum(1, 2) << std::endl;
    std::cout << "sum = " << mysum(1.1, 2.2) << std::endl;

    Point pt1 {1, 2};
    Point pt2 {2, 3};
    Point pt = mysum(pt1, pt2);
    std::cout << "pt = (" << pt.x << ", " << pt.y << ")" << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ specialization.cpp
 * $ clang++ specialization.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */