/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-11-15
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cmath>
#include <iostream>
#include <string>

/**
 * @brief Pascal's triangle
 *  Write a function that prints up to 10 rows of Pascal's triangle to the console.
 */

/**
 * @brief Solution:
Pascal's triangle is a construction representing binomial coefficients. 
The triangle starts with a row that has a single value of 1. 
Elements of each row are constructed by summing the numbers above, 
to the left and right, and treating blank entries as 0.

To print the triangle, we must:
1. Shift the output position to the right with an appropriate number of spaces,
    so that the top is projected on the middle of the triangle's base.
2. Compute each value by summing the above left and right values. 
    A simpler formula is that for a row i and column j, each new value x is 
    equal to the previous value of x multiplied by (i - j) / (j + 1), 
    where x starts at 1.
------------------------------------------------------ */
unsigned int number_of_digits(const unsigned int i)
{
    return i > 0 ? (int)log10((double)i) + 1 : 1;
}

void print_pascal_triangle(const unsigned int n)
{
    for (unsigned int i; i < n; ++i)
    {
        auto x = 1;
        std::cout << std::string((n - i - 1) * (n / 2), ' ');
        for (unsigned int j = 0; j <= i; ++j)
        {
            auto y      = x;
            x           = x * (i - j) / (j + 1);
            auto maxlen = number_of_digits(x) - 1;
            std::cout << y << std::string(n - 1 - maxlen - n % 2, ' ');
        }
        std::cout << '\n';
    }
}

// ------------------------------
int main(int argc, char **argv)
{
    unsigned int num = 0;
    std::cout << "Levels (up to 10): ";
    std::cin >> num;

    if (num > 10)
        std::cout << "Value too large" << std::endl;
    else
        print_pascal_triangle(num);

    return 0;
}
