/**
 * ______________________________________________________
 * @file main.cpp
 *
 * Calculator implementation
 *
 * SPDX-License-Identifier:	MIT
 * ______________________________________________________
 */

#include <calc/calculator.hpp>

#include <iostream>

/**
 * @brief Main entry point of the application
 *
 * @return int Exit code
 */
int main(void)
{
    ProjectName::Calc::calculator calc;

    std::cout << "5 + 6: " << calc.sum(5, 6) << std::endl;
    std::cout << "5 - 6: " << calc.sub(5, 6) << std::endl;
    std::cout << "5 * 6: " << calc.mul(5, 6) << std::endl;
    std::cout << "5 / 6: " << calc.div(5, 6) << std::endl;

    // TIP OF THE DAY: return value is optional in main() function
}