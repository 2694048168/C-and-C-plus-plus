#include <iostream>

#include "function.hpp"


int main(int argc, char const *argv[])
{
    printhello("hello CPP world.");

    std::cout << "This is main:" << std::endl;
    std::cout << "The factorial of 5 is: " << factorial(5) << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line or CMake.
 *
 * $ clang++ *.cpp
 * $ clang++ *.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */