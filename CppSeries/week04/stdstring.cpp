/**
 * @file stdstring.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-12
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief std::string data type in C++.
 * @attention
 *
 */

#include <iostream>
#include <string>

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    /* Step 1. std::string in <string> header.
    -------------------------------------------- */
    std::string str1 = "Hello";
    std::string str2 = "Wei Li";
    std::string result = str1 + ", " + str2;

    std::cout << "reslut = " << result << std::endl;
    std::cout << "the lenght or size: " << result.length() << std::endl;
    std::cout << "str1 < str2 is " << (str1 < str2) << std::endl
              << std::endl;

    /* Step 2. std::string bound check.
    ----------------------------------- */
    std::string str = "Hello, SUSTech!";
    for (size_t i = 0; i < str.length(); i++) /* no bound check! */
    {
        std::cout << i << ": " << str[i] << std::endl;
    }

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ stdstring.cpp
 * $ clang++ stdstring.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */