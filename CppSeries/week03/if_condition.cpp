/**
 * @file if_condition.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-11
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief if-elif-else conditions in C++.
 * @attention The ternary conditional operator ? : 
 *
 */

#include <iostream>

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    /* Step 1. if-elif-else Statements in C++.
    --------------------------------------------- */
    int num = 10;
    if (num < 5)
    {
        std::cout << "the number is less than 5. " << std::endl;
    }

    if (num == 5)
    {
        std::cout << "the number is equal 5. " << std::endl;
    }
    else
    {
        std::cout << "the number is not 5. " << std::endl;
    }

    // Attention: the logic clear remain.
    if (num < 5)
        std::cout << "The number is less than 5." << std::endl;
    else if (num > 10)
        std::cout << "The number is greater than 10." << std::endl;
    else
        std::cout << "The number is in range [5, 10]." << std::endl;

    // logic clear with the Conditional Expressions.
    if (num < 20)
    {
        if (num < 5)
            std::cout << "The number is less than 5" << std::endl;
        else
            std::cout << "Where I'm?" << std::endl;
    }

    int* ptr = new int[1024];
    if (ptr)
    {
        std::cout << "Memory has been allocated. " << std::endl;
    }

    // if (ptr == NULL)
    // if (ptr == nullptr)
    if (!ptr)
    {
        std::cout << "Memory allocation failed." << std::endl;
    }
    
    delete [] ptr;

    /* Step 2. The ternary conditional operator ? : 
    is also widely used to simplify some if else statements.
    --------------------------------------------------------- */
    bool is_positive = true;
    int factor = 0;
    if (is_positive)
    {
        factor = 1;
    }
    else
    {
        factor = -1;
    }

    // simplify by ternary conditional operator ? :
    factor = is_positive ? 1 : -1;

    /* sometimes(is_positive==1) the following code can be more efficient, 
     because of time consumming with if conditional Statements.
    ------------------------------------------------------------  */
    factor = is_positive * 2 - 1;
    
    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ if_condition.cpp
 * $ clang++ if_condition.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */