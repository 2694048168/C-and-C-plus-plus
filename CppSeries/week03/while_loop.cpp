/**
 * @file while_loop.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-11
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief while loop in C++.
 * @attention 死循环问题
 *
 */

#include <iostream>

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    int num = 10;
    while (num > 0)
    {
        std::cout << "num = " << num << std::endl;
        --num;
    }
    std::cout << "---------------------------\n";

    do
    {
        std::cout << "num = " << num << std::endl;
    } while (num > 0);
    std::cout << "---------------------------\n";

    /* we can debug the while loop process with breakpoint(F9),
    and loopup the value of variable 'num'.
    ----------------------------------------- */
    num = 10;
    while (num >= 0)
    {
        if ((num != 0) && (num % 2) == 0)
        {
            // std::cout << "num = " << num << std::endl;
            --num;
            continue;
        }
        else if ((num % 2) == 1)
        {
            std::cout << "num = " << num << std::endl;
            --num;
        }
        else /* num == 0 */
        {
            std::cout << "while loop break." << std::endl;
            break;
        }
    }

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ while_loop.cpp
 * $ clang++ while_loop.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */