/**
 * @file assert.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-14
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief assert <cassert> header file in C++
 * @attention assert and NDEBUG macro
 *
 */

#include <iostream>
#include <cassert>

/**
 * @brief main function
 */
int main(int argc, char const *argv[])
{
    assert( argc == 2);

#ifdef NDEBUG
    std::cout << "this is release version." << std::endl;
#else
    std::cout << "this is debug version." << std::endl;
#endif // NDEBUG

    std::cout << "This is an assert example." << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ assert.cpp
 * $ clang++ assert.cpp -std=c++17
 * $ clang++ assert.cpp -DNDEBUG
 * $ clang++ assert.cpp -DNDEBUG -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */