/**
 * @file hello.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-11
 * @brief An introduction to C++ with compile and link,
 *   which introduce how to convert source code to binary.
 *
 * @copyright Copyright (c) 2023
 *
 */

/**
 * Verbose Mode (-v)
 * You can see the detailed compilation process by enabling -v (verbose) option.
 * For example, g++ -v -o hello hello.cpp clang++ -v -o hello hello.cpp
 *
 * What goes inside the compilation process?
 * Compiler converts a C/C++ program into an executable.
 * There are four phases for a C program to become an executable:
 * 1. Pre-processing (hello.cpp ---> hello.ii)
 * 2. Compilation (hello.ii ---> hello.s)
 * 3. Assembly (hello.s ---> hello.o)
 * 4. Linking (hello.o ---> hello)
 *
 * For example following command line,
 * g++ -Wall -save-temps -o hello hello.cpp
 * clang++ -Wall -save-temps -o hello hello.cpp
 *
 * 1. Pre-processing
 * g++ -Wall -E hello.cpp -o hello.ii
 * clang++ -Wall -E hello.cpp -o hello.ii
 *
 * 2. Compile
 * g++ -Wall -S hello.ii -o hello.s
 * clang++ -Wall -S hello.ii -o hello.s
 *
 * 3. Assembly
 * g++ -Wall -c hello.s -o hello.o
 * clang++ -Wall -c hello.s -o hello.o
 *
 * 4. Linking
 * g++ -Wall hello.o -o hello
 * clang++ -Wall hello.o -o hello
 *
 */

#include <iostream>
#include <vector>
#include <string>

#define PI 3.1415926

/**
 * @brief main function and the entry of program.
 */
int main(int argc, char const *argv[])
{
    /* Step 1. the basic output stream in C++.
    --------------------------------------------- */ 
    std::vector<std::string> msg{"Hello", "C++", "World", "!"};

    for (const std::string &word : msg)
    {
        std::cout << word << " ";
    }
    std::cout << std::endl;

    std::cout << "Hello World.\n"
              << "Hello Wei Li." << std::endl;

    /* Step 2. the argument of command line in C++.
    ------------------------------------------------- */ 
    for (size_t i = 0; i < argc; ++i)
    {
        std::cout << "the " << i << " argument: " << argv[i] << "\n";
    }
    std::cout << std::endl;

    /* Step 3. the macro via #define in C++.
    ----------------------------------------- */ 
    float radius_cicle = 4.f;
    // implicit type conversion: float ---> double
    double area_cicle = PI * radius_cicle * radius_cicle;
    std::cout << "the area of a cicle: " << area_cicle << std::endl;

    return 0;
}


/** Build(compile and link) commands via command-line.
 * 
 * $ clang++ hello.cpp
 * $ clang++ hello.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac 
 * 
*/