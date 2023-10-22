/**
 * @file 11_preprocessor_directives.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

// #include <iostream>
#include <vector>

// Preprocessor directives
// https://cplusplus.com/doc/tutorial/preprocessor/
// step 1. macro definitions (#define, #undef)
#define GetMax(a, b) ((a) > (b) ? (a) : (b))

#define SIZE 100
std::vector<int> vec1(SIZE);
#undef SIZE
#define SIZE 200
std::vector<int> vec2(SIZE);

// Function macro definitions
// accept two special operators (# and ##) in the replacement sequence:
#define PrintStr(str)           #str
#define PrintTwoStr(str1, str2) str1##str2

// step 2. Conditional inclusions (#ifdef, #ifndef, #if, #endif, #else and #elif)
#ifdef __cplusplus
#    include <iostream>
#else
#    include <stdio.h>
#endif

// step 3.
// Line control (#line)
// Error directive (#error)
// Source file inclusion (#include)
// Pragma directive (#pragma)

// step 4. Predefined macro names

// -----------------------------
int main(int argc, char **argv)
{
    unsigned int num_magic = 42;

    std::cout << "the max(num_magic, 24): " << GetMax(num_magic, 24) << '\n';
    std::cout << "the max(num_magic, 64): " << GetMax(num_magic, 64) << '\n';

    std::cout << "the size of vec1: " << vec1.size() << '\n';
    std::cout << "the size of vec2: " << vec2.size() << '\n';

    std::PrintTwoStr(c, out) << PrintStr(test Function macro) << '\n';

    // ======== Predefined macro names ========
    std::cout << "-------- Predefined macro names: --------\n";
    std::cout << __LINE__ << '\n';
    std::cout << __FILE__ << '\n';
    std::cout << __DATE__ << '\n';
    std::cout << __TIME__ << '\n';
    std::cout << __cplusplus << '\n';
    std::cout << __STDC_HOSTED__ << '\n';

    std::cout << "This is the line number " << __LINE__;
    std::cout << " of file " << __FILE__ << ".\n";
    std::cout << "Its compilation began " << __DATE__;
    std::cout << " at " << __TIME__ << ".\n";
    std::cout << "The compiler gives a __cplusplus value of " << __cplusplus;

    return 0;
}
