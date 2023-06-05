/**
 * @file event17.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-23
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

namespace ProjectName::FunctionModule::ClassName {
int seed = 42;

}

int main(int argc, const char **argv)
{
    // 每一个编译器(GCC, Clang, MSVC)的每一个版本针对 Feature-test macro 的值有所不同
    std::cout << "__cplusplus: " << __cplusplus << "\n";
    std::cout << "__cpp_lib_chrono: " << __cpp_lib_chrono << "\n";

    std::cout << "the random seed: " << ProjectName::FunctionModule::ClassName::seed << std::endl;

    return 0;
}

// 需要考虑编译器的差异(GCC/Clang/MSVC)
// 编译器的版本差异(GCC10/11/12/13, Clang13/14/15/16)
// C++标准的差异(-std=C++17/20/23)
/* 
$ clang++ event17.cpp
__cplusplus: 201402

$ g++ event17.cpp
__cplusplus: 201703

$ cl event17.cpp
__cplusplus: 199711

 */