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
    std::cout << "__cplusplus: " << __cplusplus << std::endl;

    std::cout << "the random seed: " << ProjectName::FunctionModule::ClassName::seed << std::endl;

    return 0;
}

/* 
$ clang++ event17.cpp
__cplusplus: 201402

$ g++ event17.cpp
__cplusplus: 201703

$ cl event17.cpp
__cplusplus: 199711

 */