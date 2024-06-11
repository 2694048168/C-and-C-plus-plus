/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-06-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */
// Here we can see consumer can easily include and use our package.

#include <hello.hpp>

// -------------------------------------
int main(int argc, const char **argv)
{
    ProjectName::ModuleName::greeter g;

    g.greet();
}
