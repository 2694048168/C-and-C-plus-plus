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

#include <module/lib.hpp>

// -----------------------------------
int main(int argc, const char **argv)
{
    ProjectName::ModuleName::greeter g;

    g.greet();

    return 0;
}