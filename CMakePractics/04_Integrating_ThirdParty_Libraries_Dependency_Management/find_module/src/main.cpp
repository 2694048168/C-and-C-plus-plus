/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-06-13
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <obscure/obscure.hpp>

// -------------------------------------
int main(int argc, const char **argv)
{
    obscure::Obscure greeter{"Arcane wizard"};
    greeter.greet();

    return 0;
}
