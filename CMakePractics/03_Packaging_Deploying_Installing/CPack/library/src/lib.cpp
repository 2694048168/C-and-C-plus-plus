/**
 * @file lib.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-06-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <module/lib.hpp>

#include <iostream>

namespace ProjectName { namespace ModuleName {

void greeter::greet()
{
    std::cout << "Hello, world!" << std::endl;
}

}} // namespace ProjectName::ModuleName