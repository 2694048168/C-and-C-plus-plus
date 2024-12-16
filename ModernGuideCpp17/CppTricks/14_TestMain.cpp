/**
 * @file 14_TestMain.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "14_DynamicPolymorphism.hpp"
#include "14_StaticPolymorphism.hpp"

#include <iostream>

int main(int /* argc */, char ** /* argv */)
{
    IInfo *pA = new A();
    IInfo *pB = new B();
    std::cout << pA->getClassName() << std::endl;
    std::cout << pB->getClassName() << std::endl;

    C classC;
    D classD;
    std::cout << classC.getClassName() << std::endl;
    std::cout << classD.getClassName() << std::endl;

    return 0;
}
