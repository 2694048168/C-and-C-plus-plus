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

    std::cout << "============================================\n\n";
    ColorPrinter colorful;
    BWPrinter    blackwhite;

    std::cout << "彩色打印机开始工作啦：\n";
    colorful.print();

    std::cout << "\n换个打印机试试: \n";
    blackwhite.print();

    std::cout << "============================================\n\n";
    Cat  kitty;
    Duck donald;

    std::cout << "=== 铲屎官回家啦 ===\n\n";
    kitty.makeSound(); // 编译时就知道要调用哪个喵星人啦！
    kitty.findFood();

    std::cout << "\n=== 池塘边热闹起来啦 ===\n\n";
    donald.makeSound(); // 鸭鸭的叫声也在编译时就确定好啦！
    donald.findFood();

    std::cout << "============================================\n\n";
    // 像搭积木一样的链式调用
    RobotBuilder().name("小闪电").color("星空蓝").power(100);

    return 0;
}
