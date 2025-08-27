/**
 * @file main.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-08-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */

import Math; // 导入自定义模块
import MathModule;
/** VS2022 使用其他模块(采用 C++20 module 生成的库)
* 1. module 应该提供 Math.cppm.ifc 文件, 替代原始 头文件 方式
* 2. module 同时需要提供 lib 符号导入库, dll 可执行文件
* 3. 使用端, 依然通过 lib 链接符号(符号的可见性)
* 4. 使用端, 项目属性配置 C/C++ ---> 常规 ---> 其他 BMI 目录(指向 Math.cppm.ifc 文件路径)
* 5. 使用端, 项目属性配置 C/C++ ---> 命令行 ---> /reference Math.cppm.ifc(指明模块BMI文件)
*/

#include <iostream>

// ------------------------------------
int main(int argc, const char **argv)
{
    std::cout << "add result = " << Math::add(5, 3) << std::endl;
    std::cout << "sub result = " << Math::sub(5, 3) << std::endl;
    std::cout << "div result = " << Math::div(5, 3) << std::endl;
    std::cout << "mul result = " << Math::mul(5, 3) << std::endl;

    Math::MathDemo calculator;
    std::cout << "calculator add result = " << calculator.Add(5, 3) << std::endl;
    std::cout << "calculator sub result = " << calculator.Sub(5, 3) << std::endl;
    std::cout << "calculator div result = " << calculator.Div(5, 3) << std::endl;
    std::cout << "calculator mul result = " << calculator.Mul(5, 3) << std::endl;

    // ---------------------------------------
    std::cout << "add result = " << MathModule::add(5, 3) << std::endl;
    std::cout << "sub result = " << MathModule::sub(5, 3) << std::endl;
    std::cout << "div result = " << MathModule::div(5, 3) << std::endl;
    std::cout << "mul result = " << MathModule::mul(5, 3) << std::endl;

    MathModule::MathDemo calculator_;
    std::cout << "calculator_ add result = " << calculator_.Add(5, 3) << std::endl;
    std::cout << "calculator_ sub result = " << calculator_.Sub(5, 3) << std::endl;
    std::cout << "calculator_ div result = " << calculator_.Div(5, 3) << std::endl;
    std::cout << "calculator_ mul result = " << calculator_.Mul(5, 3) << std::endl;

    // system("pause");
    return 0;
}
