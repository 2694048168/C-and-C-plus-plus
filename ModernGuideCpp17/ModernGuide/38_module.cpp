/**
 * @file 38_module.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-08-26
 * 
 * @copyright Copyright (c) 2025
 * 
 * 1. 将模块接口文件预编译为二进制模块接口(BMI), 在Clang下后缀为.pcm
 * clang++ -std=c++20 --precompile 38_module.cppm -o module.pcm
 * clang++ -std=c++20 --precompile 38_module.cppm -Xclang -emit-module-interface -o module.pcm
 * 
 * 2. 将模块接口自身也编译成一个目标文件(.o)
 * clang++ -std=c++20 -c 38_module.cppm -o module.o
 * 
 * 3. 编译主程序时，通过 -fmodule-file 告诉它BMI文件的位置
 * clang++ -std=c++20 -c 38_module.cpp -fmodule-file=math="module.pcm" -o main.o
 * 
 * 4. 最后，像往常一样将所有目标文件链接起来
//  * clang++ main.o module.o module_impl.o -o my_program.exe
 * clang++ main.o module.o -o my_program.exe
 * 
 */

#include <iostream>

import math; // ✨ 导入数学模块！不再需要 #include

// -----------------------------------
int main(int argc, const char *argv[])
{
    std::cout << "1 + 2 = " << add(1, 2) << std::endl;   // 直接调用
    std::cout << "44 - 2 = " << sub(44, 2) << std::endl; // 直接调用

    return 0;
}
