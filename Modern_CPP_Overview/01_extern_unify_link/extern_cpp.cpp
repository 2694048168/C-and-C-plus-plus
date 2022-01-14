/**
 * @file externrn_cpp.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief 在 C++ 代码中使用 C 语言的代码，通过头文件的 extern "C" 方式进行混和编译，然后链接时统一
 * @version 0.1
 * @date 2022-01-06
 * 
 * @copyright Copyright (c) 2022
 * 
 */

// head files order
#include "add.h"

#include <iostream>
#include <functional>


int main(int argc, char** argv)
{
    // 函数式编程方式
    [out = std::ref(std::cout << "Result from C code: " << add(3, 5))]()
    {
        out.get() << ".\n";
    }();
    
    return 0;
}
