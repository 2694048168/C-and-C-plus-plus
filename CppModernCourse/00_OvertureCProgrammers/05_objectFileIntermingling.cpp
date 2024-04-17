/**
 * @file 05_objectFileIntermingling.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>

/**
 * @brief C and C++ 对象文件的混用, 链接阶段(C编译器链接C++编译器生成的对象文件, 反之亦然),
 * step 1. C and C++ 代码中调用约定需要匹配(__stdcall, __cdecl)
 *      C 语言: __cdecl、__stdcall、__fastcall、naked、__pascal;
 *      C++ 语言: __cdecl、__stdcall、__fastcall、naked、__pascal、__thiscall
 * step 2. C++编译器对导出符号有修饰, C编译器对导出符号无修饰, 有差异需要做匹配
 * 
 * NOTE: 调用函数时堆栈和寄存器的设置协议可能不同(调用约定); 
 *       链接器必须通过唯一的名称(符号 symbol)来识别对象;
 * C++编译器通过装饰目标对象, 将叫作装饰名称的字符串与该对象相关联, 由于函数重载、调用约定
 * 和命名空间的使用, 编译器必须通过装饰对函数的额外信息进行编码, 而不仅仅是其名称, 
 * 这样做是为了确保链接器能够唯一地识别该函数/符号(symbol), 
 * 在C++中没有关于这种装饰的标准, 这就是在编译单元之间进行链接时应该使用相同的工具链和设置的原因;
 * C语言的链接器对C++的名称装饰一无所知, 如果在C++中对C代码进行链接时不停止装饰, 就会产生问题(反之亦然),
 * 这个问题的解决办法很简单, 只要使用 extern"C" 语句将要用C语言风格的链接方式编译的代码包装起来即可,
 * 
 */

// header.h or header.hpp
// #pragma once
#ifdef __cplusplus
extern "C"
{
#endif

struct NumberTensor
{
    int   number = 0;
    float tensor = 0.f;
};

void extract_number()
{
    std::cout << "extract_number\n";
}

#ifdef __cplusplus
}
#endif

/**
 * @brief 这个头文件 header.h 可以在C和C++代码之间共享,
 *  之所以起作用是因为 _cplusplus 是一个特殊的标识符, C++编译器定义了它, 但C编译器没有,
 * 在预处理完成后，C编译器只看到中间代码区, 这只是一个简单的C头文件, 在预处理过程中, 
 * #ifdef __cplusplus 语句之间的代码被删除了, 所以 extern "C" 包装器并不可见;
 * 对于C++编译器来说, _cplusplus 被定义在header.h 中, 中间代码都用 extern "c" 来包装,
 * 所以编译器知道要使用C链接;
 * 这样现在C源代码可以调用已编译的C++代码, C++源代码也可以调用已编译的C代码.
 * 
 */

// ------------------------------------
int main(int argc, const char **argv)
{
    extract_number();

    return 0;
}
