/**
 * @file 19_SpecialMacros.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <string>

/**
  * @brief 宏定义中的#、##、__VA_ARGS__和...的含义和作用
  * 在C/C++中宏 Macro 是预处理器(preprocessor)的一部分,在预处理阶段被执行文本替换
  * 宏分为类对象宏（object-like）和类函数宏（function-like）
  * 二者的主要区别在于使用时, 类对象宏在使用时类似于数据对象,类函数宏在使用时类似于函数调用
  * 
  * 宏中还可以使用#、##、@#、__VA_ARGS__、...等具有特殊含义的符号或变量
  * 
  */

/* #在C/C++宏定义中的作用就是实现字符串化 Stringizing
* 就是当参数前面有一个#时, 预处理器会将该参数替换为一个字符串字面量
* #用在宏中通常用于调试目的，或者当需要将宏参数作为字符串处理时
*/
#define STRINGIFY(x) #x // # 和 x 之间可以有一个或多个空格

/* ##在C/C++宏定义中的作用是标记粘贴 Token-pasting
* 从字面意思不太好理解, 但其实就是当宏定义中两个参数（token）中间有一个##时
* 预处理器会将这两个参数（token）拼接成一个参数（即一个token）
* 所以##在宏定义中的作用可以简单理解为是一个连接操作符
* ##用在宏中通常用来创建具有动态名称的变量或函数
*/
#define CONCAT(a, b) a##b
#define varNum(x)    var##x

/* 可变参数宏,宏也是支持可变长参数的
* 它是通过__VA_ARGS__这个系统预定义宏和省略号...来实现的
* 关于宏,当宏定义体的内容较长一行写不完,
* 或为了便于阅读而需要分行书写时,
* 可以使用续行符（\，反斜杠）来实现
*/
#define PRINT1(templt, ...) fprintf(stdout, templt, ##__VA_ARGS__)

// #define PRINT2(templt, args...) fprintf(stdout, templt, args)

// -------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    printf("==========================\n");
    std::string greet = STRINGIFY(Hello World);    // 这里宏替换后的类型是const char*
    std::cout << (greet + " in C++") << std::endl; // 输出: Hello World in C++
    printf("%s\n", greet.c_str());                 // 输出: Hello World

    printf("==========================\n");
    int CONCAT(my, Variable) = 42; // 这行代码等同于 int myVariable = 42;

    std::cout << "myVariable = " << myVariable << std::endl; // myVariable = 42
    int var1  = 9;
    varNum(1) = 32;                              // 扩展为var1 = 32;
    std::cout << "var1 = " << var1 << std::endl; // var1 = 32

    printf("==========================\n");
    PRINT1("abc\n");                 // abc
    PRINT1("%d, %s\n", 42, "hello"); // 42, hello
    // PRINT2("%d, %s\n", 43, "world"); // 43, world

    printf("==========================\n");
    std::cout << "the file-name macro: " << __FILE__ << '\n';
    std::cout << "the line-number macro: " << __LINE__ << '\n';
    std::cout << "the function-name macro: " << __FUNCTION__ << '\n';

    return 0;
}
