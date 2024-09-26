/**
 * @file 07_sprintf.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

// C 语言支持的 文件I/O 操作, 和 C++ 语言层面提供的接口有点差异
// https://subingwen.cn/c/file/

// C 语言层面支持的 内存布局
// https://subingwen.cn/c/memory-layout/

// ------------------------------------
int main(int argc, const char **argv)
{
    // sprintf 是一个 C 语言标准库函数，用于将格式化的数据写入字符串中
    // int sprintf(char* str, const char* format, ...);
    char        str[100];
    int         num = 123;
    float       f   = 3.14f;
    const char *p   = "hello, world!";

    sprintf(str, "整数：%d, 浮点数：%.2f, 字符串: %s", num, f, p);
    printf("格式化后的字符串：%s\n", str);

    // 除了将格式化数据写入字符串数组中，还有类似的函数,
    // 可以将格式化数据写入文件中(fprintf)或标准输出流中(printf),
    // printf(const char *const Format, ...)
    // fprintf(FILE *const Stream, const char *const Format, ...)
    // 这些函数在使用方式和格式化字符串的语法上类似

    /* 
      * step 1. 整数类型
  * 整数类型存储的是整数: short int、int、long int 和 long long int;
  * 每个类型都可以是有符号(signed)或无符号(unsigned)的;
  * 有符号变量可以是正数、负数或零，无符号变量必须是非负数;
  * 整数类型默认是有符号的 int 类型, 这意味着可以在程序中使用简写符号 short、 long 和long long,
  * 而不是 short int、long int 和 long long int;
  * =================== | ========================
  * Type                | printf-format-specifier
  * short               | %hd
  * unsigned short      | %hu
  * int                 | %d
  * unsigned int        | %u
  * long                | %ld
  * unsigned long       | %lu
  * long long           | %lld
  * unsigned long long  | %llu
  * =================== | ========================

  * 2.浮点格式指定符
     * 格式指定符 %f 显示带有小数位的浮点数,
     * 而 %e 则以科学计数法显示相同的数字,
     * 也可以让 printf 使用 %g 格式指定符, 选择 %e 或 %f 中更紧凑的一个;
     * 对于 double, 只需在说明符前面加上小写字母l,
     * 而对于 long double, 在前面加上大写字母L;
     * 一般来说，使用 %g 来打印浮点类型,
     * NOTE: 在实践中, 可以省略 double 格式指定符中的l前缀, 
     *      因为 printf 会将浮点数参数提升为双精度类型.
    
    *为了将这些字符转换为 char, 可以使用转义序列,
  *  Reserved Characters and Their Escape Sequences
  * Value               | Escape sequence | 字符名称
  * --------------------|-----------------|-----------
  * Newline             | \n              | 换行
  * Tab (horizontal)    | \t              | Tab(水平)
  * Tab (vertical)      | \v              | Tab(垂直)
  * Backspace           | \b              | 退格
  * Carriage return     | \r              | 回车
  * Form feed           | \f              | 换页
  * Alert               | \a              | 报警声
  * Backslash           | \\              | 反斜杠
  * Question mark       | ? or \?         | 问号
  * Single quote        | \'              | 单引号
  * Double quote        | \"              | 双引号
  * The null character  | \0              | 空字符
  * ================================================== 

  * @brief std::byte类型
  * 系统程序员有时会直接使用原始内存, 原始内存是一个没有类型的位(bit)集合,
  * 这种情况下可以使用 std:.byte 类型, 它定义在 <cstddef>头文件中.
  * std:byte 类型允许按位进行逻辑运算, 使用这种类型而不是整数类型来处理原始数据, 常常可以避免难以调试的编程错误.
  * NOTE: <cstddef>中的大多数其他基本类型不同, std::byte 在C语言中没有确切的对应类型"C类型".
  * 
  * ======== size_t类型
  * size_t类型(<cstddef>头文件中)用来表示对象的大小,
  * size_t对象保证其最大值足以代表所有对象的最大字节数,
  * 从技术上讲, 这意味着 size_t 可以占用2个字节, 也可以占用200个字节, 具体取决于实现方式,
  * 在实践中, 它通常与64位架构系统的 unsigned long long 相同.
  * NOTE: size_t是<stddef>头文件中的一个C类型, 但它与C++的 std::size_t相同.
  *
  * 1.sizeof 一元运算符, sizeof 接受一个类型并返回该类型的大小(以字节为单位),
  * sizeof 运算符总是返回一个 size_t对象.
  * 2.格式指定符, size_t 的格式指定符通常是 %zd (十进制表示)或 %zx(十六进制表示)
  * 
  * ======== void
  * void 类型表示一个空的值集合, 因为 void 对象不能拥有值, 所以C++不允许使用 void 对象;
  * 只在特殊情况下使用 void, 比如用作不返回任何值的函数的返回类型.
  * 
     */

    return 0;
}
