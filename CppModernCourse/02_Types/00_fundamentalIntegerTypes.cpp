/**
 * @file 00_fundamentalIntegerTypes.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/**
  * @brief 基本类型是最基本的对象类型: 整数、浮点数、字符、布尔、 byte、 size t和 void;
  * 把基本类型称为原始类型或内置类型, 因为它们是核心语言的一部分, 几乎总是使用的;
  * ! 这些类型可以在任何平台上工作, 但它们的特性, 如大小和内存布局, 则取决于具体的实现
  * 基本类型取得了一种平衡, 试图映射从C++结构到计算机硬件的直接关系;
  * 另一方面, 它们简化了跨平台代码的编写, 允许程序员写一次代码就可以在许多平台上运行.
  * 
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
  * ! 编译器会在格式指定符和整数类型不匹配时发出警告;
  * NOTE: 如果想确保整数的大小, 那么可以使用 <cstdint> 库中的整数类型;
  * int8t、int16t、int32t或int64t;
  * 可以在这个库中找到速度最快、最小、最大、有符号和无符号整数类型, 以满足各种要求;
  * ! 但由于这个头文件并不总是在每个平台上都可用, 因此只在没有其他选择时使用 cstdint 类型.
  * 
  * 默认情况下, 整数字面量的类型一般是 int、long 或 long long;
  * 整数字面量的类型是这三种类型中最小的那种, 这是由语言定义的, 并将由编译器强制执行;
  * 如果想更灵活, 则可以给整数字面量提供后缀来指定它的类型(后缀不区分大小写).
  * 1. unsigned 对应后缀 u 或 U;
  * 2. long 对应后缀 l 或 L;
  * 3. long long 对应后缀 ll 或 LL;
  * 把 unsigned 后缀和 long 后缀或 long long 后缀结合起来可以指定整数类型的符号性和大小.
  *
  */

// -----------------------------------
#include <cstdio>

int main(int argc, const char **argv)
{
    /**
    * @brief 字面量是程序中的硬编码值, 可以使用四种硬编码的、整数字面量表示:
    * 1. 二进制: 使用前缀 0b, 
    * 2. 八进制: 使用前缀 0,
    * 3. 十进制: 这是默认的, 
    * 4. 十六进制: 使用前缀 0x,
    * 这是同一组整数的四种不同写法:
    * 
    */
    printf("======== Integer Types ========\n");
    unsigned short a = 0b10101010;
    printf("========= %hu\n", a);

    int b = 0123;
    printf("========= %d\n", b);

    unsigned long long d = 0xFFFFFFFFFFFFFFFF;
    printf("========= %llu\n", d);
    /** NOTE: 整数字面量可以包含任何数量的单引号('), 以方便阅读;
     * 编译器会完全忽略这些引号, 1000000 和 1'000'000 都是表示一百万的字面量. 
     *
     * 有时打印无符号整数的十六进制表示或八进制表示(较少见)是很有用的,
     * 可以使用printf 格式指定符 %x 和 %o 实现这个目的
     */
    unsigned int num_a = 366732608;
    printf("Yabba %x !\n", num_a);
    unsigned int num_b = 69;
    printf("There are %u, %o leaves here.\n", num_b, num_b);

    printf("======== Integer literal type ========\n");
    /**
     * @brief 允许的最小类型仍能表示整数字面量的类型就是最终类型
     * 这意味着, 在特定整数允许的所有类型中, 最小的类型将被应用
     * 例如, 整数字面量 112114 可以是 int、long、long long 类型的,
     * 由于 int 可以存储 112114, 因此最终的整数字面量是 int 类型的;
     * 如果真的想采用 long 类型, 则可以指定为 112114L(或 112114l);
     * 
     */
    auto num   = 112114;
    auto num_1 = 112114L;
    auto num_2 = 112114LL;
    printf("the number of bytes for %d is %llu\n", num, sizeof(num));
    printf("the number of bytes for %ld is %llu\n", num_1, sizeof(num_1));
    printf("the number of bytes for %lld is %llu\n", num_2, sizeof(num_2));

    return 0;
}
