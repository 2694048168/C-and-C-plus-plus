/**
 * @file 04_fundamentalTypes.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

/**
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

void taunt()
{
    printf("\nHey, laser lips, your mama was a snow blower.\n");
}

// -----------------------------------
int main(int argc, const char **argv)
{
    size_t size_c = sizeof(char);
    printf("char: %zd\n", size_c);

    size_t size_s = sizeof(short);
    printf("short: %zd\n", size_s);

    size_t size_i = sizeof(int);
    printf("int: %zd\n", size_i);

    size_t size_l = sizeof(long);
    printf("long: %zd\n", size_l);

    size_t size_ll = sizeof(long long);
    printf("long long: %zd\n", size_ll);

    taunt();

    return 0;
}
