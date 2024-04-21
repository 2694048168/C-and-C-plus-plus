/**
 * @file 03_fundamentalBoolTypes.cpp
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
  * @brief 布尔类型有两种状态: 真和假, 布尔类型只有一个: bool;
  * 整数类型和布尔类型可以互相转换: true 状态转换为1, false 状态转换为0;
  * 任何非零的整数都转换为 true, 而0则转换为 false;
  *
  * 1. 布尔字面量
  * 要初始化布尔类型, 需要使用两个布尔字面量, 即 true 和 false;
  * 2. 格式指定符bool, 没有格式指定符, 可以在 printf 中使用 int 格式指定符 %d
  *   来产生1(代表 true)或0(代表 false), 原因是 printf 将任何小于 int 的整数值提升为 int;
  * 
  */

// --------------------------------------
int main(int argc, const char **argv)
{
    bool b_1 = true;
    bool b_2 = false;
    printf("the Boolean Literals %d, %d\n", b_1, b_2);

    /**
     * @brief 3.比较运算符
     * 运算符是对操作数进行计算的函数, 操作数是一种简单对象,
     * 可以使用几个运算符来构建布尔表达式, 比较运算符接受两个参数并返回一个布尔值.
     * 
     */
    printf(" 7 == 7: %du\n", 7 == 7);
    printf(" 7 != 7: %d\n", 7 != 7);
    printf("10 > 20: %d\n", 10 > 20);
    printf("10 >= 20: %d\n", 10 >= 20);
    printf("10 < 20: %d\n", 10 < 20);
    printf("20 <= 20: %d\n", 20 <= 20);

    /**
     * @brief 4.逻辑运算符
     * 逻辑运算符在 bool 类型上处理布尔逻辑, 通过操作数的数量来对运算符分类;
     * 一元运算符需要一个操作数, 二元运算符需要两个, 三元运算符需要三个, 以此类推;
     * 取否运算符(!)接受一个操作数, 并返回与操作数相反的结果
     * 逻辑运算符"与"(&&)和"或"(||)是二元的, 
     * 逻辑"与"(AND)只在两个操作数都为 true 时返回 true,
     * 逻辑"或"(OR)只要有操作数为 true 就返回 true.
     * NOTE: 阅读布尔表达式时, !的发音是"not", 如表达式 a&&!b 表示"a AND not b".
     * 
     */
    bool t = true;
    bool f = false;
    printf("!true: %d\n", !t);
    printf("true && false: %d\n", t && f);
    printf("true && !false: %d\n", t && !f);
    printf("true || false: %d\n", t || f);
    printf("false || false: %d\n", f || f);

    return 0;
}
