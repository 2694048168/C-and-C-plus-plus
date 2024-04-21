/**
 * @file 05_arrayTypes.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstddef>
#include <cstdio>
#include <iterator>

/**
 * @brief 数组 Arrays are sequences of identically typed variables. 
 * 数组是相同类型变量的序列, 数组的类型包括它所包含的元素的类型和数量;
 * 在声明语法中可以把这些信息组织在一起: 元素类型在方括号前面, 数组大小在方括号中间.
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    /** ========= 数组初始化
     * 有一种使用大括号初始化数组值的快捷方式:
     * 可以省略数组的长度, 因为它可以在编译时从大括号中的元素数量推断出来. 
     * 
     * ======== 访问数组元素
     * 使用方括号包围所需的索引即可访问数组元素,
     * 在C++中, 数组的索引是从零开始的, 所以第个元素的索引是0.
     * 
     */
    int arr[] = {1, 2, 3, 4};
    printf("The third element is %d.\n", arr[2]);
    arr[2] = 100;
    printf("The third element is %d.\n", arr[2]);

    /**
     * @brief ======== for循环简介
     * for 循环可以重复(或迭代)执行某些语句特定次数,可以规定一个起点和其他条件,
     * init statement(初始化语句)在第一次迭代之前执行, 所以它可以初始化 for 循环中使用的变量,
     * conditional 是一个表达式, 在每次迭代前被求值, 
     *  如果它被评估为 true, 迭代继续进行; 如果为false, for 循环就会终止.
     * iteration-statement 在每次迭代后执行, 这在必须递增变量以覆盖一个数值范围的情况下很有用
     * for(init-statement; conditional; iteration-statement)
     * {
     *   //--snip--
     * }
     *
     */
    unsigned long maximum  = 0;
    unsigned long values[] = {10, 50, 20, 40, 0};
    for (size_t idx = 0; idx < 5; ++idx)
    {
        if (values[idx] > maximum)
            maximum = values[idx];
    }
    printf("The maximum value is %lu\n", maximum);

    /**
     * @brief ======== 基于范围的for循环 for-range
     * 通过基于范围(range-based)的 for 循环来消除迭代器变量idx,
     * 对于像数组这样的特定对象, for 知道如何在对象中的值的范围内进行迭代
     * for(element-type element-name : array-name)
     * {
     *   //--Snip--
     * }
     * 声明迭代器变量 element-name 的类型为 element-type,
     * element-type 必须与要迭代的数组内的元素类型相匹配.
     * 
     */
    // for (const auto &elem : values)
    for (unsigned long elem : values)
    {
        if (elem > maximum)
            maximum = elem;
    }
    printf("The maximum value is %lu\n", maximum);

    /**
     * @brief ======== 数组中元素的数量
     * 使用 sizeof 运算符可以获得数组的总大小(以字节为单位),
     * 可以使用一个简单的技巧来确定数组的元素数: 用数组的大小除以单个元素的大小;
     * 该计算发生在编译时, 所以以这种方式评估数组的长度没有运行时成本.
     * sizeof(x)/sizeof(y) 构造太过于偏重技巧, 它被广泛用于旧代码中.
     *
     * std:size 函数安全地获得元素的数量, std:size 可以与任何暴露了 size 方法的容器一起使用
     * 
     */
    short  array_[]     = {104, 105, 32, 98, 105, 108, 108, 0};
    size_t num_elements = sizeof(array_) / sizeof(short);
    printf("the size of array: %zd\n", num_elements);
    printf("the size of array: %zd\n", std::size(array_));

    printf("\n========= C-Style Strings =========\n");
    /**
     * @brief C风格字符串 C-Style Strings
     * 字符串是由字符组成的连续序列, C风格的字符串或null结尾字符串会在未尾附加一个零(null),
     * 以表示字符串结束了, 因为数组元素是连续的, 所以可以在字符类型的数组中存储字符串.
     * 
     * ----- String Literals 字符串字面量
     * 用引号("")括住文本即可声明字符串字面量, 像字符字面量一样, 字符串字面量也支持Unicode,
     * 只要在前面加上适当的前缀, 如L.
     * NOTE: 使用字符串字面量: printf 语句的格式化字符串便是字符串字面量.
     * 
     * ------ 窄字符串(char*)的格式指定符是 %s,
     * 将字符串纳入格式化字符串
     * NOTE: 将Unicode打印到控制台出乎意料得复杂, 通常情况下需要确保选择了正确的代码页.
     * 
     */
    char english[] = "A book holds a house of gold.\n";
    printf("%s", english);
    // char16_t chinese[] = u"\u4e66\u4e2d\u81ea\u6709\u9ec4\u91d1\u5c4b";
    // printf("%s", chinese);

    /**
     * @brief 连续的字符串字词会被串联在一起, 任何中间的空白或换行符都会被忽略,
     * 可以在源代码中将字符串字面量分多行放置, 编译器会将它们视为一个整体.
     * 通常情况下, 只有当字符串字面量很长, 在源代码中会跨越多行时, 这样的结构才有利于提高可读性.
     * 
     */
    char house[]
        = "a "
          "house "
          "of "
          "gold.";
    printf("A book holds %s\n ", house);

    printf("\n==== Printing the letters of the alphabet using ASCII ====\n");
    /**
     * @brief ASCII美国信息交换标准代码(American Standard Code for Information Interchange)
     * ASCII 表将整数与字符一一匹配, 于十进制(0d)和十六进制(0x)的整数值, 表中都给出了控制代码或可打印字符.
     * ASCII代码0~31是控制设备的控制代码字符,控制代码有:
     * -- 0(NULL), 被编程语言用作字符串的结束符;
     * -- 4(EOT), EOT意指End Of Transmission, 即传输结束,终止shell会话和与PostScript打印机的通信;
     * -- 7(BELL), 使设备发出声音;
     * -- 8(BS), BS意指BackSpace, 即退格, 使设备擦除最后一个字符;
     * -- 9(HT), HT意指Horizontal Tab, 即水平制表符, 将光标向右移动几个空格;
     * -- 10(LF), LF意指Line Feed, 即换行, 在大多数操作系统中被用作行末标记;
     * -- 13(CR), (CR意指Carriage Return, 即回车, 在Windows系统中与LF结合使用, 作为行末标记;
     * -- 26(SUB), 指替代字符(SUBstitute character)、文件结束和<Ctrl+Z>, 在大多数操作系统上暂停当前执行的交互式进程.
     * 
     * 1. 声明一个长度为27的 char 数组来存放26个英文字母和一个null结尾符0;
     * 2. 采用 for 循环, 使用迭代器变量idx从0到25进行迭代, 
     *    字母a的ASCI值为97, 在迭代器变量idx上添加97, 生成小写字母表 alphabet,
     * 3. 要使 alphabet 成为以null结尾的字符串, 需要将alphabet[26]设置为0.
     * 4. 打印大写字母表, 字母A的ASCII值是65, 所以相应地重置了字母表的每个元素并再次调用 printf.
     * 
     */
    char alphabet[27];
    for (size_t idx = 0; idx < 26; ++idx)
    {
        alphabet[idx] = idx + 97;
    }
    alphabet[26] = 0;
    printf("==== Printing the letters of the alphabet in lowercase ====\n");
    printf("%s\n", alphabet);

    printf("==== Printing the letters of the alphabet in uppercase ====\n");
    for (size_t idx = 0; idx < 26; ++idx)
    {
        alphabet[idx] = idx + 65;
    }
    printf("%s\n", alphabet);

    return 0;
}
