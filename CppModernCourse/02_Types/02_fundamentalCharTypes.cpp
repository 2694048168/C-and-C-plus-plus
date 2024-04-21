/**
 * @file 02_fundamentalCharTypes.cpp
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
  * @brief 字符类型: 字符类型存储人类语言信息, 六种字符类型是:
  * 1. char: 默认类型, 总是1个字节, 可能是也可能不是有符号的(例如:ASCII);
  * 2. char16_t: 用于2字节的字符集(例如:UTF-16);
  * 3. char32_t: 用于4字节的字符集(例如:UTF-32);
  * 4. signed char: 与 char 相同, 但保证是有符号的;
  * 5. unsigned char: 与 char 相同, 但保证是无符号的;
  * 6. wchar_t: 足够大以包含实现平台地区环境语言设置中的最大字符(例如:Unicode).
  *
  * 字符类型 char、signed char 和 unsigned char 被称为容字符,
  * 而 char16_t、char32_t 和 wchar_t 由于其相对的存储要求, 被称为宽字符.
  * 
  * 2. 转义序列, 有些字符不能在屏幕上显示, 相反它们会迫使显示器做一些事情,
  * 比如将光标移到屏幕的左边(回车)或将光标向下移动一行(换行);
  * 其他字符可以在屏幕上显示, 但它们是C++语法的一部分, 如单引号或双引号, 所以必须非常小心地使用;
  * 为了将这些字符转换为 char, 可以使用转义序列,
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
  * 
  */

// ------------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief 1. 字符字面量
     * 字符字面量是一个单一的、恒定的字符,
     * 所有字符都用单引号(")括起来, 如果字符是 char 以外的其他类型,
     * 还必须提供一个前缀: L代表 wchar_t, u代表 char16_t, U代表 char32_t.
     * 
     */
    char     ch_1 = 'A';
    wchar_t  ch_2 = L'A';
    // char16_t ch_3 = u'A';
    // char32_t ch_4 = U'A';
    printf("the char value %c, and size %llu\n", ch_1, sizeof(ch_1));
    printf("the char value %c, and size %llu\n", ch_2, sizeof(ch_2));
    // char16_t and char32_t are nothing special.
    // printf("the char value %s, and size %llu\n", ch_3, sizeof(ch_3));
    // printf("the char value %s, and size %llu\n", ch_4, sizeof(ch_4));

    /**
    * @brief 3. Unicode转义字符
    * 可以使用通用字符名(universal character name)来指定Unicode字符字面量,
    * 使用通用字符名的方式有两种: 前缀\u加后面4位的Unicode码位, 或前缀\U加后面8位的Unicode码位;
    *
    */
    // auto ch_5 = 'u0041';
    // auto ch_6 = 'U0001F37A';
    // printf("the char value %s, and size %llu\n", ch_5, sizeof(ch_5));
    // printf("the char value %s, and size %llu\n", ch_6, sizeof(ch_6));

    /**
     * @brief 4.格式指定符
     * char 的 printf 格式指定符为 %c;
     * wchar_t 的格式指定符是 %lc;
     * NOTE: 所有Windows二进制文件的前两个字节是字符M和Z,
     *   这是对MS-DOS可执行二进制文件格式的设计者 Mark Zbikowski 的致敬.
     * 
     */
    char    x = 'M';
    wchar_t y = L'Z';
    printf("Windows binaries start with %c%lc.\n", x, y);

    return 0;
}
