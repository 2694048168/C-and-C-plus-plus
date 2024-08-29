/**
 * @file 00_raw_string.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-08-28
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 在C++11中添加了定义原始字符串的字面量,
 * 定义方式为：R "xxx(原始字符串)xxx" 其中（）两边的字符串可以省略.
 * 原始字面量R可以直接表示字符串的实际含义, 而不需要额外对字符串做转义或连接等操作.
 * 比如编程过程中，使用的字符串中常带有一些特殊字符，对于这些字符往往要做专门的处理，
 * 使用了原始字面量就可以轻松的解决这个问题了，比如打印路径：
 *
 */

#include <iostream>
#include <string>

// ------------------------------------
int main(int argc, const char **argv)
{
    // C++ '\' 这个字符又是转义字符
    // \h和\w转义失败，对应的字符会原样输出
    std::string str = "D:\hello\world\test.text";
    std::cout << str << std::endl;

    // 路径的间隔符为\但是这个字符又是转义字符，因此需要使用转义字符将其转义，
    // 最终才能得到一个没有特殊含义的普通字符 '\'
    std::string str1 = "D:\\hello\\world\\test.text";
    std::cout << str1 << std::endl;

    // 路径的原始字符串，无需做任何处理
    std::string str2 = R"(D:\hello\world\test.text)";
    std::cout << str2 << std::endl << std::endl;

    // 在C++11之前如果一个字符串分别写到了不同的行里边，需要加连接符 '\'
    std::string str_html
        = "<html>\
        <head>\
        <title>\
        海贼王\
        </title>\
        </head>\
        <body>\
        <p>\
        我是要成为海贼王的男人!!!\
        </p>\
        </body>\
        </html>";
    std::cout << str_html << std::endl;

    std::string str_html2 = R"(<html>
        <head>
        <title>
        海贼王
        </title>
        </head>
        <body>
        <p>
        我是要成为海贼王的男人!!!
        </p>
        </body>
        </html>)";
    std::cout << str_html2 << std::endl;

    /* 使用原始字面量R "xxx(raw string)xxx",
    （）两边的字符串在解析的时候是会被忽略的, 因此一般不用指定.
    如果在（）前后指定了字符串，那么前后的字符串必须相同，否则会出现语法错误. */
    std::string str_ignore = R"Ithaca(D:\hello\world\test.text)Ithaca";
    std::cout << std::endl << str_ignore << std::endl;

    return 0;
}
