/**
 * @file 00_raw_string_literal.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-14
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <string>

/** C++11 中添加了定义原始字符串的字面量,
 * 定义方式为：R “xxx(原始字符串)xxx”, 其中（）两边的字符串可以省略.
 * 原始字面量R可以直接表示字符串的实际含义,而不需要额外对字符串做转义或连接等操作.
*/
// -----------------------------
int main(int argc, char **argv)
{
    // the file path on Windows
    std::string filepath = "D:\Development\GitRepository\ModernGuideCpp17";
    std::cout << "the file path: " << filepath << "\n";

    std::string filepath_escape = "D:\\Development\\GitRepository\\ModernGuideCpp17";
    std::cout << "the file path: " << filepath_escape << "\n";

    std::string filepath_raw = R"(D:\Development\GitRepository\ModernGuideCpp17)";
    std::cout << "the file path: " << filepath_raw << "\n";

    // the multiple line string for readable
    std::string multi_line_str = "<html>\
        <head>\
        <title>\
        Wei Li\
        </title>\
        </head>\
        <body>\
        <p>\
        https://github.com/2694048168\
        </p>\
        </body>\
        </html>";

    std::cout << "the multi-line info: " << multi_line_str << "\n";

    std::string multi_line_str_raw = R"(<html>
        <head>
        <title>
        Wei Li
        </title>
        </head>
        <body>
        <p>
        https://github.com/2694048168
        </p>
        </body>
        </html>)";

    std::cout << "the multi-line info: " << multi_line_str_raw << "\n";

    return 0;
}