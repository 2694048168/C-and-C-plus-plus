/**
 * @file 05_string_number.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-08-28
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 在C++11中提供了专门的类型转换函数,使用它们进行数值类型和字符串类型之间的转换
 * 1. 数值转换为字符串
 *  使用to_string()方法可以非常方便地将各种数值类型转换为字符串类型,这是一个重载函,
 *  函数声明位于头文件<string>中, support int,long,long long,unsigned,unsigned long,
 *  unsigned long long, float, double, long double.
 *
 * 2. 字符串转换为数值
 *  由于C++中的数值类型包括整形和浮点型, 因此针对于不同的类型提供了不同的函数,
 *  通过调用这些函数可以将字符串类型转换为对应的数值类型.
// 定义于头文件 <string>
1. int stoi( const std::string& str, std::size_t* pos = 0, int base = 10 );
2. long stol( const std::string& str, std::size_t* pos = 0, int base = 10 );
3. long long stoll( const std::string& str, std::size_t* pos = 0, int base = 10 );
4. unsigned long stoul( const std::string& str, std::size_t* pos = 0, int base = 10 );
5. unsigned long long stoull( const std::string& str, std::size_t* pos = 0, int base = 10 );
6. float stof( const std::string& str, std::size_t* pos = 0 );
7. double stod( const std::string& str, std::size_t* pos = 0 );
8. long double stold( const std::string& str, std::size_t* pos = 0 );
 *
 * ?str：要转换的字符串
 * ?pos：传出参数, 记录从哪个字符开始无法继续进行解析, 比如: 123abc, 传出的位置为3
 * ?base：若 base 为 0 ，则自动检测数值进制：若前缀为 0 ，则为八进制，
 * ?若前缀为 0x 或 0X，则为十六进制，否则为十进制。
 * 这些函数虽然都有多个参数，但是除去第一个参数外其他都有默认值，一般情况下使用默认值就能满足需求
 *
 */

#include <iostream>
#include <string>

// -----------------------------------
int main(int argc, const char **argv)
{
    std::string pi   = "pi is " + std::to_string(3.1415926);
    std::string love = "love is " + std::to_string(5.20 + 13.14);
    std::cout << pi << std::endl;
    std::cout << love << std::endl << std::endl;

    std::string str1 = "45";
    std::string str2 = "3.14159";
    std::string str3 = "9527 with words";
    std::string str4 = "words and 2";

    int   my_int1 = std::stoi(str1);
    float my_int2 = std::stof(str2);
    int   my_int3 = std::stoi(str3);
    // !runtime error: 'std::invalid_argument'
    // int   my_int4 = std::stoi(str4);

    std::cout << "std::stoi(\"" << str1 << "\") is " << my_int1 << std::endl;
    std::cout << "std::stof(\"" << str2 << "\") is " << my_int2 << std::endl;
    std::cout << "std::stoi(\"" << str3 << "\") is " << my_int3 << std::endl;
    // std::cout << "std::stoi(\"" << str4 << "\") is " << my_int4 << std::endl;

    // 如果字符串中所有字符都是数值类型，整个字符串会被转换为对应的数值，并通过返回值返回
    // 如果字符串的前半部分字符是数值类型，后半部不是，那么前半部分会被转换为对应的数值，并通过返回值返回
    // 如果字符第一个字符不是数值类型转换失败

    return 0;
}
