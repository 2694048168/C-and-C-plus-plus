/**
 * @file 04_numericConversions.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-09
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>

/**
 * @brief 数值转换 Numeric Conversions
 * *STL 提供了用于在 string/wstring 与基本数值类型之间进行转换的函数.
 * 给定一个数值类型, 可以使用 std::to_string 和 std::to_wstring 函数
 * 来生成它的string 或wstring 表示, 这两个函数对所有数值类型都有重载.
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("\nSTL string conversion function\n");
    using namespace std::literals::string_literals;
    assert("8675309"s == std::to_string(8675309));

    double time_consume = 34.21;

    std::string message = "the time consume " + std::to_string(time_consume) + " ms\n";
    printf("[INFO]%s", message.c_str());

    // "to_wstring"
    // 由于 double 类型固有的不准确性,单元测试在你的系统上可能会失败
    // assert(L"109951.1627776"s == std::to_wstring(109951.1627776));
    printf("the string value: %ls\n", std::to_wstring(109951.1627776).c_str());

    /**
     * @brief 将 string 或 wstring 转换为数值类型,
     * 每个数值转换函数的第一个参数都是一个包含数字的 string 或者 wstring;
     * 接着可以提供一个指向 size_t 的指针;
     * 如果提供了, 转换函数将写入它能够转换的最后一个字符的索引(如果它解码了所有字符,则写入输入字符串的长度),
     * 默认情况下, 此索引参数为nullptr, 在这种情况下, 转换函数不会写入索引;
     * 当目标类型是整数时, 可以提供第三个参数: 用于表明 string 编码的数字的进制的 int 参数;
     * 此参数是可选的, 默认为 10.
     * !如果无法执行转换, 则转换函数抛出 std::invalid_argument;
     * !如果转换后的值超出相应类型的范围, 则抛出 std::out_of_range;
     */
    printf("\nSTL string conversion function\n");
    assert(std::stoi("8675309"s) == 8675309);

    try
    {
        auto num = std::stoi("1099511627776"s);
        printf("the number is: %d\n", num);
    }
    catch (std::out_of_range &exp)
    {
        printf("the out of range exception: %s\n", exp.what());
    }
    catch (const std::exception &exp)
    {
        printf("the exception: %s\n", exp.what());
    }

    // stoul with all valid characters
    size_t     last_character{};
    const auto result = std::stoul("0xD3C34C3D"s, &last_character, 16);
    assert(result == 0xD3C34C3D);
    assert(last_character == 10);

    {
        size_t     last_character{};
        const auto result = std::stoul("42six"s, &last_character);
        assert(result == 42);
        assert(last_character == 2);
    }

    // std::stod("2.7182818"s) == Approx(2.7182818)
    double val = std::stod("2.7182818"s);
    printf("the approx value: %f\n", val);

    return 0;
}
