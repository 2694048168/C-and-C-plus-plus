/**
 * @file 01_string.cpp
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
#include <string>

/**
 * @brief 元素和迭代器访问 Element and Iterator Access
 * std::string 为连续的元素提供了随机访问的迭代器, 
 * 所以它相应地向 vector 公开了相似的元素访问和迭代器访问方法,
 * *为了与 C 风格的 API 互相操作, std::string 还公开了 c_str 方法,
 * 该方法以 const char* 的形式返回以空字符结尾的字符串的只读版本.
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("\nstring's c_str method makes null-terminated strings\n");
    std::string word("horripilation");

    auto as_cstr = word.c_str();
    assert(as_cstr[0] == 'h');
    assert(as_cstr[1] == 'o');
    assert(as_cstr[11] == 'o');
    assert(as_cstr[12] == 'n');
    assert(as_cstr[13] == '\0');

    /**
     * @brief 一般而言, c_str 和 data 返回的结果相同,
     * 只是 data 返回的引用可以是非 const的,
     * 每当操作字符串时, 实现通常都会确保支持以空字符结尾的字符串的连续内存.
     */
    std::string word_("pulchritudinous");
    printf("c_str: %s at 0x%p\n", word_.c_str(), word_.c_str());
    printf("data: %s at 0x%p\n", word_.data(), word_.data());

    /**
     * @brief 字符串比较 String Comparisons
     * std::string 支持使用常规比较运算符与其他字符串和原始 C 风格字符串进行比较;
     * ?1. 如果运算符左右的大小和内容都相等, 则等于 operator== 返回true, 而 operator!= 返回false;
     * ?2. 其余的比较运算符执行字典序比较, 这意味着它们按字母顺序排序, 其中 A ＜ Z ＜ a ＜ z;
     * ?3. 如果所有其他条件相同, 则短词小于长词(例如pal ＜ palindrome);
     */
    printf("\nstd::string supports comparison with\n");
    using namespace std::literals::string_literals;
    std::string word2("allusion");
    // "operator== and !="
    assert(word2 == "allusion");
    assert(word2 == "allusion"s);
    assert(word2 != "Allusion"s);
    assert(word2 != "illusion"s);

    // "operator＜"
    assert(word2 < "illusion");
    assert(word < "illusion"s);
    assert(word > "Illusion"s);

    return 0;
}
