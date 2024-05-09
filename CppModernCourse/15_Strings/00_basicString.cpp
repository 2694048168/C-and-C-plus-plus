/**
 * @file 00_basicString.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <cstdio>
#include <string>

/**
 * @brief STL 为人类语言(单词、句子和标记语言)提供了一个特殊的字符串容器,
 * 头文件＜string＞ 中的 std::basic_string 是一个专注于字符串的基础字符类型的类模板,
 * 作为一个顺序容器, basic_string 本质上类似于向量, 但是提供了一些用于处理语言的工具.
 * STL basic_string 与 C 风格的字符串或以空字符结尾的字符串相比, 在安全性和功能上均有重大改进,
 * 并且在现代程序中, 人类语言数据非常多, 所以可能会发现 basic_string 是必不可少的. 
 * 
 * ====STL 在头文件＜string＞中提供了四种 basic_string 特化:
 * *1. char 的 std::string 用于 ASCII 之类的字符集;
 * *2. wchar_t 的 std::wstring 足够大, 可以包含实现语言环境的最大字符;
 * *3. char16_t 的 std::u16string 用于 UTF-16 之类的字符集;
 * *4. char32_t 的 std::u32string 用于 UTF-32 之类的字符集;
 * ?可以用适当的基础类型来特化上面的四种字符串, 因为这几种对于特定基础字符类型专门实现的字符串都有相似的接口
 * 
 * ====构造字符串
 * basic_string 容器使用三个模版参数:
 * 1. 基础字符类型 T;
 * 2. 基础类型的特征 Traits;
 * 3. 内存分配器 Alloc;
 * *std::basic_string<T, Traits=std::char_traits<T>, Alloc=std::allocator<T>>
 * STL 中的 std::char_traits＜T＞ 模版类根据基础类型 T 提取对应的字符和字符串操作,
 *  除非想要自定义字符类型,否则不需要实现自己的类型特征, 因为 char_traits＜T＞ 对于
 *  char、wchar_t、char_16 和 char_32 有专门的实现.
 * 
 * ====字符串存储和小字符串优化 String Storage and Small String Optimizations
 * 就像 vector 一样, std::string 使用动态存储空间来连续存储其中的元素, 
 * *因此vector和 std::string 具有非常相似的复制/移动构造/赋值语义,
 * 最受欢迎的 STL 实现有小字符串优化(Small String Optimization, SSO),
 * *如果字符串足够小的话, SSO 会将字符串的内容放在对象的储存区域[static](而不是动态存储空间heap),
 * *小于 24字节 的字符串通常会被 SSO 实现优化, 实现者做这个优化的原因是现在大多数程序里面的字符串都是比较小的.
 * 
 */

// ----------------------------------
int main(int argc, const char **argv)
{
    printf("\nstd::string supports constructing\n");
    // empty strings
    std::string cheese;
    assert(cheese.empty());

    // repeated characters
    std::string roadside_assistance(3, 'A');
    assert(roadside_assistance == "AAA");

    printf("the empty strings: %s\n", cheese.c_str());
    printf("the repeated characters: %s\n", roadside_assistance.c_str());

    /** ====从 C 风格字符串构造 string
     * @brief 字符串构造函数还提供了两个基于 const char* 的构造函数,
     * 如果参数指向以空字符结尾的字符串, 则字符串构造函数可以自行确定输入的字符串的长度,
     * 如果指针没有指向以空字符结尾的字符串, 或者只想使用字符串的前面部分, 
     * 则可以传递一个长度参数, 告诉字符串构造函数要复制几个元素.
     */
    printf("\nstd::string supports constructing substrings\n");
    auto word = "gobbledygook"; // const char*
    assert(std::string(word) == "gobbledygook");
    assert(std::string(word, 6) == "gobble");

    /**
     * @brief 作为 STL 容器, string 完全支持复制语义和移动语义.
     * 可以从子字符串(另一个字符串的连续子集)构造字符串.
     */
    printf("\nstd::string supports copy&move or substrings constructing\n");
    std::string word_("catawampus");
    assert(std::string(word_) == "catawampus"); //copy constructing
    // 一个起始位置参数 0 和长度参数 3
    assert(std::string(word_, 0, 3) == "cat"); //constructing from substrings
    // 一个起始位置参数 0 和长度参数直到最后
    assert(std::string(word_, 4) == "wampus");
    assert(std::string(std::move(word_)) == "catawampus"); //move constructing

    /**
     * @brief std::string 还支持使用 std::string_literals::operator""s 
     * 进行字符串字面量构造, 主要的好处是符号化更方便,
     * 也可以使用 operator""s 将空字符轻松地嵌入字符串中
     */
    printf("\nconstructing a string with\n");
    //  std::string(char*) stops at embedded nulls
    std::string str("idioglossia\0hello hay!");
    assert(str.length() == 11);
    printf("the string constructing stops at null: %s\n", str.c_str());
    // "operator\"\"s incorporates embedded nulls"
    using namespace std::string_literals;
    auto str_lit = "idioglossia\0hello hay!"s;
    assert(str_lit.length() == 22);
    printf("the string constructing string_literals: %s\n", str_lit.c_str());

    return 0;
}
