/**
 * @file 03_searchManipulation.cpp
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
 * @brief std::string 还提供了几种搜索方法,
 * 使用这些方法能够定位感兴趣的子字符串和字符
 * ?1. find 方法
 * ?2. rfind 方法bin
 * ?3. find_＊_of 方法
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief std::string 提供的第一个方法是 find,
     * 它将 string, C-style 的字符串或 char 作为第一个参数,
     * 此参数代表要在字符串中定位的元素; 也可以提供第二个 size_t 类型位置参数, 告诉从哪里开始查找;
     * !如果 find 未能找到子字符串, 则返回特殊的size_t 值、常量、静态成员 std::string::npos
     */
    printf("\nstd::string find\n");
    using namespace std::literals::string_literals;
    std::string word("pizzazz");
    // locates substrings from strings
    assert(word.find("zz"s) == 2); // pi(z)zazz

    // accepts a position argument
    assert(word.find("zz"s, 3) == 5); // pizza(z)z

    // locates substrings from char*
    assert(word.find("zaz") == 3); // piz(z)azz

    // returns npos when not found
    assert(word.find('x') == std::string::npos);

    /**
     * @brief rfind 方法是 find 的另一种版本,
     * 它采用相同的参数但搜索方向是反的.
     * 例如如果正在寻找字符串末尾的特定标点符号, 那么可能想要使用此功能.
     */
    printf("\nstd::string rfind\n");
    // locates substrings from strings
    assert(word.rfind("zz"s) == 5); // pizza(z)z

    // accepts a position argument
    assert(word.rfind("zz"s, 3) == 2); // pi(z)zazz

    // locates substrings from char*
    assert(word.rfind("zaz") == 3); // piz(z)azz
    // returns npos when not found
    assert(word.rfind('x') == std::string::npos);

    /**
     * @brief find 和 rfind 定位字符串中确切的子序列,
     * *而一系列相关函数则查找给定参数中包含的第一个字符;
     * find_first_of 函数接受一个字符串并定位参数中包含的第一个字符;
     * 同样,也可以可选地提供一个 size_t 位置参数来指示 find_first_of 从字符串的哪里开始;
     * 如果 find_first_of 找不到匹配的字符, 则返回 std::string::npos
     */
    printf("\nstd::string find_first_of\n");
    std::string sentence("I am a Zizzer-Zazzer-Zuzz as you can plainly see.");
    // locates characters within another string
    assert(sentence.find_first_of("Zz"s) == 7); // (Z)izzer

    // accepts a position argument
    assert(sentence.find_first_of("Zz"s, 11) == 14); // (Z)azzer
    // returns npos when not found
    assert(sentence.find_first_of("Xx"s) == std::string::npos);

    /** std::string 提供了三种 find_first_of 变体:
    * 1. find_first_not_of 返回未包含在字符串参数中的第一个字符, 有时候与其提供包含要查找元素的字符串,
    *    不如提供包含不想查找字符的字符串;
    * 2. find_last_of 反向执行匹配, 它不是从字符串的开头或位置参数开始搜索到结尾,
    *    而是从字符串的末尾或位置参数开始搜索到开头;
    * 3. find_last_not_of 结合了上述两个变体的功能: 传递一个包含不想查找元素的字符串, 然后反向搜索;
    */
    // find_last_of finds last element within another string
    assert(sentence.find_last_of("Zz"s) == 24); // Zuz(z)

    // find_first_not_of finds first element not within another string
    assert(sentence.find_first_not_of(" -IZaeimrz"s) == 22); // Z(u)zz

    // find_last_not_of finds last element not within another string
    assert(sentence.find_last_not_of(" .es"s) == 43); // plainl(y)

    return 0;
}
