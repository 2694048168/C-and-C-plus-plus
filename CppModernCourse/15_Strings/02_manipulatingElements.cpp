/**
 * @file 02_manipulatingElements.cpp
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
 * @brief 操作元素 Manipulating Elements
 * ?1. 添加元素 Adding Elements
 * *push_back;
 * *operator+=;
 * *append;
 * 
 * ?2. 删除元素 Removing Elements
 * *pop_back;
 * *clear;
 * *erase;
 * 
 * ?3. 替换元素 Replacing Elements
 * 要同时插入和删除元素,请使用 std::string 暴露的 replace 方法,该方法有很多重载
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("\nstd::string supports appending with\n");
    std::string word{"butt"};
    // push_back
    word.push_back('e');
    assert(word == "butte");

    // operator+=
    word += "erfinger";
    assert(word == "butteerfinger");

    // append char and char*
    word.append(1, 's');
    assert(word == "butteerfingers");
    word.append("stockings", 5);
    assert(word == "butteerfingersstock");

    // append (half-open range)
    std::string other("onomatopoeia");
    word.append(other.begin(), other.begin() + 2);
    printf("the last string: %s\n", word.c_str());

    printf("\nstd::string supports removal with\n");
    std::string word2("therein");
    // pop_back
    word2.pop_back();
    word2.pop_back();
    assert(word2 == "there");

    // clear
    word2.clear();
    assert(word2.empty());

    // erase using half-open range
    word2 = "therein";
    word2.erase(word2.begin(), word2.begin() + 3);
    assert(word2 == "rein");

    // erase using an index and length
    word2.clear();
    word2 = "therein";
    word2.erase(5, 2);
    assert(word2 == "there");
    printf("the last string after remove: %s\n", word2.c_str());

    printf("\nstd::string replace works with\n");
    std::string word3("substitution");
    // a range and a char*
    word3.replace(word3.begin() + 9, word3.end(), "e");
    assert(word3 == "substitute");

    // two ranges
    word3.clear();
    word3 = "substitution";
    std::string other_("innuendo");
    word3.replace(word3.begin(), word3.begin() + 3, other_.begin(), other_.begin() + 2);
    assert(word3 == "institution");
    printf("the replace string after: %s\n", word3.c_str());

    // an index/length and a string
    word3.clear();
    word3 = "substitution";
    std::string other__("vers");
    word3.replace(3, 6, other__);
    assert(word3 == "subversion");

    // TODO: std::string 类提供了一个 resize 方法,可以用来手动设置字符串的长度
    printf("\nstd::string resize\n");
    std::string word4("shamp");
    // can remove elements
    word4.resize(4);
    assert(word4 == "sham");

    // can add elements
    word4.resize(7, 'o');
    assert(word4 == "shamooo");
    printf("the resize string after: %s\n", word4.c_str());

    // TODo:可以使用 substr 方法生成子字符串, 该方法接受两个可选参数: 位置参数和长度参数;
    // 位置参数默认为 0(字符串的开头), 长度参数默认为字符串的剩余部分;
    printf("\nstd::string substr with\n");
    std::string word5("hobbits");
    // "no arguments copies the string"
    assert(word5.substr() == "hobbits");

    // position takes the remainder
    assert(word5.substr(3) == "bits");

    // position/index takes a substring
    assert(word5.substr(3, 3) == "bit");

    return 0;
}
