/**
 * @file 05_stringView.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-10
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <cstdio>
#include <string>
#include <string_view>

/**
 * @brief 字符串视图 String View
 * 字符串视图(string view)是一个对象, 表示一个恒定的、连续的字符序列.
 * 它与 const 字符串引用非常相似, 事实上, 字符串视图类通常实现为指向字符序列的指针和长度
 * STL 在＜string_view＞头文件中提供了类模板 std::basic_string_view,
 *  *它类似于 std::basic_string, 事实上它的设计目的是替代 const string&
 * 
 */

// 它计算字母 v 在字符序列中出现的频率
size_t count_vees(std::string_view my_view)
{
    size_t result{};
    for (auto letter : my_view)
        if (letter == 'v')
            result++;
    return result;
}

/** 其实, 如果用 std::string 调用 count_vees, 那确实没有区别, 现代编译器会生成相同的代码.
 * @brief 如果用字符串字面量来调用 count_vees, 就会有很大的不同,
 * 当为 const string& 传递字符串字面量时, 就会构造一个 std::string;
 * 当为 string_view 传递字符串字面量时, 将构造一个 std::string_view;
 * 构造 std::string 的代价是比较高的, 因为它必须动态分配内存并复制字符.
 * 而 std::string_view 只包含指针和长度(不需要复制和分配)
 */
size_t count_vees_(const std::string &my_view)
{
    size_t result{};
    for (auto letter : my_view)
        if (letter == 'v')
            result++;
    return result;
}

// ------------------------------------
int main(int argc, const char **argv)
{
    printf("\nstd::string_view supports\n");
    // default construction
    std::string_view view;
    assert(view.data() == nullptr);
    assert(view.size() == 0);
    assert(view.empty());

    // construction from string
    std::string      word{"sacrosanct"};
    std::string_view view2(word);
    assert(view2 == "sacrosanct");

    // construction from C-string
    auto             word3 = "viewership";
    std::string_view view3(word3);
    assert(view3 == "viewership");

    // construction from C-string and length
    std::string_view view4(word3, 4);
    assert(view4 == "view");

    printf("\nstd::string_view is modifiable with\n");
    std::string_view view5("previewing");
    // remove_prefix
    view5.remove_prefix(3);
    assert(view5 == "viewing");
    // remove_suffix
    view5.remove_suffix(3);
    assert(view5 == "view");

    /**
     * @brief 所有权、用法和效率 Ownership, Usage, and Efficiency
     * string_view 最常见的用法可能是作为函数参数,
     * *当需要与不可变的字符序列进行交互时, 它应是首要选择.
     */
    printf("the count of str is %lld\n", count_vees("view"));

    return 0;
}
