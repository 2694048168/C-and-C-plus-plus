/**
 * @file 01_pairUtility.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <cstdio>
#include <utility>

/**
 * @brief pair
 * pair 是一个类模板, 一个 pair 对象中可包含两个不同类型的对象;
 * 对象是有序的, 可以通过成员 first 和 second 访问它们;
 * pair 支持比较运算符, 有默认的复制构造函数和移动构造函数, 使用结构化绑定语法.
 * 
 * *stdlib 的＜utility＞头文件中有 std::pair
 * 
 * 
 */

struct Socialite
{
    const char *birth_name;
};

struct Valet
{
    const char *surname;
};

// -----------------------------------
int main(int argc, const char **argv)
{
    Socialite bertie{"Wilberforce"};
    Valet     reginald{"Jeeves"};

    printf("std::pair permits access to members\n");
    std::pair<Socialite, Valet> inimitable_duo{bertie, reginald};
    assert(inimitable_duo.first.birth_name == bertie.birth_name);
    assert(inimitable_duo.second.surname == reginald.surname);

    printf("the value of first: %s, and second: %s\n\n", inimitable_duo.first.birth_name,
           inimitable_duo.second.surname);

    // std::pair member extraction and structured binding syntax
    printf("std::pair member extraction and structured binding syntax\n");
    std::pair<Socialite, Valet> inimitable_{bertie, reginald};

    auto &[idle_rich, butler] = inimitable_;
    assert(idle_rich.birth_name == bertie.birth_name);
    assert(butler.surname == reginald.surname);
    printf("the value of first: %s, and second: %s\n\n", idle_rich.birth_name, butler.surname);

    return 0;
}
