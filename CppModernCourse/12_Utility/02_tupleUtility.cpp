/**
 * @file 02_tupleUtility.cpp
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
#include <tuple>

/**
 * @brief tuple
 * tuple(元组)是一个类模板, 包含任意数量的不同类型元素, 这是对 pair 的泛化,
 * 但 tuple 不会像 pair 那样将其成员公开为 first、second, 
 * 相反需要使用非成员函数模板 get 来提取元素.
 * 
 * *stdlib 的＜tuple＞头文件中有 std::tuple 和 std::get
 * 
 */

struct Acquaintance
{
    const char *nickname;
};

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
    Socialite    bertie{"Wilberforce"};
    Valet        reginald{"Jeeves"};
    Acquaintance hildebrand{"Tuple"};

    printf("std::tuple permits access to members with std::get\n");
    using Trio = std::tuple<Socialite, Valet, Acquaintance>;
    Trio truculent_trio{bertie, reginald, hildebrand};

    auto &bertie_ref = std::get<0>(truculent_trio);
    assert(bertie_ref.birth_name == bertie.birth_name);
    printf("the value: %s", bertie_ref.birth_name);

    auto &tuple_ref = std::get<Acquaintance>(truculent_trio);
    assert(tuple_ref.nickname == hildebrand.nickname);
    printf("the value: %s", tuple_ref.nickname);

    // std::tuple 支持成员提取和结构化绑定语法
    // 和 std::pair 一样, std::tuple 也支持结构化绑定语法

    return 0;
}
