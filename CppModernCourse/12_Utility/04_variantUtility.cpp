/**
 * @file 04_variantUtility.cpp
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
#include <variant>

/**
 * @brief variant
 * variant(变体)是存储单个值的类模板, 值的类型仅限于作为模板参数提供的用户自定义列表.
 * 该变体是类型安全的联合体, 它与 any 类型共享许多功能, 但 variant 要求显式枚举将要存储的所有类型
 * 
 * *stdlib 的＜variant＞头文件中有 std::variant
 * 
 * ====构造 variant
 * 如果满足以下两个条件之一, 则只能默认构造 variant:
 * 1. 第一个模板参数是默认可构造的;
 * 2. 它是 monostate 的, 一种表明 variant 具有空状态的类型;
 * 
 * ====要将值存储到 variant 中, 使用 emplace 方法模板, 
 * 与 any 一样, variant 也接受与要存储的类型相对应的单个模板参数,
 * 此模板参数必须包含在 variant 的模板参数列表中.
 * 要提取值, 可以使用非成员函数模板 get 或 get_if, 它们接受所需的类型或对应于所需类型的模板参数列表的索引,
 * 如果 get 失败, 则抛出 bad_variant_access 异常,
 * 而 get_if 失败则返回 nullptr;
 * *可以使用 index() 成员来确定哪种类型对应于 variant 的当前状态,
 * *该成员返回模板参数列表中当前对象类型的索引.
 * 
 */

struct BugblatterBeast
{
    BugblatterBeast()
        : is_ravenous{true}
        , weight_kg{20000}
    {
    }

    bool is_ravenous;
    int  weight_kg;
};

struct EscapeCapsule
{
    EscapeCapsule(int x)
        : weight_kg{x}
    {
    }

    int weight_kg;
};

// -----------------------------------
int main(int argc, const char **argv)
{
    // Constructing a variant
    printf("======== (std::variant) ========\n");
    std::variant<BugblatterBeast, EscapeCapsule> hagunemnon;

    assert(hagunemnon.index() == 0);
    hagunemnon.emplace<EscapeCapsule>(600);
    assert(hagunemnon.index() == 1);

    auto val = std::get<EscapeCapsule>(hagunemnon).weight_kg;
    assert(val == 600);
    printf("the value : %d\n", val);

    auto num = std::get<1>(hagunemnon).weight_kg;
    assert(num == 600);
    printf("the value : %d\n", num);

    try
    {
        auto ret = std::get<0>(hagunemnon);
        printf("the ret value: %d\n", ret.weight_kg);
    }
    catch (const std::bad_variant_access &exp)
    {
        printf("the throw exception: std::bad_variant_access, %s\n", exp.what());
    }
    catch (const std::exception &exp)
    {
        printf("the throw exception %s\n", exp.what());
    }

    // std::visit 允许将可调用对象应用于 std::variant 包含的类型
    printf("\nstd::visit 允许将可调用对象应用于 std::variant 包含的类型\n");
    auto lbs = std::visit([](auto &x) { return 2.2 * x.weight_kg; }, hagunemnon);
    assert(lbs == 1320);
    printf("the value: %f", lbs);

    /**
     * @brief 比较 variant 和 any
     * 宇宙足够大, 可以同时容纳 any 和 variant, 
     * 一般来说, 不能认为其中一种优于另一种, 因为每一种都有其优点和缺点.
     * any 更灵活, 它可以接受任意类型; 而 variant 只允许包含预定义类型的对象,
     * 它还主要避免了模板的使用, 因此通常更容易编程.
     * variant 不太灵活, 因此更安全; 使用 visit 功能, 可以在编译时检查操作的安全性;
     * 使用 any, 需要构建自己的类似 visit 的功能, 并且需要在运行时检查(例如, any_cast 的结果).
     * 最后, variant 可以比 any 有更高的性能, 尽管在包含的类型太大时, any 可以执行动态分配, 但 variant 不能.
     * 
     */

    return 0;
}
