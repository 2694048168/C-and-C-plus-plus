/**
 * @file 11_ownershipSmartPtrMoveSemantics.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <memory>

/**
 * @brief 所有权模型来管理动态对象的生命周期, 
 * 一旦没有智能指针拥有动态对象(引用计数的方式), 该对象就会被销毁, 
 * unique_ptr 就是一种这样的智能指针, 它模拟了独占的所有权.
 * 
 * 智能指针与普通的原始指针不同, 因为原始指针只是一个简单的内存地址,
 *  必须手动协调所有涉及该地址的内存管理, 但智能指针可以自行处理所有这些混乱的细节;
 *  用智能指针包装动态对象, 可以很放心, 一旦不再需要这个对象, 内存就会被适当地清理掉;
 *  编译器知道不再需要该对象了, 因为当它超出作用域时, 智能指针的析构函数会被调用.
 * 
 */
struct Foundation
{
    const char *founder;
};

struct Mutant
{
    // constructor sets foundation appropriately
    Mutant(std::unique_ptr<Foundation> foundation)
        : m_foundation(std::move(foundation))
    {
    }

    std::unique_ptr<Foundation> m_foundation;
};

// -----------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief 动态地分配了一个 Foundation, 而产生的 Foundation* 指针
     * 被传给 second_foundation 的构造函数, 使用大括号初始化语法.
     *  second_foundation 的类型是 unique_ptr, 
     *  它只是一个包裹着动态 Foundation 的RAII对象,
     *  当 second_foundation 被析构时, 动态 Foundation 就会适当地销毁.
     * 
     */
    std::unique_ptr<Foundation> second_foundation{new Foundation{}};
    // Access founder member variable just like a pointer:
    second_foundation->founder = "Wanda";
    std::cout << "the name of Foundation: " << second_foundation->founder << '\n';

    // ============ Move Semantics ============
    std::cout << "\n============ Move Semantics ============\n";
    /**
     * @brief 有时想转移对象的所有权, 这种情况很常见,
     * 例如使用 unique_ptr 时, 不能复制unique_ptr, 因为一旦某个副本被销毁,
     * 剩下的 unique_ptr 将持有对已删除对象的引用;
     * 与其复制对象, 不如使用C++的移动(move)语义将所有权从一个指针转移到另一个指针.
     * 
     * 创建 std::unique_ptr<Foundation>, 在使用它一段时间后,
     *  决定将所有权转移到一个 Mutant 对象, std::move 函数告诉编译器想转移所有权,
     *  在构造 the_mule 后, Foundation 的生命周期通过其成员变量与 the_mule 的生命周期联系在一起.
     * 
     */
    std::unique_ptr<Foundation> first_foundation{new Foundation{}};
    // ... use first_foundation
    first_foundation->founder = "first_foundation";
    std::cout << "the name of Foundation: " << first_foundation->founder << '\n';

    Mutant the_mule{std::move(first_foundation)};
    // ! second_foundation is in a 'moved-from' state
    // std::cout << "the name of Foundation: " << first_foundation->founder << '\n';
    // * the_mule owns the Foundation
    std::cout << "the name of Foundation: " << the_mule.m_foundation->founder << '\n';

    return 0;
}
