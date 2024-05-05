/**
 * @file 02_sharedPointers.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>
#include <memory>

/**
 * @brief 共享指针 Shared Pointers
 * 共享指针拥有对单个动态对象的可转移非专属所有权,
 * 共享指针可以移动, 这说明它们可以转移, 它们也可以复制, 这说明它们的所有权是非专属的.
 * 
 * 非专属所有权意味着 shared_ptr 在销毁对象之前需要检查是否还有其他 shared_ptr 对象拥有该对象,
 * 这样, 就由最后一个所有者负责释放拥有的对象; (引用计数方式)
 * 
 * TODO: 共享指针和独占指针之间的主要功能区别在于共享指针可以复制
 * 
 * ===== 弱指针 Weak Pointers 
 * 弱指针是一种特殊的智能指针, 它对所引用的对象的所有权;
 * *弱指针允许跟踪对象并仅在被跟踪对象仍然存在时将弱指针转换为共享指针, 这允许生成对象的临时所有权.
 * 像共享指针一样, 弱指针是可移动和可复制的.
 * *弱指针的一个常见用法是缓存, 缓存是一种临时存储数据以便可以更快地检索数据的数据结构;
 * 缓存可以保有指向对象的弱指针, 因此一旦所有其他所有者释放对象, 对象就会销毁;
 * 缓存可以定期扫描其存储的弱指针并去除那些没有其他所有者的弱指针.
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    // std::shared_ptr 指针支持所有与 std::unique_ptr 相同的构造函数
    std::shared_ptr<int> my_ptr{new int{808}};
    printf("the shared-pointer address: %p, and value: %d\n", my_ptr.get(), *my_ptr);

    auto my_ptr_ = std::make_shared<int>(88);
    printf("the shared-pointer address: %p, and value: %d\n", my_ptr_.get(), *my_ptr_);

    /**
     * @brief 指定分配器 Specifying an Allocator
     * 分配器负责分配,创建,销毁和释放对象; 
     * 默认分配器 std::allocator 是在 ＜memory＞头文件中定义的模板类;
     * 默认分配器从动态存储空间中分配内存并接受模板参数.
     * 
     * shared_ptr 构造函数和 make_shared 都有一个分配器类型模板参数:
     * * std::shared_ptr＜int＞ sh_ptr{ new int{ 10 }, 
     * * [](int* x) { delete x; }, std::allocator＜int＞{} };
     * 
     */
    // 共享指针是可转移的(它们可以移动), 并且它们具有非专属所有权(它们可以复制)
    auto my_ptr_move = std::move(my_ptr);
    // printf("the shared-pointer address: %p, and value: %d\n", my_ptr.get(), *my_ptr);
    printf("the shared-pointer address: %p, and value: %d\n", my_ptr_move.get(), *my_ptr_move);

    // 共享数组是一个拥有动态数组并支持 operator[] 的共享指针;
    // 它的工作原理和独占数组一样, 只是它具有非专属所有权.

    // 删除器对共享指针的工作方式与对独占指针的工作方式相同,
    // 除了不需要提供具有删除器类型的模板参数.

    return 0;
}
