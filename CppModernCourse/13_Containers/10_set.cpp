/**
 * @file 10_set.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-06
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <array>
#include <cassert>
#include <cstdio>
#include <set>

/**
  * @brief 关联容器允许进行非常快速的搜索,此容器系列有三个维度:
  * 1. 元素是否只包含键(集合)或键值对(映射);
  * 2. 元素是否有序;
  * 3. 键是否唯一;
  * 
  * ====STL 的＜set＞头文件中可用的 std::set 包含称为键(key)的已排序的唯一元素;
  * 因为集合存储已排序的元素, 所以可以有效地插入、删除和搜索元素;
  * 此外集合支持对其元素进行有序迭代, 并且可以使用比较器对象完全控制键的排序.
  * 
  * ====移动语义和复制语义
  * 除了移动构造函数和复制构造函数之外, 移动赋值运算符和复制赋值运算符也可用.
  * *与其他容器的复制操作一样, 集合的复制可能非常慢, 因为每个元素都需要被复制;
  * *而移动操作通常很快, 因为元素驻留在动态内存中, 集合可以简单地传递所有权而不会干扰元素
  * 
  */

// ----------------------------------
int main(int argc, const char **argv)
{
    printf("std::set supports\n");
    std::set<int> emp;
    assert(emp.empty());

    std::set<int> fib{1, 1, 2, 3, 5};
    assert(fib.size() == 4);

    // "copy construction"
    auto fib_copy(fib);
    assert(fib_copy.size() == 4);

    // "move construction"
    auto fib_moved(std::move(fib));
    assert(fib.empty());
    assert(fib_moved.size() == 4);

    // range construction
    std::array<int, 5> fib_array{1, 1, 2, 3, 5};

    std::set<int> fib_set(fib_array.cbegin(), fib_array.cend());
    assert(fib_set.size() == 4);

    /**
     * @brief 元素访问 Element Access
     * 有几个选项可用于从集合中提取元素, 基本方法是 find, 它接收对键的 const 引用并返回一个迭代器;
     * 如果集合包含与元素匹配的键, find 将返回一个指向该元素的迭代器;
     * 如果集合中没有, 它将返回一个指向 end 的迭代器;
     * *lower_bound 方法返回一个迭代器, 指向不小于键参数的第一个元素;
     * *upper_bound 方法返回大于给定键的第一个元素;
     *
     * set 类支持两种额外的查找方法, 主要是为了兼容非唯一关联容器:
     * 1. count 方法返回匹配键的元素数, 因为集合元素是唯一的, 所以 count 返回 0 或 1;
     * 2. equal_range 方法返回一个半开半闭区间, 该区间包含与给定键匹配的所有元素,
     *    该方法返回一个迭代器对(std::pair),其中第一个迭代器(first)指向匹配元素,
     *    第二个迭代器(second)指向第一个迭代器指向的元素之后的元素.
     * 如果 equal_range 找不到匹配的元素, 则第一个和第二个都指向大于给定键的第一个元素;
     * 换句话说, equal_range 返回的迭代器对等价于 lower_bound(作为第一个迭代器)
     *    和 upper_bound(作为第二个迭代器)
     * 
     */
    printf("\nstd::set allows access\n");
    std::set<int> fib_{1, 1, 2, 3, 5};
    assert(*fib_.find(3) == 3);
    assert(fib_.find(100) == fib_.end());

    // with count
    assert(fib_.count(3) == 1);
    assert(fib_.count(100) == 0);

    // with lower_bound
    auto itr = fib.lower_bound(3);
    printf("the lower_bound: %d\n", *itr);

    // with upper_bound
    itr = fib.upper_bound(3);
    printf("the upper_bound: %d\n", *itr);

    // with equal_range
    auto pair_itr = fib.equal_range(3);
    assert(*pair_itr.first == 3);
    assert(*pair_itr.second == 5);

    /**
     * @brief 添加元素 Adding Elements
     * 将元素添加到集合的方式有三种:
     * 1. 使用 insert 将现有元素复制到集合中;
     * 2. 使用 emplace 就地在集合中构造一个新元素;
     * 3. emplace_hint 也就地构造新元素, 在于 emplace_hint 方法将迭代器作为第一个参数,
     *    这个迭代器是搜索的起点(即提示,hint), 如果迭代器接近新插入元素的正确位置, 这可以提供显著的加速效果.
     */
    printf("\nstd::set allows insertion\n");
    std::set<int> fib_adding{1, 1, 2, 3, 5};

    // "with insert"
    fib_adding.insert(8);
    assert(fib.find(8) != fib.end());

    // "with emplace"
    fib_adding.emplace(42);
    assert(fib.find(42) != fib.end());

    // with emplace_hint
    fib_adding.emplace_hint(fib_adding.end(), 24);
    assert(fib.find(24) != fib.end());

    /**
     * @brief 移除元素 Removing Elements
     * 使用 erase 方法可以从集合中删除元素, 方法被重载了,
     * *因而可接受键、迭代器或半开半闭区间.
     */
    printf("\nstd::set allows removal\n");
    std::set<int> fib_remove{1, 1, 2, 3, 5};
    // "with erase"
    fib_remove.erase(3);
    assert(fib_remove.find(3) == fib_remove.end());

    // "with clear
    fib_remove.clear();
    assert(fib.empty());

    /**
     * @brief 存储模型 Storage Mode
     * 集合的操作都很快, 因为集合通常被实现为红黑树(red-black tree).
     * 这种结构将每个元素视为一个节点, 每个节点有一个父节点和最多两个子节点, 两个子节点分别为它的左分支和右分支.
     * 每个节点的子节点都经过了排序, 因此左分支的所有子节点都小于右分支的子节点;
     * 这样只要树的分支大致平衡(高度相等), 就可以比线性迭代方法更快地执行搜索;
     * ?红黑树在插入和删除元素后有额外的机制来重新平衡分支.
     */

    return 0;
}
