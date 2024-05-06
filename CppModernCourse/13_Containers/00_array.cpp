/**
 * @file 00_array.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <array>
#include <cassert>
#include <cstdio>

/**
 * @brief 标准模板库(Standard Template Library,STL)是标准库的一部分,
 * 它提供容器和操作它们的算法, 迭代器充当两者之间的接口.
 * 容器是一种特殊的数据结构, 它们以一定的组织方式存储对象, 该组织方式遵循特定访问规则.
 * 容器分为以下三种:
 * 1. 顺序容器连续存储元素, 就像在数组中一样;
 * 2. 关联容器存储已排序的元素;
 * 3. 无序关联容器存储哈希对象;
 * 
 * 关联容器和无序关联容器提供对单个元素的快速搜索.
 * 所有容器都是围绕其包含对象的 RAII 包装器, 因此它们管理着它们拥有的元素的存储期和生命周期.
 * 此外, 每个容器都提供了一组成员函数, 用于对对象集合执行各种操作.
 * *为特定应用程序选择哪个容器取决于所需的操作、包含的对象的特征以及特定访问模式下的效率.
 * 
 * ===== std::array  是一个顺序容器,
 *  包含固定大小的、连续的一系列元素, 将内置数组的纯粹性能和效率与现代支持复制构造,
 * 移动构造, 复制赋值, 移动赋值, 知道容器大小, 提供边界检查成员访问和其他高级功能的便利性相结合.
 * 
 */

std::array<int, 10> static_array;

// -----------------------------------
int main(int argc, const char **argv)
{
    // Sequence Containers  顺序容器是允许顺序成员访问的 STL 容器
    assert(static_array[0] == 0);
    printf("the first value: %d\n", static_array[0]);

    std::array<int, 10> local_array;
    printf("uninitialized without braced initializers: %d\n", local_array[0]);

    printf("initialized with braced initializers:\n");
    std::array<int, 10> local_array_init{1, 1, 2, 3};
    assert(local_array_init[0] == 1);

    /**
     * @brief 元素访问, 访问任意数组元素的三种主要方法是:
     * 1. operator[];
     * 2. at
     * 3. get
     * 
     * *如果索引参数超出范围,at 将抛出 std::out_of_range 异常, 而 operator[] 将导致未定义行为.
     * *使用 get 的一个主要好处是可以进行编译时边界检查
     */
    printf("the last value: %d\n", local_array_init[2]);
    printf("the last value: %d\n", local_array_init.at(2));
    printf("the last value: %d\n", std::get<2>(local_array_init));

    /**
     * @brief 存储模型 Storage Mode
     * std::array 不额外进行分配, 就像内置数组一样, 它包含所有元素,
     * 这意味着复制成本通常很高昂, 因为每个组成元素都需要被复制;
     * 移动成本可能也很高昂, 具体取决于底层类型是否也具有移动构造和移动赋值机制, 它们的成本相对较小.
     * 
     * 四种不同的方法提取指向数组第一个元素的指针:
     * 1. 最直接的方法是使用 data 方法, 这将返回一个指向第一个元素的指针;
     * 2. 第一个元素上使用地址运算符 &, 该元素可以使用 operator[]
     * 3. 第一个元素上使用地址运算符 &, 该元素可以使用 at ;
     * 4. 第一个元素上使用地址运算符 &, 该元素可以使用 front;
     * 
     */
    printf("We can obtain a pointer to the first element using\n");
    std::array<char, 9> color{'o', 'c', 't', 'a', 'r', 'i', 'n', 'e'};

    const auto *color_ptr = color.data();
    assert(color_ptr == &color.at(0));
    assert(color_ptr == &color[0]);
    assert(color_ptr == &color.front());
    printf("the address of array: %p\n", color_ptr);
    printf("the first value of array: %c\n", *color_ptr);

    /**
     * @brief 关于数组还可以使用 size 或 max_size 方法(它们对于 std::array 来说是相同的)
     * 查询 array 的大小, 因为 array 具有固定的大小,
     * 所以这些方法的值是静态的并且在编译时是已知的
     */
    printf("the size or max_size of std::array: %lld\n", static_array.size());
    printf("the size or max_size of std::array: %lld\n", color.max_size());

    return 0;
}
