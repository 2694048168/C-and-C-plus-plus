/**
 * @file 02_vector.cpp
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
#include <vector>

/**
 * @brief STL 的＜vector＞头文件中的 std::vector 是一个顺序容器,
 *  可以保存动态大小的,连续的一系列元素.
 * 向量(vector)动态管理其存储空间, 不需要程序员的外部帮助.
 * 
 * !如果有固定数量的元素, 那么请务必考虑使用数组, 因为与向量相比, 数组在开销方面更少.
 * 类模板 std::vector＜T, Allocator＞ 接受两个模板参数;
 *  第一个是包含类型 T, 第二个是分配器类型 Allocator,是可选的,默认为std::allocator＜T＞
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("std::vector supports default construction\n");

    std::vector<const char *> vec;
    assert(vec.empty());
    if (vec.empty())
        printf("the vector is empty\n");

    printf("\nstd::vector supports braced initialization\n");
    std::vector<int> fib{1, 1, 2, 3, 5};
    printf("the size of vector: %lld\n", fib.size());

    printf("\nvector 支持大括号初始化列表和填充构造函数\n");
    std::vector<int> five_nine{5, 9};
    assert(five_nine[0] == 5);
    assert(five_nine[1] == 9);

    std::vector<int> five_nines(5, 9);
    assert(five_nines[0] == 9);
    assert(five_nines[4] == 9);
    printf("the value of all element: %d\n", five_nines[0]);

    /**
     * @brief 从半开半闭区间构造向量,具体是通过传入要复制的范围的 begin 和 end 迭代器来完成;
     * 在各种编程上下文中, 可能希望拼接出某个范围的子集并将其复制到向量中以进行进一步处理
     * 
     */
    printf("\nstd::vector supports construction from iterators\n");
    std::array<int, 5> fib_arr{1, 1, 2, 3, 5};
    std::vector<int>   fib_vec(fib_arr.begin(), fib_arr.end());
    assert(fib_vec.size() == fib_arr.size());
    printf("the value of fib_vec[4] == %d\n", fib_vec[4]);

    /**
     * @brief 移动语义和复制语义 Move and Copy Semantics
     * 使用向量vector, 可以获得完整的复制构造, 移动构造, 复制赋值和移动赋值的支持;
     * 任何向量的复制操作成本都可能非常高昂, 因为它们是按元素复制的;
     * 移动操作通常非常快,因为包含的元素在动态内存中, 并且移出向量可以简单地将所有权传递给移入目标向量,不需要移动包含的元素.
     * 
     * ====元素访问 Element Access
     * 元素访问操作: at, operator[], front, back, data;
     * 可以使用 size 方法查询向量中包含的元素的数量, 此方法的返回值在运行时可能会有所不同;
     * 可以使用 empty 方法确定向量是否包含元素;
     * 
     * ====添加元素 Adding Elements
     * *如果要替换向量中的所有元素, 可以使用assign 方法, 接受初始化列表并替换所有现有元素.
     */
    printf("\nstd::vector assign replaces existing elements\n");
    std::vector<int> message{13, 80, 110, 114, 102, 110, 101};
    assert(message.size() == 7);
    message.assign({67, 97, 101, 115, 97, 114});
    assert(message.at(5) == 114);
    printf("the assign after vector size: %lld\n", message.size());

    /**
     * ====如果要将单个新元素插入向量中, 可以使用 insert 方法, 该方法需要两个参数:
     * 一个是迭代器, 另一个是要插入的元素;
     * 它将在迭代器指向的现有元素之前插入给定元素的副本.
     *
     * !使用 insert 后, 现有的迭代器将失效, 不能重复使用迭代器 third_element,
     * !向量可能已经调整大小并在内存中重定位, 旧的迭代器处在垃圾内存中.
     *
     */
    printf("\nstd::vector insert places new elements\n");
    std::vector<int> zeros(3, 0);
    printf("the insert before value: %d\n", zeros.at(2));

    auto third_element = zeros.begin() + 2;
    zeros.insert(third_element, 10);
    assert(zeros[2] == 10);
    assert(zeros.size() == 4);
    printf("the insert after value: %d\n", zeros.at(2));

    //如需将元素插入向量的末尾, 请使用push_back方法,只需提供要复制到向量中的元素
    printf("\nstd::vector push_back places new elements\n");
    zeros.push_back(42);
    printf("the value of last element: %d\n", *--zeros.end());

    /**
     * @brief 可以使用 emplace 和 emplace_back 方法就地构造新元素.
     * emplace 方法是一个可变参数模板, 与 insert 方法一样, 它接受迭代器作为第一个参数, 剩余的参数被转发给适当的构造函数;
     * emplace_back 方法也是一个可变参数模板, 与 push_back 一样, 它不需要迭代器参数,
     *  它接受任意数量的参数并将这些参数转发给适当的构造函数.
     */
    printf("\nstd::vector emplace methods forward arguments\n");
    std::vector<std::pair<int, int>> factors;
    factors.emplace_back(2, 30);
    factors.emplace_back(3, 20);
    factors.emplace_back(4, 15);
    factors.emplace(factors.begin(), 1, 60);
    assert(factors[0].first == 1);
    assert(factors[0].second == 60);
    //TODO: Because the emplacement methods can construct elements in place,
    //TODO: it seems they should be more efficient than the insertion methods.

    return 0;
}
