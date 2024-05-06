/**
 * @file 03_vectorStorageModel.cpp
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
#include <cstdint>
#include <cstdio>
#include <vector>

/**
 * @brief 存储模型 Storage Model
 *  尽管向量元素在内存中是连续的, 就像数组一样, 但它们的相似之处就止于此;
 * 向量具有动态大小, 因此它必须能够调整大小, 向量的分配器管理动态内存.
 * 因为分配成本很高昂, 所以向量将请求比当前包含元素数量更多的内存;
 * 一旦它不能再添加更多的元素, 它将请求额外的内存, 向量的内存始终是连续的, 
 * 因此如果现有向量末尾没有足够的空间,它将分配一个全新的内存区域并将向量的所有元素移动到新区域.
 * 
 * *向量包含的元素数量称为它的大小(size);
 * *而在必须调整大小之前它理论上可以容纳的元素数量称为容量(capacity);
 * 可以通过 capacity 方法获取向量当前的容量, 
 * 可以通过 max_size 方法获取向量可以调整到的绝对最大容量.
 * 
 * 1. 如果能提前知道需要的确切容量, 则可以使用 reserve 方法, 它接受一个 size_t参数(所需容量对应的元素数量);
 * 2. 如果刚刚删除了几个元素并想将内存返回给分配器,则可以使用 shrink_to_fit 方法, 
 *    该方法声明有多余的容量, 分配器可以决定是否减少容量(这是一个非绑定调用);
 * 3. 还可以使用 clear 方法删除(delete)向量中的所有元素并将其大小(size)设置为零.
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("\nstd::vector exposes size management methods\n");
    std::vector<std::array<uint8_t, 1024>> kb_store;
    assert((kb_store.max_size() > 0));
    assert(kb_store.empty());

    size_t elements{1024};
    kb_store.reserve(elements);
    assert(kb_store.empty());
    assert(kb_store.capacity() == elements);
    printf("the capacity: %lld\n", kb_store.capacity());
    printf("the size: %lld\n", kb_store.size());

    kb_store.emplace_back();
    kb_store.emplace_back();
    kb_store.emplace_back();
    assert(kb_store.size() == 3);

    kb_store.shrink_to_fit();
    assert(kb_store.capacity() >= 3);
    printf("the capacity: %lld\n", kb_store.capacity());

    kb_store.clear();
    assert(kb_store.empty());
    assert(kb_store.capacity() >= 3);
    printf("the capacity after clear: %lld\n", kb_store.capacity());
    printf("the size after clear: %lld\n", kb_store.size());

    return 0;
}
