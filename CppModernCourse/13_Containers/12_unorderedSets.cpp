/**
 * @file 12_unorderedSets.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <cstdio>
#include <functional>
#include <unordered_set>

/**
 * @brief STL 的＜unordered_set＞头文件中可用的 std::unordered_set 也是一个关联容器,
 * *这种容器包含未排序的唯一键,
 * unordered_set 支持大多数与普通集合和multiset 相同的操作,
 * *但其内部存储模型完全不同
 * 
 * ?====与使用比较器将元素排序的红黑树不同, unordered_set 通常实现为哈希表;
 * 如果键之间没有自然顺序并且不需要以这种顺序遍历集合, 则可以使用 unordered_set;
 * 在许多情况下, 都可以使用普通集合或 unordered_set;
 * *尽管它们看起来很相似, 但它们的内部表示是完全不同的, 因此它们具有不同的性能特征;
 * 如果很关注性能问题, 请衡量两者的性能并使用更合适的那个.
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief 哈希函数有几个要求:
     * 1. 它接受一个键并返回 size_t 哈希码
     * 2. 它不会抛出异常
     * 3. 相等的键产生相等的哈希码
     * 4. 不相等的键产生不相等的哈希码的概率很高(哈希冲突的概率很低)
     * 
     * STL 在＜functional＞头文件中提供了哈希器类模板 std::hash＜T＞,
     * 其中包含基本类型、枚举类型、指针类型、optional、variant、智能指针等的特化
     */
    printf("std::hash<long> returns\n");
    std::hash<long> hasher;
    auto            hash_code_42 = hasher(42);
    printf("equal hash codes for equal keys\n");
    assert(hash_code_42 == hasher(42));
    printf("unequal hash codes for unequal keys\n");
    assert(hash_code_42 != hasher(43));

    printf("std::equal_to<long> returns\n");
    std::equal_to<long> long_equal_to;
    printf("true when arguments equal\n");
    assert(long_equal_to(42, 42));

    printf("false when arguments unequal\n");
    // assert(long_equal_to(42, 43));

    printf("\n=========== std::unordered_set ================\n");
    std::unordered_set<unsigned long> sheep(100);
    // allows bucket count specification on construction
    assert(sheep.bucket_count() >= 100);
    assert(sheep.bucket_count() <= sheep.max_bucket_count());
    printf("sheep.max_load_factor() = %f\n", sheep.max_load_factor());

    // allows us to reserve space for elements
    sheep.reserve(100'000);
    sheep.insert(0);
    assert(sheep.load_factor() <= 0.00001);

    while (sheep.size() < 100'000)
    {
        sheep.insert(sheep.size());
    }
    assert(sheep.load_factor() <= 1.0);

    return 0;
}
