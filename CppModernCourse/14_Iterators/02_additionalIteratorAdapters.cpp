/**
 * @file 02_additionalIteratorAdapters.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <cstdio>
#include <iterator>
#include <list>
#include <vector>

/**
 * @brief C++ STL 还提供了移动迭代器适配器和反向迭代器适配器来修改迭代器行为
 * ?1. 移动迭代器适配器 Move Iterator Adapters
 * 移动迭代器适配器是一个类模板, 将所有迭代器访问操作转换为移动操作.
 * ?2. 反向迭代器适配器 Reverse Iterator Adapters
 * 反向迭代器适配器是一个类模板, 交换了迭代器的递增和递减运算符
 */

struct Movable
{
    Movable(int id)
        : id{id}
    {
    }

    Movable(Movable &&m)
    {
        id   = m.id;
        m.id = -1;
    }

    int id;
};

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("\nmove iterators convert accesses into move operations\n");
    std::vector<Movable> donor;
    donor.emplace_back(1);
    donor.emplace_back(2);
    donor.emplace_back(3);

    std::vector<Movable> recipient{
        std::make_move_iterator(donor.begin()),
        std::make_move_iterator(donor.end()),
    };

    assert(donor[0].id == -1);
    assert(donor[1].id == -1);
    assert(donor[2].id == -1);
    /**
     * @brief 这会将所有的迭代器操作转换为移动操作,
     * 所以 Movable 的移动构造函数被调用,
     * 因此 donor 的所有元素都处于移出状态,(-1),
     * 并且 recipient 的所有元素都与 donor 之前的元素相匹配.
     */
    assert(recipient[0].id == 1);
    assert(recipient[1].id == 2);
    assert(recipient[2].id == 3);

    printf("\nreverse iterators can initialize containers\n");
    std::list<int>   original{3, 2, 1};
    std::vector<int> easy_as{original.crbegin(), original.crend()};

    assert(easy_as[0] == 1);
    assert(easy_as[1] == 2);
    assert(easy_as[2] == 3);

    printf("\nmake_reverse_iterator converts a normal iterator\n");
    auto begin = std::make_reverse_iterator(original.cend());
    auto end   = std::make_reverse_iterator(original.cbegin());

    std::vector<int> easy_as_{begin, end};
    assert(easy_as_[0] == 1);
    assert(easy_as_[1] == 2);
    assert(easy_as_[2] == 3);

    return 0;
}
