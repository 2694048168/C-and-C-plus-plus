/**
 * @file 09_bitset.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-06
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <bitset>
#include <cassert>
#include <cstdio>

/**
 * @brief bitset(位集)是一种存储固定大小的位(bit)序列的数据结构.
 * *可以操纵每一位,
 * STL 在＜bitset＞头文件中提供了 std::bitset,
 * 类模板 bitset 接受与所需大小相对应的单个模板参数;
 * ?使用 bool array 可以实现类似的功能, 但 bitset 针对空间效率进行了优化,并提供了一些特殊的便利操作. 
 * 
 */

// ----------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief 默认构造的 bitset 包含所有 0(false)位,
     * *要使用其他内容初始化 bitset, 可以提供一个 unsigned long long 值;
     * 这个整数的位布局被用来设置 bitset 的值, 使用operator[] 可以访问 bitset 中的各位.
     */
    printf("std::bitset supports integer initialization\n");

    std::bitset<4> bs{0b0101}; // 从右边往左边
    assert(bs[0] == true);
    assert(bs[1] == false);
    assert(bs[2] == 1);
    assert(bs[3] == 0);

    printf("std::bitset supports string initialization\n");
    std::bitset<4> bs1(0b0110);
    std::bitset<4> bs2("0110");
    assert(bs1 == bs2);

    // TODO: https://en.cppreference.com/w/cpp/utility/bitset
    // TODO: https://cplusplus.com/reference/bitset/bitset/

    return 0;
}
