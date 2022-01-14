/**
 * @file memory_alignment.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief Memory Alignment 内存对齐问题
 * @version 0.1
 * @date 2022-01-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>

struct Storage
{
    char a;
    int b;
    double c;
    long long d;
};

struct alignas(max_align_t) AlignasStorage
{
    char a;
    int b;
    double c;
    long long d;
};

int main(int argc, char const *argv[])
{
    std::cout << alignof(Storage) << std::endl;
    std::cout << alignof(AlignasStorage) << std::endl;
    return 0;
}
