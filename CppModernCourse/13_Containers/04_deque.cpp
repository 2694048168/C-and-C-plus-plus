/**
 * @file 04_deque.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-06
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>
#include <deque>

/**
 * @brief 双端队列(deque, 读作"deck")是一个顺序容器, 具有从前后快速插入和删除的操作.
 * deque 是一个合成词(double-ended queue), STL 实现的 std::deque 可从＜deque＞头文件中获得. 
 * 
 * *向量和双端队列具有非常相似的接口,但在内部它们的存储模型完全不同;
 * 向量可以保证所有元素在内存中都是连续的, 而双端队列的内存通常是分散的;
 * 这使得调整大小的重量级操作更加高效, 并能够在容器的前端快速插入/删除元素.
 * *对于构造和访问成员的操作, 向量和双端队列是相同的.
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("std::deque supports front insertion\n");

    std::deque<char> deck;
    deck.push_front('a'); // a
    deck.push_back('i');  // ai
    deck.push_front('c'); // cai
    deck.push_back('n');  // cain

    for (const auto &elem : deck)
    {
        printf("the value: %c\n", elem);
    }

    return 0;
}
