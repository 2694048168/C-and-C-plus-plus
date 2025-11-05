/**
 * @file LeastFrequentlyUsed.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief LFU 
 * @version 0.1
 * @date 2025-11-04
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <iostream>
#include <list>
#include <string>
#include <unordered_map>
#include <utility>

/**
 * @brief LRU(Least Frequently Used) 缓存是一种缓存淘汰机制,
 * 淘汰 使用次数最少 且 最久没用的元素;
 * 
 */
class LFUCache
{
private:
    struct Node
    {
        int value;
        int freq;

        std::list<int>::iterator it;
    };

    std::unordered_map<int, Node> nodes;

    std::unordered_map<int, std::list<int>> freqList;

    int mMinFreq;
    int mCapacity;

private:
    void touch(int key)
    {
        int  f  = nodes[key].freq;
        auto it = nodes[key].it;
        freqList[f].erase(it);

        if (freqList[f].empty() && f == mMinFreq)
            ++mMinFreq;

        ++nodes[key].freq;
        freqList[f + 1].push_back(key);
        nodes[key].it = freqList[f + 1].begin();
    }

public:
    int get(int key)
    {
        if (0 == mCapacity || !nodes.count(key))
            return -1;

        touch(key);

        return nodes[key].value;
    }

    void put(int key, int value)
    {
        if (0 == mCapacity)
            return;

        if (nodes.count(key))
        {
            nodes[key].value = value;
            touch(key);
            return;
        }

        if (mCapacity == nodes.size())
        {
            int oldKey = freqList[mMinFreq].back();
            freqList[mMinFreq].pop_back();
            nodes.erase(oldKey);
        }

        mMinFreq = 1;
        freqList[1].push_front(key);
        nodes[key] = {value, 1, freqList[1].begin()};
    }

public:
    explicit LFUCache(size_t capacity)
        : mCapacity(capacity)
        , mMinFreq(0)
    {
    }

    ~LFUCache()
    {
        mCapacity = 0;
        mMinFreq  = 0;
    }
};

// -------------------------------------
// g++ LeastFrequentlyUsed.cpp -std=c++20
// clang++ LeastFrequentlyUsed.cpp -std=c++20
// -------------------------------------
int main(int argc, const char *argv[])
{
    LFUCache cache(2);

    cache.put(1, 1);
    cache.put(2, 2);

    std::cout << "Get elem key=1, value=" << cache.get(1) << "\n";

    cache.put(3, 3);

    std::cout << "Get elem key=2, value=" << cache.get(2) << "\n";
    std::cout << "Get elem key=3, value=" << cache.get(3) << "\n";
    std::cout << "Get elem key=1, value=" << cache.get(1) << "\n";

    return 0;
}
