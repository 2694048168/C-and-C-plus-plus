/**
 * @file LeastRecentlyUsed.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief LRU 缓存是一种缓存淘汰机制
 * @version 0.1
 * @date 2025-10-16
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
 * @brief LRU(Least Recently Used) 缓存是一种缓存淘汰机制,
 * 每次访问(读/写) 一个元素, 就把该元素标记为最近使用;
 * 当缓存满了, 就淘汰最久没用的元素;
 * 
 */
template<typename K, typename V>
class LRUCache
{
private:
    size_t mCapacity = 0;

    std::list<std::pair<K, V>> mItemList{};

    std::unordered_map<K, typename std::list<std::pair<K, V>>::iterator> mIndexMap;

public:
    // 命中返回true, 并把valu写入out; 同时将当前节点移动到表头
    bool Get(const K &key, V &out)
    {
        auto iter = mIndexMap.find(key);
        if (mIndexMap.end() == iter)
        {
            return false;
        }

        mItemList.splice(mItemList.begin(), mItemList, iter->second);
        out = iter->second->second;
        return true;
    }

    // 插入或更新: 存在则更新并把当前节点移动到表头; 不存在则可能需要淘汰尾部的节点;
    void Put(const K &key, V value)
    {
        auto iter = mIndexMap.find(key);
        if (mIndexMap.end() != iter)
        {
            iter->second->second = std::move(value);
            mItemList.splice(mItemList.begin(), mItemList, iter->second);
            return;
        }

        if (mCapacity == mItemList.size())
        {
            auto &old = mItemList.back();
            mIndexMap.erase(old.first);
            mItemList.pop_back();
        }
        mItemList.push_front(std::make_pair(key, std::move(value)));
        mIndexMap[mItemList.front().first] = mItemList.begin();
    }

    void DebugPrint()
    {
        std::cout << "[MRU ---> LRU] ";
        for (auto &elem : mItemList)
        {
            std::cout << "(" << elem.first << " : " << elem.second << ")";
        }
        std::cout << "\n";
    }

public:
    explicit LRUCache(size_t capacity)
        : mCapacity(capacity)
    {
    }

    ~LRUCache()
    {
        mCapacity = 0;
        mItemList.clear();
        mIndexMap.clear();
    }
};

// -------------------------------------
// g++ LeastRecentlyUsed.cpp -std=c++20
// clang++ LeastRecentlyUsed.cpp -std=c++20
// -------------------------------------
int main(int argc, const char *argv[])
{
    LRUCache<int, std::string> cache(3);

    cache.Put(1, "One");
    cache.Put(2, "Two");
    cache.Put(3, "Three");
    cache.DebugPrint();

    std::string out;
    if (cache.Get(2, out))
    {
        std::cout << "Get elem key=2, value=" << out << "\n";
    }
    cache.DebugPrint();

    cache.Put(4, "Fourth");
    cache.DebugPrint();

    return 0;
}
