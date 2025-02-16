/**
 * @file 25_LeastFrequentlyUsed.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 实现一个线程安全的 LFU 缓存, 经典的缓存淘汰策略，其核心思想是优先移除访问次数最少的缓存项
 * @version 0.1
 * @date 2025-02-16
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <iostream>
#include <list>
#include <map>
#include <shared_mutex>
#include <thread>
#include <unordered_map>

/**
  * @brief LFU 的全称是 "Least Frequently Used", 即“最少使用优先”
  * 1. 每个缓存项都有一个访问频率计数器;
  * 2. 每次访问缓存时，该缓存项的计数器加 1;
  * 3. 当缓存达到容量限制时，移除访问频率最小的缓存项;
  * LFU 的核心难点在于如何高效地管理计数器和快速查找需要淘汰的缓存项.
  *
  * 设计线程安全的 LFU
  * 1. 数据结构选择
  *  1.1 频率计数器管理std::map<int, std::list<Key>> 来维护每个频率的键集合,按频率递增排序
  *  1.2 缓存数据存储std::unordered_map<Key, std::pair<Value, int>> 存储键、值及其对应的频率
  *  1.3 互斥锁std::shared_mutex 实现读写分离,提升并发性能
  * 2. 接口设计
  *  2.1 get(const Key&)：获取缓存项的值，更新其频率计数;
  *  2.2 put(const Key&, const Value&)：插入或更新缓存项;
  *  2.3 remove(const Key&)：手动删除缓存项;
  * 3. 最佳实践
  *  3.1 合理设置容量: 根据业务需求设置缓存容量，避免频繁的淘汰操作;
  *  3.2 读写分离优化性能: std::shared_mutex 实现读写分离，减少读操作时的锁竞争;
  *  3.3 按需扩展功能: 可以添加 TTL（Time To Live）机制，以支持缓存项的自动过期;
  * 
  */
template<typename Key, typename Value>
class ThreadSafeLFUCache
{
public:
    ThreadSafeLFUCache(size_t capacity)
        : capacity_(capacity)
    {
    }

    Value get(const Key &key)
    {
        std::shared_lock<std::shared_mutex> read_lock(mutex_);
        if (cache_.find(key) == cache_.end())
        {
            throw std::runtime_error("Key not found");
        }
        // Increase frequency
        updateFrequency(key);
        return cache_[key].first;
    }

    void put(const Key &key, const Value &value)
    {
        std::unique_lock<std::shared_mutex> write_lock(mutex_);
        if (capacity_ == 0)
            return;

        if (cache_.find(key) != cache_.end())
        {
            // Update existing value and frequency
            cache_[key].first = value;
            updateFrequency(key);
        }
        else
        {
            if (cache_.size() >= capacity_)
            {
                evict();
            }
            // Add new item
            cache_[key] = {value, 1};
            freq_map_[1].push_back(key);
            key_iter_map_[key] = --freq_map_[1].end();
        }
    }

    void remove(const Key &key)
    {
        std::unique_lock<std::shared_mutex> write_lock(mutex_);
        if (cache_.find(key) == cache_.end())
            return;

        int freq = cache_[key].second;
        freq_map_[freq].erase(key_iter_map_[key]);
        if (freq_map_[freq].empty())
        {
            freq_map_.erase(freq);
        }
        cache_.erase(key);
        key_iter_map_.erase(key);
    }

private:
    void updateFrequency(const Key &key)
    {
        int freq = cache_[key].second;
        cache_[key].second++;
        freq_map_[freq].erase(key_iter_map_[key]);

        if (freq_map_[freq].empty())
        {
            freq_map_.erase(freq);
        }

        freq_map_[freq + 1].push_back(key);
        key_iter_map_[key] = --freq_map_[freq + 1].end();
    }

    void evict()
    {
        auto it            = freq_map_.begin();
        Key  key_to_remove = it->second.front();
        it->second.pop_front();

        if (it->second.empty())
        {
            freq_map_.erase(it);
        }

        cache_.erase(key_to_remove);
        key_iter_map_.erase(key_to_remove);
    }

private:
    size_t                                                     capacity_;
    std::unordered_map<Key, std::pair<Value, int>>             cache_;
    std::map<int, std::list<Key>>                              freq_map_;
    std::unordered_map<Key, typename std::list<Key>::iterator> key_iter_map_;
    mutable std::shared_mutex                                  mutex_;
};

int main(int argc, const char *argv[])
{
    ThreadSafeLFUCache<int, std::string> cache(3);

    cache.put(1, "one");
    cache.put(2, "two");
    cache.put(3, "three");

    try
    {
        std::cout << "Key 1: " << cache.get(1) << std::endl; // Access key 1
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }

    cache.put(4, "four"); // Evicts key 2
    try
    {
        std::cout << "Key 2: " << cache.get(2) << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
