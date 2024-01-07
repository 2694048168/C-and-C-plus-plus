/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Priority queue 优先队列
 * @version 0.1
 * @date 2023-12-10
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <iterator>
#include <vector>

/**
 * @brief Priority queue
 * 
 * Write a data structure that represents a priority queue 
 * that provides constant time lookup for the largest element, 
 * but has logarithmic time complexity for adding and removing elements. 
 * A queue inserts new elements at the end and removes elements from the top. 
 * By default, the queue should use operator< to compare elements, 
 * but it should be possible for the user to provide a comparison function object 
 * that returns true if the first argument is less than the second. 
 * 
 * push() to add a new element
 * pop() to remove the top element
 * top() to provide access to the top element
 * size() to indicate the number of elements in the queue
 * empty() to indicate whether the queue is empty
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
template<class T, class Compare = std::less<typename std::vector<T>::value_type>>
class PriorityQueue
{
    typedef typename std::vector<T>::value_type      value_type;
    typedef typename std::vector<T>::size_type       size_type;
    typedef typename std::vector<T>::reference       reference;
    typedef typename std::vector<T>::const_reference const_reference;

public:
    bool empty() const noexcept
    {
        return data.empty();
    }

    size_type size() const noexcept
    {
        return data.size();
    }

    void push(const value_type &value)
    {
        data.push_back(value);
        std::push_heap(std::begin(data), std::end(data), comparer);
    }

    void pop()
    {
        std::pop_heap(std::begin(data), std::end(data), comparer);
        data.pop_back();
    }

    const_reference top() const
    {
        return data.front();
    }

    void swap(PriorityQueue &other) noexcept
    {
        swap(data, other.data);
        swap(comparer, other.comparer);
    }

private:
    std::vector<T> data;
    Compare        comparer;
};

template<class T, class Compare>
void swap(PriorityQueue<T, Compare> &lhs, PriorityQueue<T, Compare> &rhs) noexcept(noexcept(lhs.swap(rhs)))
{
    lhs.swap(rhs);
}

// ------------------------------
int main(int argc, char **argv)
{
    PriorityQueue<int> q;
    for (const auto &elem : {1, 5, 3, 1, 13, 21, 8})
    {
        q.push(elem);
    }

    assert(!q.empty());
    assert(q.size() == 7);

    while (!q.empty())
    {
        std::cout << q.top() << ' ';
        q.pop();
    }

    return 0;
}
