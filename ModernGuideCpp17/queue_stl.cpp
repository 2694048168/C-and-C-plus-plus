/**
 * @file queue_stl.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <array>
#include <functional>
#include <iostream>
#include <queue>
#include <string_view>
#include <string>
#include <typeinfo>

/**
 * @brief the queue container adaptors and 
 *        the priority queue container adapter in C++ STL.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // -------------- queue ---------------------
    std::queue<int> queue_1;
    queue_1.push(10);
    queue_1.push(30);
    queue_1.push(40);
    queue_1.push(42);

    auto print = [&queue_1](const std::string_view msg)
    {
        std::queue<int> queue_back = queue_1;
        while (!queue_back.empty())
        {
            std::cout << queue_back.front() << " ";
            queue_back.pop();
        }
        std::cout << "\n";
    };
    print("the elements of queue: ");

    std::cout << "\nqueue.size() : " << queue_1.size();
    std::cout << "\nqueue.front() : " << queue_1.front();
    std::cout << "\nqueue.back() : " << queue_1.back();

    // -------------- priority queue ---------------------
    std::array<int, 6> arr{10, 2, 4, 8, 6, 9};
    std::cout << "\n\nthe elements of array: ";
    for (const auto &elem : arr)
    {
        std::cout << elem << " ";
    }
    std::cout << "\n";

    std::priority_queue<int> prior_queue;
    // pushing array sequentially one by one
    for (size_t i = 0; i < 6; ++i)
    {
        prior_queue.push(arr[i]);
    }
    std::cout << "the elements of priority queue(default): ";
    while (!prior_queue.empty())
    {
        std::cout << prior_queue.top() << " ";
        prior_queue.pop();
    }

    std::priority_queue<int, std::vector<int>, std::greater<int>> prior_queue_minHeap(arr.begin(), arr.end());
    std::cout << "\nthe elements of priority queue(min heap): ";
    while (!prior_queue_minHeap.empty())
    {
        std::cout << prior_queue_minHeap.top() << " ";
        prior_queue_minHeap.pop();
    }

    // -------------- priority queue ---------------------
    // declare integer value_type for priority queue
    std::priority_queue<int>::value_type AnInt;
    std::cout << "\n\nthe type of AnInt: " << typeid(AnInt).name();
    std::cout << "\nthe type of int: " << typeid(int).name();
 
    // declare string value_type for priority queue
    std::priority_queue<std::string>::value_type AString;
    std::cout << "\nthe type of AString: " << typeid(AString).name();
    std::cout << "\nthe type of string: " << typeid(std::string).name();

    return 0;
}