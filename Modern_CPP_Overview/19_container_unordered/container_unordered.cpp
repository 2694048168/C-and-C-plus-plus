/**
 * @file container_unordered.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief Unordered Container; std::unordered_map; std::unordered_set; std::unordered_multimap; std::unordered_multiset;
 * @version 0.1
 * @date 2022-01-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <string>
#include <unordered_map>
#include <map>

/* Unordered Container 
We are already familiar with the ordered container std::map and std::set in traditional C++. 
These elements are internally implemented by red-black trees. 
The average complexity of inserts and searches is O(log(size)). 

The elements in the unordered container are not sorted, 
and the internals is implemented by the Hash table. 
The average complexity of inserting and searching for elements is O(constant), 
Significant performance gains can be achieved without concern for the order of the elements inside the container.

C++11 introduces two sets of unordered containers: 
std::unordered_map and std::unordered_multimap
and std::unordered_set and std::unordered_multiset.

Their usage is basically similar to the original std::map and std::multimap and std::set and set::multiset
Since these containers are already familiar to us, we will not compare them one by one. 
Let’s compare std::map and std::unordered_map directly:
 */

int main(int argc, char** argv)
{
    // initialized in same order
    std::unordered_map<int, std::string> unorder_map = {
        {1, "first"},
        {3, "third"},
        {2, "second"}
    };

    std::map<int, std::string> order_map = {
        {1, "first"},
        {3, "third"},
        {2, "second"}
    };

    // iterates in the same way
    std::cout << "std::unordered_map\n";
    // for (const auto &n : unorder_map)
    // {
    //     std::cout << "Key: [" << n.first << "] Value:[" << n.second << "]\n";
    // }
    for (const auto &[key, value] : unorder_map) /* structured binding in C++17 */
    {
        std::cout << "Key: [" << key << "] Value:[" << value << "]\n";
    }
    
    std::cout << "std::map\n";
    // for (const auto &element : order_map)
    // {
    //     std::cout << "Key: [" << element.first << "] Value:[" << element.second << "]\n";
    // }
    for (const auto &[key, value] : order_map) /* structured binding in C++17 */
    {
        std::cout << "Key: [" << key << "] Value:[" << value << "]\n";
    }
    
    return 0;
}
