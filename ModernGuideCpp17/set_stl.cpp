/**
 * @file set_stl.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <algorithm>
#include <functional>
#include <iostream>
#include <set>

template<typename T>
bool print_container(const T &container, const char *msg)
{
    if (container.empty())
    {
        std::cout << "the container is empty, please check.\n" << std::endl;
        return false;
    }

    std::printf("%s: ", msg);
    for (const auto elem : container)
    {
        std::cout << elem << " ";
    }
    std::printf("\n");

    return true;
}

/**
 * @brief Sets are a type of associative container in C++ STL.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // ------------------------------------
    std::set<char> set_char;
    set_char.insert('a');
    set_char.insert('a');
    set_char.insert('F');
    set_char.insert('G');
    set_char.insert('G');
    set_char.insert('d');

    std::cout << "the size of set: " << set_char.size() << std::endl;
    for (const auto &str : set_char)
    {
        std::cout << str << " ";
    }
    std::cout << "\n" << std::endl;

    std::set<int> set_1{1, 3, 5, 3, 8, 7, 2, 42};
    print_container(set_1, "the elements of set(default)");

    std::set<int, std::greater<int>> set_2{1, 3, 5, 3, 8, 7, 2, 42};
    print_container(set_2, "the elements of set(greater)");

    // ------------------------------------
    std::cout << "--------------------------" << std::endl;
    // iterator pointing to position where 42 is
    std::set<int>::iterator iter_position;
    iter_position = set_1.find(7);
    auto pos      = set_1.find(42);
    std::cout << "the position of 7 in set: " << *iter_position << std::endl;
    std::cout << "the position of 42 in set: " << *pos << std::endl;

    // ------------------------------------
    std::cout << "--------------------------" << std::endl;
    // check if 11 is present or not
    if (set_2.count(11))
        std::cout << "11 is present in the set\n";
    else
        std::cout << "11 is not present in the set\n";

    // checks if 18 is present or not
    if (set_2.count(8))
        std::cout << "18 is present in the set\n";
    else
        std::cout << "18 is not present in the set\n";

    // ----------- multiset ----------------
    std::cout << "--------------------------" << std::endl;
    std::multiset<int, std::greater<int>> set_multi_1;

    // insert elements in random order
    set_multi_1.insert(40);
    set_multi_1.insert(30);
    set_multi_1.insert(60);
    set_multi_1.insert(20);
    set_multi_1.insert(50);

    // 50 will be added again to the multiset unlike set
    set_multi_1.insert(50);
    set_multi_1.insert(30);
    print_container(set_multi_1, "the elements of multiset: ");

    set_multi_1.erase(set_multi_1.begin(), set_multi_1.find(30));
    print_container(set_multi_1, "after erase of multiset: ");

    return 0;
}