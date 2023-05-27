/**
 * @file list_stl.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cstdio>
#include <functional>
#include <iostream>
#include <list>
#include <string_view>
#include <iterator>
#include <string>

template<typename T>
bool print_container(const T &container, const std::string_view msg)
{
    if (container.empty())
    {
        std::cout << "the container is empty, please check.\n" << std::endl;
        return false;
    }

    std::cout << msg;
    for (const auto &elem : container)
    {
        std::cout << elem << " ";
    }
    std::printf("\n");

    return true;
}

/**
 * @brief the list container in C++ STL.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // ------------------------------------
    std::list<int> list_1{1, 2, 34, 5, 6};
    print_container(list_1, "the list_1 contents: ");

    std::list<int> list_2;
    list_2.push_back(42);
    list_2.push_front(66);
    list_2.push_back(43);

    std::list<int>::iterator iter;
    for (iter = list_2.begin(); iter != list_2.end(); ++iter)
    {
        std::cout << "\t" << *iter;
    }
    std::printf("\n");

    std::cout << "the front of list: " << list_2.front() << std::endl;
    std::cout << "the back of list: " << list_2.back() << std::endl;

    list_1.pop_front();
    print_container(list_1, "After pop front of list: ");
    list_1.pop_back();
    print_container(list_1, "After pop back of list: ");

    list_2.reverse();
    print_container(list_2, "After reverse of list: ");

    list_2.sort();
    print_container(list_2, "After sort of list: ");

    // -----------------------------------------
    /* Insert an integer before 66 by searching */
    auto it = std::find(list_2.begin(), list_2.end(), 66);
    if (it != list_2.end())
        list_2.insert(it, 41);
    print_container(list_2, "After insert of list: ");

    std::list<std::string> list_3 {"Wei", "Li", "Li", "Wei"};
    print_container(list_3, "the contents of list: ");
    list_3.unique();
    print_container(list_3, "After unique of list: ");

    std::list<std::string> list_4;
    list_4.splice(list_4.begin(), list_3); /* attention! */
    print_container(list_3, "the contents of list: ");
    print_container(list_4, "the contents of list: ");

    list_1.merge(list_2); /* attention! */
    print_container(list_2, "the contents of list: ");
    print_container(list_1, "the contents of list: ");

    return 0;
}