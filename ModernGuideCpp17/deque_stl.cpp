/**
 * @file deque_stl.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <deque>
#include <iostream>

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
 * @brief the deque container in C++ STL.(double-ended deque)
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // ----------------------------
    std::deque<int> deque1;
    deque1.push_back(10);
    deque1.push_front(20);
    deque1.push_back(30);
    deque1.push_front(15);
    print_container(deque1, "the contents of deque: ");

    std::cout << "\ndeque.size: " << deque1.size();
    std::cout << "\ndeque.max_size: " << deque1.max_size();
    std::cout << "\ndeque.at(2): " << deque1.at(2);
    std::cout << "\ndeque.front(): " << deque1.front();
    std::cout << "\ndeque.back(): " << deque1.back();

    deque1.pop_back();
    print_container(deque1, "\nafter pop_back of deque: ");

    deque1.pop_front();
    print_container(deque1, "after pop_front of deque: ");

    // ----------------------------
    std::deque<int> deque2{1, 2, 3, 4, 5};
    print_container(deque2, "\nthe contenst of deque: ");

    std::deque<int>::iterator iter = deque2.begin();
    ++iter;

    iter = deque2.insert(iter, 42);
    print_container(deque2, "after insert of deque: ");

    std::cout << "the iter address: " << &(*iter) << std::endl;
    std::cout << "the insert address: " << &*(deque2.begin()+1) << std::endl;
    std::cout << "the begin address: " << &(*deque2.begin()) << std::endl;

    return 0;
}