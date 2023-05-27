/**
 * @file heap_stl.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <algorithm>
#include <iostream>
#include <vector>

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
 * @brief The heap data structure can be implemented in a range using STL
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // ------------------------------------
    std::vector<int> vec1{20, 30, 40, 25, 15, 42};

    // Converting vector into a heap using make_heap()
    std::make_heap(vec1.begin(), vec1.end());

    std::cout << "the maximum element of heap: " << vec1.front() << std::endl;

    // ------------------------------------
    std::cout << "-------------------------" << std::endl;
    std::vector<int> vec2{20, 30, 40, 10};
    std::make_heap(vec2.begin(), vec2.end());
    print_container(vec2, "init heap: ");

    vec2.push_back(50);
    print_container(vec2, "vector.push_back: ");

    std::push_heap(vec2.begin(), vec2.end());
    print_container(vec2, "push_heap: ");

    // make_heap(): Converts given range to a heap.
    // push_heap(): Arrange the heap after insertion at the end.
    // TODO pop_heap(): Moves the max element at the end for deletion.
    // TODO sort_heap(): Sort the elements of the max_heap to ascending order.
    // TODO is_heap(): Checks if the given range is max_heap.
    // TODO is_heap_until(): Returns the largest sub-range that is max_heap.

    return 0;
}