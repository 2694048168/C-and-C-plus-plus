/**
 * @file container_linear.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief std::array linear container
 * @version 0.1
 * @date 2022-01-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <vector>
#include <array>
#include <algorithm>

/* std::array
1. Why introduce std::array instead of std::vector directly?
2. Already have a traditional array, why use std::array?
 */

void function_Cstyle_interface(int *p, int len)
{
    for (int i = 0; i != len; ++i)
    {
        std::cout << p[i] << ' ';
    }
    std::cout << '\n';
}

int main(int argc, char **argv)
{
    // step 1. The first question:
    std::cout << "---- First question ----" << std::endl;
    std::vector<int> vec;
    std::cout << "vector size: " << vec.size() << std::endl;         /* output: 0 */
    std::cout << "vector capacity: " << vec.capacity() << std::endl; /* output: 0 */

    /* As you can see, the storage of std::vector is automatically managed and automatically expanded as needed.
    But if there is not enough space, you need to redistribute more memory,
    and reallocating memory is usually a performance-intensive operation. */
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);
    std::cout << "vector size: " << vec.size() << std::endl;         /* output: 3 */
    std::cout << "vector capacity: " << vec.capacity() << std::endl; /* output: 4 */

    /* the auto-expansion logic here is very similar to Golang's slice. */
    vec.push_back(4);
    vec.push_back(5);
    std::cout << "vector size: " << vec.size() << std::endl;         /* output: 5 */
    std::cout << "vector capacity: " << vec.capacity() << std::endl; /* output: 8 */

    /* As can be seed below, although the container empties the element,
    the memory of the emptied element is not returned. */
    vec.clear();
    std::cout << "vector size: " << vec.size() << std::endl;         /* output: 0 */
    std::cout << "vector capacity: " << vec.capacity() << std::endl; /* output: 8 */

    /* Additional memory can be returned to the system via the shrink_to_fit() call */
    vec.shrink_to_fit();
    std::cout << "vector size: " << vec.size() << std::endl;         /* output: 0 */
    std::cout << "vector capacity: " << vec.capacity() << std::endl; /* output: 8 */

    // step 2. The second question:
    std::cout << "---- Second question ----" << std::endl;
    std::array<int, 4> arr = {1, 2, 3, 4};
    std::cout << arr.empty() << std::endl; /* check if container is empty */
    std::cout << arr.size() << std::endl;  /* return the size of the container */

    // iterator support
    for (auto &i : arr)
    {
        std::cout << i + 1 << ' ';
    }
    std::cout << '\n';

    // use lambda expression for sort
    std::sort(arr.begin(), arr.end(), [](int a, int b)
              { return b < a; });

    // int len = 8; /* array size must be constexpr */
    constexpr int len = 8;
    std::array<int, len> arr_new = {1, 2, 3, 4, 5, 6, 7, 8};
    function_Cstyle_interface(arr_new.data(), len);

    /* illegal, different than C-style array, std::array will not deduce to T*
    int *arr_ptr = arr;
    When we started using std::array, it was inevitable that we would encounter a C-style compatible interface. 
    There are three ways to do this:
     */
    // C-style parameter passing
    // function_Cstyle_interface(arr, arr.size()); /* illegal, cannot convert implicitly std::array is not as *arr_ptr */
    function_Cstyle_interface(&arr[0], arr.size());
    function_Cstyle_interface(arr.data(), arr.size());

    return 0;
}
