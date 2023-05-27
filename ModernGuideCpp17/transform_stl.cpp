/**
 * @file transform_stl.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <vector>

/**
 * @brief : print the array element on console for <T> type.
 * 
 * @tparam T : the type of element in array.
 * @param num_array : the reference of std::array.
 * @param arr_size : the number of elements in the array.
 * @return true : represents this function call successfully.
 * @return false : or the function call failed.
 */
template<typename T>
bool print_array(const T &num_array, const std::size_t arr_size)
{
    if (num_array.empty())
    {
        std::cout << "the array is empty, please check.\n" << std::endl;
        return false;
    }

    for (size_t idx = 0; idx < arr_size; ++idx)
    {
        std::cout << num_array[idx] << " ";
    }
    std::cout << "\n" << std::endl;

    return true;
}

int increment_func(int x)
{
    return (x + 1);
}

/* why not error for template function pass to std::transform? */
// template <typename T>
// T increment_func(T x)
// {
//     return (x + 1);
// }
// // Explicitly instantiate
// template int increment_func<int>(int);

/**
 * @brief the std::transform() in C++ STL.
 * for 'Unary Operation' and 'Binary Operation'
 *
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // ------------------------------------
    std::array<int, 5> arr_1{1, 2, 3, 4, 5};
    std::array<int, 5> arr_2{4, 5, 6, 7, 8};
    std::array<int, 5> arr_result{};

    std::transform(arr_1.begin(), arr_1.end(), arr_2.begin(), arr_result.begin(), std::plus<int>());

    print_array(arr_result, 5);

    // ------------------------------------
    std::vector<int> vect{1, 2, 3, 4, 5};

    std::transform(vect.begin(), vect.end(), vect.begin(), increment_func);
    print_array(vect, vect.size());

    std::transform(vect.begin(), vect.end(), vect.begin(), [](int x) { return (x + 1); });
    print_array(vect, vect.size());

    return 0;
}
