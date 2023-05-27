/**
 * @file sort_stl.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-26
 * 
 * @copyright Copyright (c) 2023
 * 
 * ----------------------------------
 * The prototype for sort is:
 * sort(startaddress, endaddress)
 *    startaddress: the address of the first element of the array
 *    endaddress: the address of the next contiguous location of 
 *             the last element of the array.
 * So actually sort() sorts in the range of [startaddress,endaddress)
 * -------------------------------------------------------------------
 * 
 * clang++ sort_stl.cpp -std=c++17
 * g++ sort_stl.cpp -std=c++17
 * 
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <string_view>

/**
 * @brief : print the array element on console for <T> type.
 * 
 * @tparam T : the type of element in array.
 * @param num_array : the pointer of array or array name.
 * @param arr_size : the number of element in the array.
 * @return true : represents this function call successfully.
 * @return false : or the function call failed.
 */
template<typename T>
bool print_array(const T *num_array, const std::size_t arr_size)
{
    if (num_array == nullptr)
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

/**
 * @brief the sort algorithm in C++ STL
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // ---------------------------------
    int arr[] = {5, 1, 8, 3, 5, 6, 0, 9};

    // ascending the array
    std::sort(std::begin(arr), std::end(arr));

    for (const auto &elem : arr)
    {
        std::cout << elem << " ";
    }
    std::cout << "\n" << std::endl;

    // ---------------------------------
    // just for address for sort: [startaddress, endaddress)
    int num[] = {1, 5, 8, 9, 6, 7, 3, 4, 2, 0};

    const std::size_t arr_size = sizeof(num) / sizeof(num[0]);

    std::cout << "before the sort: \n";
    print_array(num, arr_size);

    std::sort(num, num + arr_size);

    std::cout << "after the sort: \n";
    print_array(num, arr_size);

    // ---------------------------------
    // compare function for std::sort algorithm
    std::array<int, 10> s{5, 7, 4, 2, 8, 6, 1, 9, 0, 3};

    auto print = [&s](std::string_view const msg)
    {
        for (auto a : s)
        {
            std::cout << a << " ";
        }
        std::cout << ": " << msg << "\n";
    };

    std::sort(s.begin(), s.end());
    print("sorted with the default operator<");

    std::sort(s.begin(), s.end(), std::greater<int>());
    print("sorted with the STL compare function object");

    struct
    {
        bool operator()(int a, int b) const
        {
            return a < b;
        }

    } customLess;

    std::sort(s.begin(), s.end(), customLess);
    print("sorted with a custom function object");

    std::sort(s.begin(), s.end(), [](int a, int b) { return a > b; });
    print("sorted with a lambda expression");

    return 0;
}