/**
 * @file binary_search_stl.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-26
 * 
 * @copyright Copyright (c) 2023
 * 
 * --------------------------------------
 * The prototype for binary search is:
 * binary_search(startaddress, endaddress, valuetofind)
 *   Parameters :
 *     startaddress: the address of the first element of the array.
 *     endaddress: the address of the next contiguous location of
 *            the last element of the array.
 *     valuetofind: the target value which we have to search for.
 * Returns : true if an element equal to valuetofind is found, else false.
 * -----------------------------------------------------------------------
 * 
 */

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <vector>

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
 * @brief the binary serarch algorithm in C++ STL.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // ----------------------------------
    int arr[] = {1, 5, 8, 9, 6, 7, 3, 4, 2, 0};

    std::size_t size_arr = sizeof(arr) / sizeof(arr[0]);
    std::cout << "the array is:\n";
    print_array(arr, size_arr);

    std::cout << "Let's say we want to search for\n";
    std::cout << "2 in the array So, we first sort the array";
    std::sort(arr, arr + size_arr);

    std::cout << "\nThe array after sorting is :\n";
    print_array(arr, size_arr);

    std::cout << "\nNow, we do the binary search";

    if (std::binary_search(arr, arr + size_arr, 2))
        std::cout << "\nElement found in the arrrray";
    else
        std::cout << "\nElement not found in the array";

    std::cout << "\n\nNow, say we want to search for 10";
    if (std::binary_search(arr, arr + size_arr, 10))
        std::cout << "\nElement found in the array";
    else
        std::cout << "\nElement not found in the array";

    // ----------------------------------
    std::cout << "\n-------------------------------" << std::endl;
    std::vector<int> haystack{1, 3, 4, 5, 9};
    std::vector<int> needles{1, 2, 3};

    for (auto needle : needles)
    {
        std::cout << "Searching for " << needle << '\n';
        if (std::binary_search(haystack.begin(), haystack.end(), needle))
            std::cout << "Found " << needle << '\n';
        else
            std::cout << "no dice!\n";
    }

    return 0;
}