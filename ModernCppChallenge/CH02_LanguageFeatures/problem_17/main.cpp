/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-29
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <vector>

/* Creating a 2D array with basic operations
Write a class template that represents a two-dimensional array container with methods
 for element access (at() and data()), 
 capacity querying, iterators, filling, and swapping. 
It should be possible to move objects of this type.
----------------------------------------------------- */
// This approach has a smaller memory footprint,
// stores all data in a single contiguous chunk, and is also simpler to implement.
// For these reasons, it is the preferred solution.
// A possible implementation of the two-dimensional array class with
//  the requested functionality is shown here:
template<class T, size_t R, size_t C>
class array2d
{
private:
    typedef T                 value_type;
    typedef value_type       *iterator;
    typedef const value_type *const_iterator;

    std::vector<T> arr;

public:
    array2d()
        : arr(R * C)
    {
    }

    explicit array2d(std::initializer_list<T> l)
        : arr(l)
    {
    }

    constexpr T *data() noexcept
    {
        return arr.data();
    }

    constexpr const T *data() const noexcept
    {
        return arr.data();
    }

    constexpr T &at(const size_t r, const size_t c)
    {
        return arr.at(r * C + c);
    }

    constexpr const T &at(const size_t r, const size_t c) const
    {
        return arr.at(r * C + c);
    }

    constexpr T &operator()(const size_t r, const size_t c)
    {
        return arr[r * C + c];
    }

    constexpr const T &operator()(const size_t r, const size_t c) const
    {
        return arr[r * C + c];
    }

    constexpr bool empty() const noexcept
    {
        return R == 0 || C == 0;
    }

    constexpr size_t size(const int rank) const
    {
        if (rank == 1)
            return R;
        else if (rank == 2)
            return C;
        throw std::out_of_range("Rank is out of range!");
    }

    void fill(const T &value)
    {
        std::fill(std::begin(arr), std::end(arr), value);
    }

    void swap(array2d &other) noexcept
    {
        arr.swap(other.arr);
    }

    const_iterator begin() const
    {
        return arr.data();
    }

    const_iterator end() const
    {
        return arr.data() + arr.size();
    }

    iterator begin()
    {
        return arr.data();
    }

    iterator end()
    {
        return arr.data() + arr.size();
    }
};

template<class T, size_t R, size_t C>
void print_array2d(const array2d<T, R, C> &arr)
{
    for (int i = 0; i < R; ++i)
    {
        for (int j = 0; j < C; ++j)
        {
            std::cout << arr.at(i, j) << ' ';
        }

        std::cout << std::endl;
    }
}

/* ------------------------------------------------------------
Before looking at how we could define such a structure,
let's consider several test cases for it.
The following snippet shows all the functionality that was requested:
------------------------------------------------------------
// Function 1: element access
array2d<int, 2, 3> arr{1, 2, 3, 4, 5, 6};

for (size_t i = 0; i < arr.size(1); ++i)
    for (size_t j = 0; j < arr.size(2); ++j) arr(i, j) *= 2;

// Function 2: iterating
std::copy(std::begin(arr), std::end(arr), std::ostream_iterator<int>(std::cout, " "));

// Function 3: filling
array2d<int, 2, 3> b;
b.fill(1);

// Function 3: swapping
arr.swap(b);

// Function 3: moving
array2d<int, 2, 3> c(std::move(b));
------------------------------------------------------------ */
int main(int argc, char **)
{
    {
        std::cout << "test fill" << std::endl;

        array2d<int, 2, 3> a;
        a.fill(1);

        print_array2d(a);
    }

    {
        std::cout << "test operator()" << std::endl;
        array2d<int, 2, 3> a;

        for (size_t i = 0; i < a.size(1); ++i)
        {
            for (size_t j = 0; j < a.size(2); ++j)
            {
                a(i, j) = 1 + i * 3 + j;
            }
        }

        print_array2d(a);
    }

    {
        std::cout << "test move semantics" << std::endl;

        array2d<int, 2, 3> a{10, 20, 30, 40, 50, 60};
        print_array2d(a);

        array2d<int, 2, 3> b(std::move(a));
        print_array2d(b);
    }

    {
        std::cout << "test swap" << std::endl;

        array2d<int, 2, 3> a{1, 2, 3, 4, 5, 6};
        array2d<int, 2, 3> b{10, 20, 30, 40, 50, 60};

        print_array2d(a);
        print_array2d(b);

        a.swap(b);

        print_array2d(a);
        print_array2d(b);
    }

    {
        std::cout << "test capacity" << std::endl;

        const array2d<int, 2, 3> a{1, 2, 3, 4, 5, 6};

        for (size_t i = 0; i < a.size(1); ++i)
        {
            for (size_t j = 0; j < a.size(2); ++j)
            {
                std::cout << a(i, j) << ' ';
            }

            std::cout << std::endl;
        }
    }

    {
        std::cout << "test iterators" << std::endl;

        const array2d<int, 2, 3> a{1, 2, 3, 4, 5, 6};
        for (const auto e : a)
        {
            std::cout << e << ' ';
        }
        std::cout << std::endl;

        std::copy(std::begin(a), std::end(a), std::ostream_iterator<int>(std::cout, " "));

        std::cout << std::endl;
    }

    return 0;
}
