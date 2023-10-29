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

// #include <stdio.h>
#include <time.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <ctime>
#include <iostream>
#include <iterator>
#include <list>
#include <string>
#include <vector>

/* Container any, all, none

Write a set of general-purpose functions that enable checking whether any, all,
 or none of the specified arguments are present in a given container.
------------------------------------------------------------------- */

/* Solution:
--------------------------------------------------------------- */
template<class C, class T>
bool contains(const C &c, const T &value)
{
    return std::end(c) != std::find(std::begin(c), std::end(c), value);
}

template<class C, class... T>
bool contains_any(const C &c, T &&...value)
{
    return (... || contains(c, value));
}

template<class C, class... T>
bool contains_all(const C &c, T &&...value)
{
    return (... && contains(c, value));
}

template<class C, class... T>
bool contains_none(const C &c, T &&...value)
{
    return !contains_any(c, std::forward<T>(value)...);
}

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime()
{
    time_t     now = time(0);
    // struct tm tstruct;
    struct tm *tstruct;
    char       buf[80];
    // tstruct = *localtime(&now);
    localtime_s(tstruct, &now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    // strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", tstruct);

    return buf;
}

// -------------------------
int main(int argc, char **)
{
    std::vector<int>   v{1, 2, 3, 4, 5, 6};
    std::array<int, 6> a{
        {1, 2, 3, 4, 5, 6}
    };
    std::list<int> l{1, 2, 3, 4, 5, 6};

    assert(contains(v, 3));
    assert(contains(a, 3));
    assert(contains(l, 3));

    assert(!contains(v, 30));
    assert(!contains(v, 30));
    assert(!contains(v, 30));

    assert(contains_any(v, 0, 3, 30));
    assert(contains_any(a, 0, 3, 30));
    assert(contains_any(l, 0, 3, 30));

    assert(!contains_any(v, 0, 30));
    assert(!contains_any(a, 0, 30));
    assert(!contains_any(l, 0, 30));

    assert(contains_all(v, 1, 3, 6));
    assert(contains_all(a, 1, 3, 6));
    assert(contains_all(l, 1, 3, 6));

    assert(!contains_all(v, 0, 1));
    assert(!contains_all(a, 0, 1));
    assert(!contains_all(l, 0, 1));

    assert(contains_none(v, 0, 7));
    assert(contains_none(a, 0, 7));
    assert(contains_none(l, 0, 7));

    assert(!contains_none(v, 0, 6, 7));
    assert(!contains_none(a, 0, 6, 7));
    assert(!contains_none(l, 0, 6, 7));

    std::cout << "[" << currentDateTime() << "] All assert thought successfully\n";

    return 0;
}
