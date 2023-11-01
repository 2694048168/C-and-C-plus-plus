/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-30
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <array>
#include <cassert>
#include <chrono>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

/**
 * @brief Binary to string conversion
 * Write a function that, given a range of 8-bit integers (such as an array or vector), 
 * returns a string that contains a hexadecimal representation of the input data. 
 * The function should be able to produce both uppercase and lowercase content.
 * 
 */

/**
 * @brief Solution:
In order to write a general-purpose function that can handle various sorts of ranges,
 such as an std::array, std::vector, a C-like array, or others, 
 we should write a function template. 
In the following, there are two overloads; 
one that takes a container as an argument and a flag indicating the casing style, 
and one that takes a pair of iterators (to mark the first and then one past 
the end element of the range) and the flag to indicate casing. 
The content of the range is written to an std::ostringstream object, 
with the appropriate I/O manipulators, such as width, filling character, or case flag:
---------------------------------------------- */
template<typename Iter>
std::string bytes_to_hexstr(Iter begin, Iter end, const bool uppercase = false)
{
    std::ostringstream oss;
    if (uppercase)
        oss.setf(std::ios_base::uppercase);

    for (; begin != end; ++begin)
    {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(*begin);
    }

    return oss.str();
}

template<typename Container>
std::string bytes_to_hexstr(const Container &container, const bool uppercase = false)
{
    return bytes_to_hexstr(std::cbegin(container), std::cend(container), uppercase);
}

// ==================== 测试线程的开销问题
template<typename Time = std::chrono::microseconds, typename Clock = std::chrono::high_resolution_clock>
struct perf_timer
{
    template<typename F, typename... Args>
    static Time duration(F &&f, Args... args)
    {
        auto start = Clock::now();

        std::invoke(std::forward<F>(f), std::forward<Args>(args)...);

        auto end = Clock::now();

        return std::chrono::duration_cast<Time>(end - start);
    }
};

void test_vector()
{
    std::vector<unsigned char> v{0xBA, 0xAD, 0xF0, 0x0D};

    assert(bytes_to_hexstr(v, true) == "BAADF00D");
    assert(bytes_to_hexstr(v) == "baadf00d");
}

void test_array()
{
    std::array<unsigned char, 6> a{
        {1, 2, 3, 4, 5, 6}
    };

    assert(bytes_to_hexstr(a, true) == "010203040506");
    assert(bytes_to_hexstr(a) == "010203040506");
}

void test_buffer()
{
    unsigned char buf[5] = {0x11, 0x22, 0x33, 0x44, 0x55};

    assert(bytes_to_hexstr(buf, true) == "1122334455");
    assert(bytes_to_hexstr(buf) == "1122334455");
}

void test_multi_thread()
{
    auto thread_task1 = std::thread(test_vector);
    auto thread_task2 = std::thread(test_array);
    auto thread_task3 = std::thread(test_buffer);

    if (thread_task1.joinable())
        thread_task1.join();

    if (thread_task2.joinable())
        thread_task2.join();

    if (thread_task3.joinable())
        thread_task3.join();
}

void test_single_thread()
{
    test_vector();
    test_array();
    test_buffer();
}

// --------------------------------
int main(int argc, char **argv)
{
    // 测试耗时
    auto timer1 = perf_timer<>::duration(test_single_thread);
    auto timer2 = perf_timer<>::duration(test_multi_thread);

    std::cout << std::chrono::duration<double, std::milli>(timer1).count() << " ms\n";
    std::cout << std::chrono::duration<double, std::milli>(timer2).count() << " ms\n";

    return 0;
}
