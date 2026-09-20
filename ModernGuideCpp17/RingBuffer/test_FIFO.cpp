/**
 * @file test_FIFO.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-09-20
 * 
 * @copyright Copyright (c) 2026
 * 
 * g++ -std=c++17 -O2 -pthread -Wall -Wextra -pedantic test_FIFO.cpp -o test.exe
 * clang++ -std=c++17 -O2 -pthread -Wall -Wextra -pedantic test_FIFO.cpp -o test.exe
 * 
 */

#include "RingBuffer_FIFO.hpp"

#include <atomic>
#include <cassert>
#include <iostream>
#include <string>
#include <thread>
#include <type_traits>

static_assert(!std::is_copy_constructible<RingBuffer<int, 4>>::value, "RingBuffer should not be copy constructible");
static_assert(!std::is_copy_assignable<RingBuffer<int, 4>>::value, "RingBuffer should not be copy assignable");
static_assert(!std::is_move_constructible<RingBuffer<int, 4>>::value, "RingBuffer should not be move constructible");
static_assert(!std::is_move_assignable<RingBuffer<int, 4>>::value, "RingBuffer should not be move assignable");

void test_basic_fifo()
{
    RingBuffer<int, 8> rb;
    assert(rb.Size() == 0);

    // 该实现保留一个空槽，所以实际容量是 Capacity - 1
    for (int i = 0; i < 7; ++i)
    {
        assert(rb.push(i));
    }

    assert(rb.Size() == 7);
    assert(!rb.push(100));

    for (int i = 0; i < 7; ++i)
    {
        int v = -1;
        assert(rb.pop(v));
        assert(v == i);
    }

    assert(rb.Size() == 0);

    int v = -1;
    assert(!rb.pop(v));
}

void test_wrap()
{
    RingBuffer<int, 4> rb; // 实际容量 3

    assert(rb.push(1));
    assert(rb.push(2));
    assert(rb.push(3));
    assert(!rb.push(4));

    int v = 0;

    assert(rb.pop(v) && v == 1);
    assert(rb.push(4));
    assert(rb.Size() == 3);

    assert(rb.pop(v) && v == 2);
    assert(rb.pop(v) && v == 3);
    assert(rb.pop(v) && v == 4);
    assert(!rb.pop(v));
}

void test_size_after_wrap()
{
    RingBuffer<int, 4> rb;

    assert(rb.push(1));
    assert(rb.push(2));
    assert(rb.push(3));
    assert(rb.Size() == 3);

    int v = 0;

    assert(rb.pop(v) && v == 1);
    assert(rb.Size() == 2);

    assert(rb.push(4));
    assert(rb.Size() == 3);

    assert(rb.pop(v) && v == 2);
    assert(rb.Size() == 2);

    assert(rb.push(5));
    assert(rb.Size() == 3);
}

void test_no_placement_new_int()
{
    RingBuffer<int, 4, false> rb;

    assert(rb.push(1));
    assert(rb.push(2));
    assert(rb.push(3));
    assert(!rb.push(4));

    int v = 0;

    assert(rb.pop(v) && v == 1);
    assert(rb.pop(v) && v == 2);
    assert(rb.pop(v) && v == 3);
    assert(!rb.pop(v));
}

struct Life
{
    static std::atomic<int> alive;
    int                     value;

    Life(int v = 0)
        : value(v)
    {
        ++alive;
    }

    Life(const Life &o)
        : value(o.value)
    {
        ++alive;
    }

    Life(Life &&o) noexcept
        : value(o.value)
    {
        o.value = -1;
        ++alive;
    }

    Life &operator=(const Life &o)
    {
        value = o.value;
        return *this;
    }

    Life &operator=(Life &&o) noexcept
    {
        value   = o.value;
        o.value = -1;
        return *this;
    }

    ~Life()
    {
        --alive;
    }
};

std::atomic<int> Life::alive{0};

void test_nontrivial_placement_new()
{
    assert(Life::alive.load() == 0);

    {
        RingBuffer<Life, 4> rb;
        assert(Life::alive.load() == 0);

        assert(rb.push(Life(10)));
        assert(rb.push(Life(20)));
        assert(rb.push(Life(30)));
        assert(Life::alive.load() == 3);

        assert(!rb.push(Life(40)));
        assert(Life::alive.load() == 3);

        Life out;
        assert(Life::alive.load() == 4);

        assert(rb.pop(out));
        assert(out.value == 10);
        assert(Life::alive.load() == 3);

        assert(rb.pop(out));
        assert(out.value == 20);
        assert(Life::alive.load() == 2);

        assert(rb.pop(out));
        assert(out.value == 30);
        assert(Life::alive.load() == 1);

        assert(!rb.pop(out));
        assert(Life::alive.load() == 1);
    }

    assert(Life::alive.load() == 0);
}

void test_nontrivial_no_placement_new()
{
    assert(Life::alive.load() == 0);

    {
        RingBuffer<Life, 4, false> rb;

        // PlacementNew=false 时，std::array<Life, 4> 会默认构造 4 个对象
        assert(Life::alive.load() == 4);

        assert(rb.push(Life(10)));
        assert(rb.push(Life(20)));
        assert(rb.push(Life(30)));
        assert(!rb.push(Life(40)));

        Life out;
        assert(rb.pop(out) && out.value == 10);
        assert(rb.pop(out) && out.value == 20);
        assert(rb.pop(out) && out.value == 30);
        assert(!rb.pop(out));
    }

    assert(Life::alive.load() == 0);
}

void test_string()
{
    RingBuffer<std::string, 4> rb;

    assert(rb.push("hello"));
    assert(rb.push(std::string("world")));

    std::string s;

    assert(rb.pop(s));
    assert(s == "hello");

    assert(rb.pop(s));
    assert(s == "world");

    assert(!rb.pop(s));
}

void test_spsc_multithread()
{
    constexpr int         N = 200000;
    RingBuffer<int, 1024> rb; // 实际容量 1023

    std::atomic<bool> start{false};

    std::thread producer(
        [&]
        {
            while (!start.load(std::memory_order_acquire))
            {
                std::this_thread::yield();
            }

            for (int i = 0; i < N; ++i)
            {
                while (!rb.push(i))
                {
                    std::this_thread::yield();
                }
            }
        });

    std::thread consumer(
        [&]
        {
            while (!start.load(std::memory_order_acquire))
            {
                std::this_thread::yield();
            }

            int expected = 0;
            int v        = 0;

            while (expected < N)
            {
                if (rb.pop(v))
                {
                    assert(v == expected);
                    ++expected;
                }
                else
                {
                    std::this_thread::yield();
                }
            }
        });

    start.store(true, std::memory_order_release);

    producer.join();
    consumer.join();

    assert(rb.Size() == 0);
}

// -----------------------------------
int main(int argc, const char **argv)
{
    test_basic_fifo();
    test_wrap();
    test_size_after_wrap();
    test_no_placement_new_int();
    test_nontrivial_placement_new();
    test_nontrivial_no_placement_new();
    test_string();
    test_spsc_multithread();

    std::cout << "All RingBuffer tests passed.\n";
    return 0;
}
