/**
 * @file lock_free_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-31
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <atomic>
#include <iostream>
#include <utility>

/**
 * @brief 
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    struct A
    {
        int a[100];
        // long a;
        // int b;
    };

    struct B
    {
        int x, y;
    };

    struct C
    {
        int a[100];
    };

    std::atomic<A>           a;
    std::atomic<B>           b;
    std::atomic<C>           c;
    std::atomic<long double> d;
    std::cout << std::boolalpha << "std::atomic<A> is lock free? " << a.is_lock_free() << '\n'
              << "std::atomic<B> is lock free? " << b.is_lock_free() << '\n';

    std::cout << std::boolalpha << "std::atomic<A> is lock free? " << std::atomic_is_lock_free(&a) << '\n'
              << "std::atomic<B> is lock free? " << std::atomic_is_lock_free(&b) << '\n';

    std::cout << "c.is_always_lock_free: " << c.is_always_lock_free << std::endl;

    std::cout << std::boolalpha << "d.is_lock_free (): " << d.is_lock_free() << std::endl;

    // static constexpr bool is_always_lock_free = /*implementation-defined*/;

    return 0;
}
