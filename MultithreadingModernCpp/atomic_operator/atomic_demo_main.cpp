/**
 * @file atomic_demo_main.cpp
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
#include <mutex>
#include <thread>
#include <vector>

class Counters
{
    int a;
    int b;
}; // user-defined trivially-copyable type

std::atomic<Counters> cnt;

class A
{
    int  myInt1;
    int  myInt2;
    long myLong1;
    long myLong2;
    int *myPtr;

    A(int val = 1)
    {
        myInt1 = myInt2 = myLong1 = myLong2 = 1;
    }

    A(int i1, int i2, long l1, long l2)
    {
        myInt1  = i1;
        myInt2  = i2;
        myLong1 = l1;
        myLong2 = l2;
    }

    // A(const A& rhs)
    //     : myInt1(rhs.myInt1),
    //       myInt2(rhs.myInt2),
    //       myLong1(rhs.myLong1),
    //       myLong2(rhs.myLong2) {}  // user-defined copy ctor

    void IncrementMyInt1()
    {
        myInt1++;
    }

    // A(const A&& a) { } // user-defined copy ctor
    // A& operator = (const A& rhs) { return *this;} // user-defined copy ctor
    A operator++()
    {
        myInt1++;
        myInt2++;
        myLong1++;
        myLong2++;
        return A(myInt1, myInt2, myLong1, myLong2);
    } // user-defined increment
};

// std::atomic<std::vector<int>> v;

/**
 * @brief 
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // Demos is_trivially_copyable
    {
        std::atomic<A> a; // ! Error

        auto is_trivially_copyable = std::is_trivially_copyable<A>::value;
        auto v_isTrivial           = std::is_trivially_copyable<std::vector<int>>::value;
        std::cout << "is_trivially_copyable: " << is_trivially_copyable << std::endl;
        std::cout << "v_isTrivial: " << v_isTrivial << std::endl;
    }
    // Assignment non-atomic to atomic
    {
        std::atomic<int> a(1);
        int              b = 2;
        a                  = b;

        std::cout << "a: " << a << ", b: " << b << std::endl;
    }
    // Assignment non-atomic to atomic
    {
        struct A
        {
            int a[100];
        };

        std::atomic<A> a1, a2;
        A              a3;
        a1 = a3;
    }
    // Assignment atomic to atomic
    {
        std::atomic<int> a(1), b(2);
        std::cout << "a: " << a << ", b: " << b << std::endl;
        // a = b;  // Doesn't compile!
        a.store(b.load());
        std::cout << "a: " << a << ", b: " << b << std::endl;
    }
    // Integer
    {
        std::atomic<int> x_int(0);
        std::atomic<int> y_int{0};
        std::atomic<int> z_int = {0};
        // std::atomic<int> t_int = 0; // Doesn't compile!
        std::atomic<int> t_int;
        t_int = 0; // Compiles!

        x_int++;
        z_int++;

        std::cout << "x_int: " << x_int << std::endl;
        std::cout << "z_int: " << z_int << std::endl;
    }
    // load/store
    {
        std::atomic<int> atomic_x(1);
        int              y(2);

        y        = atomic_x;
        atomic_x = y;

        // OR
        y = atomic_x.load();
        atomic_x.store(y);
    }
    // load/store with tags
    {
        std::atomic<int> atomic_x(1);
        int              y(2);

        y        = atomic_x;
        atomic_x = y;

        // OR
        y = atomic_x.load(std::memory_order_relaxed);
        atomic_x.store(y, std::memory_order_relaxed);
    }
    // exchange
    std::cout << "Exchange with non atomic: " << std::endl;

    {
        std::atomic<int> atomic_x(1);
        int              y(2);
        auto             z = atomic_x.exchange(y);

        std::cout << "atomic_x: " << atomic_x << ", y: " << y << ", z: " << z << std::endl;
    }
    // exchange
    std::cout << "Exchange with atomic: " << std::endl;
    {
        std::atomic<int> atomic_x(1);
        std::atomic<int> atomic_y(2);
        auto             z = atomic_x.exchange(atomic_y);

        std::cout << "atomic_x: " << atomic_x << ", atomic_y: " << atomic_y << ", z: " << z << std::endl;
    }
    // exchange
    {
        // if (atomic_x == expected) {
        //   atomic_x = desired;
        //   return true;
        // } else {
        //   atomic_x = expected;
        //   return false;
        // }
        std::atomic<int> atomic_x(1);
        int              expected = 2;
        int              desired  = 3;

        bool success = atomic_x.compare_exchange_strong(expected, desired);

        std::cout << "success: " << success << ", atomic_x: " << atomic_x << ", expected: " << expected
                  << ", desired: " << desired << std::endl;

        success = atomic_x.compare_exchange_strong(expected, desired);

        std::cout << "success: " << success << ", atomic_x: " << atomic_x << ", expected: " << expected
                  << ", desired: " << desired << std::endl;
    }
    // Double
    {
        std::atomic<double> x_double(0);

        // x_double++;  // Doesn't compile!
        std::cout << "x_double: " << x_double << std::endl;
    }

    //-----------------------------------------------------
    // Exchange:
    {
        std::atomic<int> x_int(0);

        int y_int = 2;
        std::cout << "Before: x_int: " << x_int << ", y_int: " << y_int << std::endl;
        // Exchange is equivalent to: z=x, x=y
        int z_int = x_int.exchange(y_int);
        std::cout << "After: x_int: " << x_int << ", y_int: " << y_int << ", z_int: " << z_int << std::endl;
    }
    // Exchange with z,y atomic:
    {
        std::atomic<int> x_int(0);
        std::atomic<int> z_int(3);
        std::atomic<int> y_int(2);
        std::cout << "Before: x_int: " << x_int << ", y_int: " << y_int << std::endl;
        // Exchange is equivalent to: z=x, x=y
        z_int = x_int.exchange(y_int);
        std::cout << "After: x_int: " << x_int << ", y_int: " << y_int << ", z_int: " << z_int << std::endl;
    }
    // Exchange with z atomic:
    {
        std::atomic<int> x_int(0);
        std::atomic<int> z_int(0);

        int y_int = 2;
        std::cout << "Before: x_int: " << x_int << ", y_int: " << y_int << std::endl;
        // Exchange is equivalent to: z=x, x=y
        z_int = x_int.exchange(y_int);
        std::cout << "After: x_int: " << x_int << ", y_int: " << y_int << ", z_int: " << z_int << std::endl;
    }
    // Compare Exchange:
    std::cout << "Exchange unsuccessful: " << std::endl;
    {
        std::atomic<int> x_int(0);

        int y_int = 2;
        std::cout << "Before: x_int: " << x_int << ", y_int: " << y_int << std::endl;
        // compare_exchange_strong is equivalent to: if (x==y) then {x=z; return
        // true;}
        //                                           else {y=x; return false;}
        int  z_int   = 5;
        bool success = x_int.compare_exchange_strong(y_int, z_int);
        std::cout << "After: x_int: " << x_int << ", y_int: " << y_int << ", z_int: " << z_int << std::endl;
    }
    //-----------------------------------------------------
    // Exchange:
    std::cout << "Exchange successful: " << std::endl;
    {
        std::atomic<int> x_int(2);

        int y_int = 2;
        std::cout << "Before: x_int: " << x_int << ", y_int: " << y_int << std::endl;
        // compare_exchange_strong is equivalent to: if (x==y) then {x=z; return
        // true;}
        //                                           else {y=x; return false;}
        int  z_int   = 5;
        bool success = x_int.compare_exchange_strong(y_int, z_int);
        std::cout << "After: x_int: " << x_int << ", y_int: " << y_int << ", z_int: " << z_int << std::endl;
    }
    //-----------------------------------------------------

    return 0;
}
