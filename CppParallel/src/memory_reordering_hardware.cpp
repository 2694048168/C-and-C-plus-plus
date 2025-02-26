/**
 * @file memory_reordering.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief A simple example of hardware memory barriers
 * @version 0.1
 * @date 2025-02-26
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <emmintrin.h>

#include <cassert>
#include <iostream>
#include <semaphore>
#include <thread>

// A simple example of hardware memory barriers
void reorder_hardware(std::binary_semaphore &start, std::counting_semaphore<2> &end, int &write, int &read, int &reg)
{
    // Keep going forever
    while (true)
    {
        // Wait for the signal to start
        start.acquire();

        // Write to v2
        write = 1;

        // Barrier to prevent hardware memory reordering!
        _mm_mfence();

        // Read v1
        reg = read;

        // Say we're done for this iteration
        end.release();
    }
}

// -------------------------------------
int main(int argc, const char *argv[])
{
    // Semaphores for signaling threads
    std::binary_semaphore      start_t0(0);
    std::binary_semaphore      start_t1(0);
    std::counting_semaphore<2> done(0);

    // Variable for memory re-ordering
    int x  = 0;
    int y  = 0;
    int r1 = 0;
    int r2 = 0;

    // Start threads
    std::thread t1([&] { reorder_hardware(start_t0, done, x, y, r1); });
    std::thread t2([&] { reorder_hardware(start_t1, done, y, x, r2); });

    for (int i = 0;; i++)
    {
        // Zero x/y each iteration
        x = 0;
        y = 0;

        // Allow threads to continue
        start_t0.release();
        start_t1.release();

        // Wait for both threads to finish
        done.acquire();
        done.acquire();

        // Check for memory reordering
        auto reordering = (r1 == 0) && (r2 == 0);
        if (reordering)
        {
            std::cout << "ERROR! r1 = " << r1 << ", r2 = " << r2 << ", Iteration: " << i << '\n';
            assert(false);
        }
        else
            std::cout << "ALL GOOD! r1 = " << r1 << ", r2 = " << r2 << ", Iteration: " << i << '\n';
    }

    // Should not get here (infinite loop)
    return 0;
}
