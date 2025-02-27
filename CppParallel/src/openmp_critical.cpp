/**
 * @file openmp_critical.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief An example with an OpenMP critical section 临界区
 * @version 0.1
 * @date 2025-02-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <cassert>

// ------------------------------------
int main(int argc, const char *argv[])
{
    // Set up the number of iterations to run
    const int num_iterations        = 1 << 20;
    const int num_threads           = 8;
    const int iterations_per_thread = num_iterations / num_threads;

    // Integer to increment
    int sink = 0;

// Run this loop in multiple threads
#pragma omp parallel num_threads(8)
    {
        for (int i = 0; i < iterations_per_thread; i++)
        {
// Say this is a critical section
#pragma omp critical
            sink++;
        }
    }
    assert(sink == num_iterations);

    return 0;
}
