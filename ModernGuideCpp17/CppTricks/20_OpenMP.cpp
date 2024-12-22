/**
 * @file 20_OpenMP.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <Windows.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#include <cassert>
#include <chrono>
#include <iostream>

inline constexpr long NUM_STEP = 1000000;

double computer_pi_seq(const long num_steps)
{
    double step = 1.0 / num_steps;
    double sum  = 0.0;
    for (long idx = 0; idx < num_steps; ++idx)
    {
        double x = (idx + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    return sum * step;
}

double computer_pi_parallel(const long num_steps)
{
    double step = 1.0 / num_steps;
    double sum  = 0.0;
    double x    = 0.0;
// parallelize loop and reduce
#pragma omp parallel for reduction(+ : sum) private(x)
    for (long idx = 0; idx < num_steps; ++idx)
    {
        x = (idx + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    return sum * step;
}

volatile DWORD dwStart;
volatile int   global = 0;

double test_OpenMP(int num_steps)
{
    int i;
    global++;
    double x, pi, sum = 0.0, step;

    step = 1.0 / (double)num_steps;

#pragma omp parallel for reduction(+ : sum) private(x)
    for (i = 1; i <= num_steps; i++)
    {
        x   = (i - 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }

    pi = step * sum;
    return pi;
}

// ------------------------------------
int main(int argc, const char *argv[])
{
    auto start_time = std::chrono::high_resolution_clock::now();
    auto pi_seq     = computer_pi_seq(NUM_STEP);
    auto end_time   = std::chrono::high_resolution_clock::now();
    std::cout << "The time of computer_pi_seq: "
              << std::chrono::duration<float, std::milli>(end_time - start_time).count() << " ms\n";

    auto start_time_ = std::chrono::high_resolution_clock::now();
    auto pi_parallel = computer_pi_parallel(NUM_STEP);
    auto end_time_   = std::chrono::high_resolution_clock::now();
    std::cout << "The time of computer_pi_parallel: "
              << std::chrono::duration<float, std::milli>(end_time_ - start_time_).count() << " ms\n";

    assert(pi_parallel == pi_seq);
    std::cout << "The value of PI(seq): " << pi_seq << '\n';
    std::cout << "The value of PI(parallel): " << pi_parallel << '\n';

    dwStart  = GetTickCount();
    double d = test_OpenMP(NUM_STEP);
    printf_s("For %d steps, pi = %.15f, %d milliseconds\n", NUM_STEP, d, GetTickCount() - dwStart);

    dwStart = GetTickCount();
    d       = test_OpenMP(NUM_STEP);
    printf_s("For %d steps, pi = %.15f, %d milliseconds\n", NUM_STEP, d, GetTickCount() - dwStart);

    return 0;
}
