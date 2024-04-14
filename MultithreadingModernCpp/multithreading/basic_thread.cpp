/**
 * @file basic_thread.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-14
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#define _USE_MATH_DEFINES
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <thread>
#include <vector>

double work(unsigned int idx, double input)
{
    return abs(sin(input * idx)) * input;
}

void doWork(double *data, unsigned int count, unsigned int offset = 0)
{
    for (unsigned int idx = offset; idx < count; ++idx)
    {
        data[idx] = work(idx, data[idx]);
    }
}

void doWorkMT(double *data, unsigned int count, unsigned int threadCount)
{
    // work per thread
    unsigned int wpt = count / threadCount;

    // Schedule work
    std::vector<std::thread> threads;
    for (unsigned int idx = 0; idx < threadCount; idx++)
    {
        // create thread
        threads.push_back(std::thread(doWork, data, wpt, idx * wpt));
    }

    // wait for threads to finish
    for (auto iter = threads.begin(); iter != threads.end(); ++iter)
    {
        iter->join();
    }
}

// -----------------------------------
int main(int argc, const char **argv)
{
    // heavy work count
    const unsigned int workCount = 1024 * 1024 * 128;
    // const unsigned int workCount = 1024 * 1024 * 256;

    std::cout << "======== Generate Work ========\n";
    // generate work
    double *workData = (double *)malloc(sizeof(double) * workCount);
    std::srand(time(0));
    for (unsigned int idx = 0; idx < workCount; ++idx)
    {
        workData[idx] = std::rand() / (double)RAND_MAX;
    }

    std::cout << "======== Start Work ========\n";
    // work and time
    auto start_time = std::chrono::steady_clock::now();

    // doWork(workData, workCount);
    const unsigned int core_num = std::thread::hardware_concurrency();
    std::cout << "The core number is : " << core_num << '\n';
    // doWorkMT(workData, workCount, 1);
    doWorkMT(workData, workCount, core_num);

    // work work work
    auto end_time = std::chrono::steady_clock::now();

    // print timing result
    std::cout << "\n======== Timings ========"
              << "\nNS: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()
              << "\nUS: " << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()
              << "\nMS: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
              << "\n S: " << std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    // free memory
    if (nullptr != workData)
    {
        free(workData);
        workData = nullptr;
    }

    return 0;
}
