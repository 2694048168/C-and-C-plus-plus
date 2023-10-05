/**
 * @file 05_calculateWithLoop.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-05
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <chrono>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <vector>

const long long     repeat = 1000;
constexpr long long size   = 100000000;

int main(int argc, const char **argv)
{
    std::cout << "\n";
    std::vector<int> randValues;
    randValues.reserve(size);

    // random values
    std::random_device              seed;
    std::mt19937                    engine(seed());
    std::uniform_int_distribution<> uniformDist(1, 10);
    for (long long i = 0; i < size; ++i)
    {
        randValues.push_back(uniformDist(engine));
    }

    // ==== Step 1 ===================
    auto calculate_1 = [&randValues]() -> double
    {
        const auto sta = std::chrono::steady_clock::now();

        unsigned long long sum = {};
        for (const auto &elem : randValues)
        {
            sum += elem;
        }

        const std::chrono::duration<double> dur = std::chrono::steady_clock::now() - sta;

        // std::cout << "Simple Loop to Calculate: \n";
        // std::cout << "Time for addition " << dur.count() << " seconds" << '\n';
        // std::cout << "Result: " << sum << "\n\n";

        return dur.count();
    };

    double time = 0.;
    for (size_t idx = 0; idx < repeat; ++idx)
    {
        time += calculate_1();
    }
    std::cout << "Simple Loop to Calculate: \n";
    std::cout << "Time for addition " << time/repeat << " seconds" << '\n';

    // ==== Step 2 STL algorithm ===================
    auto calculate_2 = [&randValues]()
    {
        const auto sta = std::chrono::steady_clock::now();

        unsigned long long sum = std::accumulate(randValues.begin(), randValues.end(), 0);

        const std::chrono::duration<double> dur = std::chrono::steady_clock::now() - sta;

        // std::cout << "Using STL algorithm to Calculate: \n";
        // std::cout << "Time for addition " << dur.count() << " seconds" << '\n';
        // std::cout << "Result: " << sum << "\n\n";

        return dur.count();
    };

    time = 0.;
    for (size_t idx = 0; idx < repeat; ++idx)
    {
        time += calculate_2();
    }
    std::cout << "Using STL algorithm to Calculate: \n";
    std::cout << "Time for addition " << time/repeat << " seconds" << '\n';
    
    // ==== Step 3 ===================
    auto calculate_3 = [&randValues]()
    {
        const auto sta = std::chrono::steady_clock::now();

        unsigned long long sum{};
        std::mutex         myMutex;
        for (const auto &elem : randValues)
        {
            std::lock_guard<std::mutex> myMutexGuard(myMutex);
            sum += elem;
        }

        const std::chrono::duration<double> dur = std::chrono::steady_clock::now() - sta;

        // std::cout << "Simple Loop Calculate with Protection with a Lock: \n";
        // std::cout << "Time for addition " << dur.count() << " seconds" << '\n';
        // std::cout << "Result: " << sum << "\n\n";

        return dur.count();
    };

    time = 0.;
    for (size_t idx = 0; idx < repeat; ++idx)
    {
        time += calculate_3();
    }
    std::cout << "Simple Loop Calculate with Protection with a Lock: \n";
    std::cout << "Time for addition " << time/repeat << " seconds" << '\n';
    // ==========================================================

    return 0;
}
