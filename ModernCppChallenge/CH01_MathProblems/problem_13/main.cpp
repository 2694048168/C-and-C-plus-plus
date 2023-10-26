/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <utility>
#include <vector>


/* Computing the value of Pi
Write a program that computes the value of Pi 
 with a precision of two decimal digits.

 A suitable solution for approximately determining the value of Pi 
 is using a Monte Carlo simulation.

 This is a method that uses random samples of inputs to explore the behavior of
complex processes or systems. The method is used in a large variety of applications
and domains, including physics, engineering, computing, finance, business, and others.
------------------------------------------------ */

/* Solution
依靠以下想法：直径为 d 的圆的面积是 PI * d^2/4;
边长等于 d 的正方形的面积为 d^2;
两者面积的比值恒定为 PI/4.
如果把圆放在正方形里面，生成随机数, 在正方形内均匀分布，
则圆中的数字数应为与圆形面积成正比，并且正方形内的数字计数应与正方形的面积成正比.
这意味着除以在正方形和圆形中的命中应该给出 PI/4, 生成的点数越多结果应准确

For generating pseudo-random numbers we will use
 a Mersenne twister and a uniform statistical distribution:
----------------------------------------------------------------- */
template<typename E = std::mt19937, typename D = std::uniform_real_distribution<>>
double compute_pi(E &engine, D &dist, const unsigned int samples = 1000000)
{
    unsigned hit = 0;
    for (unsigned i = 0; i < samples; ++i)
    {
        auto x = dist(engine);
        auto y = dist(engine);

        if (y <= std::sqrt(1 - std::pow(x, 2)))
            hit += 1;
    }

    return 4.0 * hit / samples;
}

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

void test_demo()
{
    std::random_device rd;

    auto seed_data = std::array<int, std::mt19937::state_size>{};

    std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
    std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
    auto          eng  = std::mt19937{seq};
    auto          dist = std::uniform_real_distribution<>{0, 1};

    for (auto j = 0; j < 10; j++)
    {
        std::cout << compute_pi(eng, dist) << std::endl;
    }
}

// -----------------------------
int main(int argc, char **argv)
{
    std::cout << "=====================================\n";
    // 测试耗时
    auto timer1 = perf_timer<>::duration(test_demo);

    std::cout << "=====================================\n";
    std::cout << std::chrono::duration<double, std::milli>(timer1).count() << " ms\n";

    return 0;
}
