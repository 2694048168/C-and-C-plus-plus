/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-06-14
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <SI/stream.h>
#include <SI/velocity.h>

#include <iostream>

using namespace SI::literals;

// ---------------------------------
int main(int argc, const char **argv)
{
    constexpr auto                                  speed_of_a_swallow_in_m = 11.2_m_p_s;
    constexpr SI::kilometre_per_hour_t<long double> speed_in_km             = speed_of_a_swallow_in_m;

    std::cout << "Did you know that an unladen swallow travels at approximately " << speed_of_a_swallow_in_m
              << " which is " << speed_in_km << "\n";
}