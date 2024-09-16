/**
 * @file 00_cstdint.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief defined fixed-width integral types in modern C++
 * @version 0.1
 * @date 2024-08-12
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdint>
#include <iostream>

// ------------------------------------
int main(int argc, const char **argv)
{
    std::int32_t val = 42;
    std::cout << "the size betes of val: " << sizeof(val) << '\n';

    int num = 42;
    std::cout << "the size betes of num: " << sizeof(num) << '\n';

    std::int16_t val_16 = 42;
    std::cout << "the size betes of int16_t: " << sizeof(val_16) << '\n';

    short num_ = 42;
    std::cout << "the size betes of short: " << sizeof(num_) << '\n';


    std::int8_t val_8 = 42;
    std::cout << "the size betes of int8_t: " << sizeof(val_8) << '\n';

    std::int64_t val_64 = 42;
    std::cout << "the size betes of int64_t: " << sizeof(val_64) << '\n';

    return 0;
}
