/**
 * @file union.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-12
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief union data type
 * @attention
 *
 */

#include <iostream>
#include <cstdint>

union ipv4address
{
    uint32_t address32;
    uint8_t address8[4];
};

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    union ipv4address ip;

    std::cout << "sizeof(ip) = " << sizeof(ip) << std::endl;

    ip.address8[3] = 127;
    ip.address8[2] = 0;
    ip.address8[1] = 0;
    ip.address8[0] = 1;

    std::cout << "The address is ";
    std::cout << +ip.address8[3] << ".";
    std::cout << +ip.address8[2] << ".";
    std::cout << +ip.address8[1] << ".";
    std::cout << +ip.address8[0] << std::endl;

    std::cout << std::hex;
    std::cout << "in hex " << ip.address32 << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ union.cpp
 * $ clang++ union.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */