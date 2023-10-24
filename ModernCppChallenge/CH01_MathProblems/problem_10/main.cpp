/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-24
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <bitset>
#include <iostream>
#include <string>

/* Gray code
Write a program that displays the normal binary representations, 
Gray code representations, and decoded Gray code values for all 5-bit numbers.
--------------------------------------- */

/* Solution
Gray code, also known as reflected binary code or simply reflected binary, 
is a form of binary encoding where two consecutive numbers differ by only one bit.
To perform a binary reflected Gray code encoding, 
we need to use the following formula: 

移位操作，操作系统的位数极度高相关
--------------------------------------------- */
unsigned int gray_encode(const unsigned int num)
{
    return num ^ (num >> 1);
}

unsigned int gray_decode(unsigned int gray)
{
    for (unsigned int bit = 1U << 31; bit > 1; bit >>= 1)
    {
        if (gray & bit)
        {
            gray ^= bit >> 1;
        }
    }

    return gray;
}

std::string to_binary(unsigned int value, const int digits)
{
    return std::bitset<32>(value).to_string().substr(32 - digits, digits);
}

// -----------------------------
int main(int argc, char **argv)
{
    std::cout << "Number\tBinary\tGray\tDecoded\n";
    std::cout << "------\t------\t----\t-------\n";

    for (unsigned int n = 0; n < 32; ++n)
    {
        auto encode_gray = gray_encode(n);
        auto decode_gray = gray_decode(encode_gray);

        std::cout << n 
        << "\t" << to_binary(n, 5) 
        << "\t" << to_binary(encode_gray, 5) 
        << "\t" << decode_gray << "\n";
    }

    return 0;
}
