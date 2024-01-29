/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Caesar cipher
 * @version 0.1
 * @date 2024-01-29
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <iostream>
#include <string>
#include <string_view>

/**
 * @brief Caesar cipher
 * 
 * Write a program that can encrypt and decrypt messages using a Caesar cipher 
 * with a right rotation and any shift value. For simplicity, 
 * the program should consider only uppercase text messages and only encode letters,
 * ignoring digits, symbols, and other types of characters.
 * 
 */

/**
 * @brief Solution: 
------------------------------------------------------ */
std::string caesar_encrypt(std::string_view text, const int shift)
{
    std::string str;
    str.reserve(text.length());
    for (const auto &c : text)
    {
        if (isalpha(c) && isupper(c))
            str += 'A' + (c - 'A' + shift) % 26;
        else
            str += c;
    }

    return str;
}

std::string caesar_decrypt(std::string_view text, const int shift)
{
    std::string str;
    str.reserve(text.length());
    for (const auto &c : text)
    {
        if (isalpha(c) && isupper(c))
            str += 'A' + (26 + c - 'A' - shift) % 26;
        else
            str += c;
    }

    return str;
}

// ------------------------------
int main(int argc, char **argv)
{
    auto text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    for (int i = 1; i <= 26; ++i)
    {
        auto enc = caesar_encrypt(text, i);
        auto dec = caesar_decrypt(enc, i);
        assert(text == dec);
        std::cout << "Encrypt and decrypt message successfully\n";
    }

    return 0;
}
