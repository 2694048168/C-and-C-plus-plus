/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Vigenère cipher
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
 * @brief Vigenère cipher
 * 
 * Write a program that can encrypt and decrypt messages using the Vigenère cipher.
 * For simplicity, the input plain-text messages for encryption 
 * should consist of only uppercase letters.
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

std::string build_vigenere_table()
{
    std::string table;
    table.reserve(26 * 26);

    for (int i = 0; i < 26; ++i) table += caesar_encrypt("ABCDEFGHIJKLMNOPQRSTUVWXYZ", i);

    return table;
}

std::string vigenere_encrypt(std::string_view text, std::string_view key)
{
    std::string result;
    result.reserve(text.length());

    static auto table = build_vigenere_table();

    for (size_t i = 0; i < text.length(); ++i)
    {
        auto row = key[i % key.length()] - 'A';
        auto col = text[i] - 'A';

        result += table[row * 26 + col];
    }

    return result;
}

std::string vigenere_decrypt(std::string_view text, std::string_view key)
{
    std::string result;
    result.reserve(text.length());

    static auto table = build_vigenere_table();

    for (size_t i = 0; i < text.length(); ++i)
    {
        auto row = key[i % key.length()] - 'A';

        for (size_t col = 0; col < 26; col++)
        {
            if (table[row * 26 + col] == text[i])
            {
                result += 'A' + col;
                break;
            }
        }
    }

    return result;
}

// ------------------------------
int main(int argc, char **argv)
{
    auto text = "THECPPCHALLENGER";
    auto enc  = vigenere_encrypt(text, "SAMPLE");
    auto dec  = vigenere_decrypt(enc, "SAMPLE");

    assert(text == dec);
    std::cout << "Encrypt and decrypt is successfully\n";

    return 0;
}
