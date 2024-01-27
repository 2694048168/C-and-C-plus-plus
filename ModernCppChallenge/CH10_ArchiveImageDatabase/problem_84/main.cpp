/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief EAN-13 barcode generator
 * @version 0.1
 * @date 2024-01-27
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "pngwriter.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <iostream>
#include <numeric>
#include <string_view>

/**
 * @brief EAN-13 barcode generator
 * 
 * Write a program that can generate a PNG image with an EAN-13 barcode for any
 * international article number in version 13 of the standard. For simplicity, 
 * the image should only contain the barcode and can skip the EAN-13 number
 * printed under the barcode.
 * 
 */

/**
 * @brief Solution: pngwriter library in C++
 https://github.com/pngwriter/pngwriter
------------------------------------------------------ */
struct ean13
{
public:
    ean13(std::string_view code)
    {
        if (code.length() == 13)
        {
            if (code[12] != '0' + get_crc(code.substr(0, 12)))
                throw std::runtime_error("Not an EAN-13 format.");

            number = code;
        }
        else if (code.length() == 12)
        {
            number = code.data() + std::string(1, '0' + get_crc(code));
        }
    }

    ean13(unsigned long long code)
        : ean13(std::to_string(code))
    {
    }

    std::array<unsigned char, 13> to_array() const
    {
        std::array<unsigned char, 13> result;
        for (int i = 0; i < 13; ++i) result[i] = static_cast<unsigned char>(number[i] - '0');
        return result;
    }

    std::string to_string() const noexcept
    {
        return number;
    }

private:
    unsigned char get_crc(std::string_view code)
    {
        unsigned char weights[12] = {1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3};
        size_t        index       = 0;
        auto          sum         = std::accumulate(std::begin(code), std::end(code), 0,
                                                    [&weights, &index](const int total, const char c)
                                                    { return total + weights[index++] * (c - '0'); });

        return 10 - sum % 10;
    }

    std::string number;
};

struct ean13_barcode_generator
{
    void create(const ean13 &code, std::string_view filename, const int digit_width = 3, const int height = 50,
                const int margin = 10)
    {
        pngwriter image(margin * 2 + 95 * digit_width, height + margin * 2, 65535, filename.data());

        std::array<unsigned char, 13> digits = code.to_array();

        int x = margin;
        x     = draw_digit(marker_start, 3, image, x, margin, digit_width, height);

        for (int i = 0; i < 6; ++i)
        {
            int code = encodings[digits[1 + i]][eandigits[digits[0]][i]];
            x        = draw_digit(code, 7, image, x, margin, digit_width, height);
        }

        x = draw_digit(marker_center, 5, image, x, margin, digit_width, height);

        for (int i = 0; i < 6; ++i)
        {
            int code = encodings[digits[7 + i]][2];
            x        = draw_digit(code, 7, image, x, margin, digit_width, height);
        }

        x = draw_digit(marker_end, 3, image, x, margin, digit_width, height);

        image.close();
    }

private:
    int draw_digit(unsigned char code, unsigned int size, pngwriter &image, const int x, const int y,
                   const int digit_width, const int height)
    {
        std::bitset<7> bits(code);
        int            pos = x;
        for (int i = size - 1; i >= 0; --i)
        {
            if (bits[i])
            {
                image.filledsquare(pos, y, pos + digit_width, y + height, 0, 0, 0);
            }

            pos += digit_width;
        }

        return pos;
    }

    unsigned char encodings[10][3] = {
        {0b0001101, 0b0100111, 0b1110010},
        {0b0011001, 0b0110011, 0b1100110},
        {0b0010011, 0b0011011, 0b1101100},
        {0b0111101, 0b0100001, 0b1000010},
        {0b0100011, 0b0011101, 0b1011100},
        {0b0110001, 0b0111001, 0b1001110},
        {0b0101111, 0b0000101, 0b1010000},
        {0b0111011, 0b0010001, 0b1000100},
        {0b0110111, 0b0001001, 0b1001000},
        {0b0001011, 0b0010111, 0b1110100},
    };

    unsigned char eandigits[10][6] = {
        {0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 1, 1},
        {0, 0, 1, 1, 0, 1},
        {0, 0, 1, 1, 1, 0},
        {0, 1, 0, 0, 1, 1},
        {0, 1, 1, 0, 0, 1},
        {0, 1, 1, 1, 0, 0},
        {0, 1, 0, 1, 0, 1},
        {0, 1, 0, 1, 1, 0},
        {0, 1, 1, 0, 1, 0},
    };

    unsigned char marker_start  = 0b101;
    unsigned char marker_end    = 0b101;
    unsigned char marker_center = 0b01010;
};

// ------------------------------
int main(int argc, char **argv)
{
    assert("4006381333931" == ean13("400638133393").to_string());
    assert("0012345678905" == ean13("001234567890").to_string());
    assert("0012345678905" == ean13("001234567890").to_string());
    assert("8711253001202" == ean13("8711253001202").to_string());
    assert("5901234123457" == ean13("5901234123457").to_string());

    ean13_barcode_generator generator;

    generator.create(ean13("8711253001202"), "8711253001202.png", 5, 150, 30);

    generator.create(ean13("5901234123457"), "5901234123457.png", 5, 150, 30);

    return 0;
}
