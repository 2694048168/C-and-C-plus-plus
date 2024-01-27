/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Creating verification text PNG images
 * @version 0.1
 * @date 2024-01-27
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "pngwriter.h"

#include <array>
#include <iostream>
#include <random>
#include <string>
#include <string_view>

/**
 * @brief Creating verification text PNG images
 * 
 * Write a program that can create Captcha-like PNG images for verifying human users
 *  to a system. Such an image should have:
 * 1. A gradient-colored background
 * 2. A series of random letters displayed at different angles both to the right and left
 * 3. Several random lines of different colors across the image (on top of the text)
 * 
 */

/**
 * @brief Solution: pngwriter library in C++
 https://github.com/pngwriter/pngwriter
------------------------------------------------------ */
void create_image(const int width, const int height, std::string_view font, const int font_size,
                  std::string_view filepath)
{
    pngwriter image{width, height, 0, filepath.data()};

    std::random_device rd;
    std::mt19937       mt;
    auto               seed_data = std::array<int, std::mt19937::state_size>{};
    std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
    std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
    mt.seed(seq);
    std::uniform_int_distribution<> udx(0, width);
    std::uniform_int_distribution<> udy(0, height);
    std::uniform_int_distribution<> udc(0, 65535);
    std::uniform_int_distribution<> udt(0, 25);

    // gradient background
    for (int iter = 0; iter <= width; iter++)
    {
        image.line(iter, 0, iter, height, 65535 - int(65535 * ((double)iter) / (width)),
                   int(65535 * ((double)iter) / (width)), 65535);
    }

    // random text
    std::string font_family = font.data();
    for (int i = 0; i < 6; ++i)
    {
        image.plot_text(
            // font
            font_family.data(), font_size,
            // position
            i * width / 6 + 10, height / 2 - 10,
            // angle
            (i % 2 == 0 ? -1 : 1) * (udt(mt) * 3.14) / 180,
            // text
            std::string(1, char('A' + udt(mt))).data(),
            // color
            0, 0, 0);
    }

    // random lines
    for (int i = 0; i < 4; ++i)
    {
        image.line(udx(mt), 0, udx(mt), height, udc(mt), udc(mt), udc(mt));

        image.line(0, udy(mt), width, udy(mt), udc(mt), udc(mt), udc(mt));
    }

    image.close();
}

// ------------------------------
int main(int argc, char **argv)
{
    std::string font_path;

#ifdef _WIN32
    font_path = R"(c:\windows\fonts\arial.ttf)";
#elif defined(__APPLE__)
    font_path = R"(/Library/Fonts/Arial.ttf)";
#else
    std::cout << "Font path: ";
    std::cin >> font_path;
#endif

    create_image(200, 50, font_path, 18, "validation.png");

    return 0;
}
