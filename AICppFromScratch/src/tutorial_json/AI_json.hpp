/**
 * @file AI_json.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-23
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "nlohmann/json.hpp"

#include <iomanip>
#include <iostream>
#include <string>

using json = nlohmann::json;

namespace Ithaca {

void AIExample()
{
    std::cout << "---->[Begin] The AI Example for JSON\n\n";

    json image_request = {
        {  "model",                                 "dall-e-3"},
        { "prompt", "A C++ Programmer teaching in Miami Beach"},
        {    "num",                                          1},
        {   "size",                                "1024x1024"},
        {"quality",                                 "standard"},
        {  "style",                                    "vivid"},
    };

    std::cout << image_request << std::endl << std::endl;
    std::cout << std::setw(2) << image_request << std::endl << std::endl;

    json request = {
        {      "model", "gpt-6"},

        // clang-format off
        {"message", json::array({
            {
                {"role", "system"},
                {"content", "You are a C++ tutor"}
            },
            {
                {"role", "user"},
                {"content", "Explain Smarat Pointer"}
            }
        })
        },
        // clang-format on

        {"temperature",    0.75},
        {  "max_token",     120},
    };

    std::cout << request << std::endl << std::endl;
    std::cout << std::setw(2) << request << std::endl << std::endl;

    std::cout << "---->[end] The AI Example for JSON\n\n";
}

} // namespace Ithaca
