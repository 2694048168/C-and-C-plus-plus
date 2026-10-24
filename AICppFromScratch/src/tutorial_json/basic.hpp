/**
 * @file basic.hpp
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

void BasicExample()
{
    std::cout << "---->[Begin] The Basic for JSON\n\n";

    json instructor;
    instructor["name"]    = "Ithaca";
    instructor["role"]    = "Master";
    instructor["subject"] = "C++";

    // ?compact and pretty print into console
    std::cout << instructor << std::endl << std::endl;
    std::cout << std::setw(4) << instructor << std::endl << std::endl;

    json student = {
        {    "name", "Wei Li"},
        {     "age",       42},
        {"enrolled",     true}
    };

    std::cout << student << std::endl << std::endl;
    std::cout << std::setw(2) << student << std::endl << std::endl;

    std::string simple_json = R"(
    {
        "name": "Moe",
        "age": 25,
        "cite": "shanghai",
        "enrolled": true
    }
    )";

    json person = json::parse(simple_json);
    std::cout << person << std::endl << std::endl;
    std::cout << std::setw(2) << person << std::endl << std::endl;

    // ?use dump when you need std::string serialization
    // compact string
    std::string compact = student.dump();
    // pretty print string with 2-space indent
    std::string pretty = student.dump(2);

    std::cout << compact << std::endl << std::endl;
    std::cout << pretty << std::endl << std::endl;

    std::cout << "---->[End] The Basic for JSON\n\n";
}

} // namespace Ithaca
