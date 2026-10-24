/**
 * @file nested_objects.hpp
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

void NestedExample()
{
    std::cout << "---->[Begin] The Nested Example for JSON\n\n";

    json student = {
        {    "name", "Wei Li"},
        {     "age",       42},
        {      "id",    28881},
        {"enrolled",     true},

        // clang-format off
        { "address", {
            {"street", "No.324 Coean Drive"}, 
            {"city", "Minge"}, 
            {"state", "Floda"}}
        }
        // clang-format on
    };

    std::cout << student << std::endl << std::endl;
    std::cout << std::setw(2) << student << std::endl << std::endl;

    json course = {
        { "code",            "CS2301"},
        {"title", "AI via Modern C++"},

        // clang-format off
        {"instructor",{
            {"name", "Ithaca"},
            {"emial", "Ithaca@unitestrry.edu.cn"},
            {"office", "No. 302"}
        }},

        {"schedule",{
            {"days", "Mon/Wed/Fri"},
            {"time", "14:00 PM"},
            {"room", "Lab. 102"}
        }}
        // clang-format on
    };

    std::cout << course << std::endl << std::endl;
    std::cout << std::setw(2) << course << std::endl << std::endl;

    json student_array = json::array();
    student_array.push_back({
        { "name", "Larry"},
        {  "age",      25},
        {"agent",    true}
    });
    student_array.push_back({
        { "name", "Itahca"},
        {  "age",       45},
        {"agent",     true}
    });
    student_array.push_back({
        { "name", "carry"},
        {  "age",      35},
        {"agent",   false}
    });

    std::cout << student_array << std::endl << std::endl;
    std::cout << std::setw(2) << student_array << std::endl << std::endl;

    // ?Object containing arrays
    json department = {
        // clang-format off
        {"name", "Computer Science"},
        {"building", "Tech. Hall"},
        {"instructor", {
            "Dr. Frank",
            "Dr. Simth",
            "Prof. Johson"
        }},
        {"course", {
            "CS201",
            "CS301",
            "CS501"
        }},
        // clang-format on
    };

    std::cout << department << std::endl << std::endl;
    std::cout << std::setw(2) << department << std::endl << std::endl;

    std::cout << "---->[end] The Nested Example for JSON\n\n";
}

} // namespace Ithaca
