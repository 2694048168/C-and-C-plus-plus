/**
 * @file iterating_json.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-24
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "nlohmann/json.hpp"

#include <iomanip>
#include <iostream>
#include <string>
#include <typeinfo>

using json = nlohmann::json;

namespace Ithaca {

void IteratingExample()
{
    std::cout << "---->[Begin] The iterating Example for JSON\n\n";

    json courses = {"Beginner for C++", "Modern C++ and C++ 20", "AI with modern C++"};

    std::cout << "the size of array: " << courses.size() << "\n";

    // ? loop through array read-only(const)
    for (const auto &elem : courses)
    {
        std::cout << "the cource is: " << elem << "\n";
    }
    // ! the json object and C++ object
    for (auto &elem : courses)
    {
        elem = elem.get<std::string>() + " update";
        std::cout << "the cource is: " << elem.get<std::string>() + " update" << "\n";
    }
    std::cout << "the json array: " << courses << "\n";

    // * loop through object(key-value pairs)
    json instructor = {
        {"name",     "Frank"},
        { "age",          33},
        {"role", "Professor"},
    };

    for (const auto &[key, value] : instructor.items())
    {
        // ?the key and value type is C++ or json-obj?
        std::cout << "the key: " << key << " and value: " << value << std::endl;

        std::cout << "The key type: " << typeid(key).name() << "\n";
        std::cout << "The value type-name: " << value.type_name() << "\n";
    }

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

    for (const auto &elem : student_array)
    {
        std::cout << elem.at("name").get<std::string>() << " is " << elem.at("age").get<int>() << " years old\n";
    }

    // * Looping  through nested strtucted
    json response_msg = {
        {  "model","gpt-6"  },
        {"message",
         {
         {{"role", "system"}, {"content", "You are a tutor"}},
         {{"role", "user"}, {"content", "Expalin pointers"}},
         {{"role", "assistant"}, {"content", "pointes smart store address..."}},
         }}
    };

    const auto &message_arr = response_msg.at("message");
    for (const auto &msg : message_arr)
    {
        std::cout << msg.at("role") << " : " << msg.at("content") << std::endl;
    }

    std::cout << "---->[end] The iterating Example for JSON\n\n";
}

} // namespace Ithaca