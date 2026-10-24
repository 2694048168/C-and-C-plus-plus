/**
 * @file accessing_json.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-23
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "nlohmann/json.hpp"

#include <format>
#include <iomanip>
#include <iostream>
#include <string>

using json = nlohmann::json;

namespace Ithaca {

void AccessingExample()
{
    std::cout << "---->[Begin] The Accessing JSON data Example\n\n";

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

    // ?Accessing simple object properity using [] operator
    std::string name     = student["name"];
    int         age      = student["age"];
    bool        enrolled = student["enrolled"];
    std::cout << name << " age: " << age << " and enroled is " << std::boolalpha << enrolled << "\n";

    json instrutor = {
        {   "name",    "Wei Li"},
        {    "age",          42},
        {   "role", "Professor"},
        {"tenured",        true},
    };

    // ?Accessing simple object properity using at() method
    std::string name_    = instrutor.at("name");
    int         age_     = instrutor.at("age");
    std::string role_    = instrutor.at("role");
    bool        tenured_ = instrutor.at("tenured");
    std::cout << role_ << " " << name_ << " age: " << age_ << " and tenured is " << std::boolalpha << tenured_ << "\n";

    std::cout << "---->[end] The Accessing JSON data Example\n\n";

    // ?accessing object with nested
    std::string street = student["address"]["street"];
    std::string city   = student.at("address").at("city");
    std::string state  = student.at("address").at("state");

    auto message = std::format("The address: {} and city {} . {} \n", street, city, state);
    std::cout << message << std::endl;

    // ?accessing object with array data or elementment
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

    size_t size_array = student_array.size();

    std::string first  = student_array[0]["name"];
    std::string second = student_array.at(1).at("name");
    std::string third  = student_array.at(2).at("name");
    std::cout << "The size of array is: " << size_array;
    std::cout << " and they are " << first << ", " << second << ", " << third << "\n";

    // ?Accessing complex structures (OpenAI response)
    // clang-format off
    json response = {
        {"id", "chatcmpl-123"},
        {"model", "gpt-6"},
        // 关键修改：直接用大括号代表数组，内部用大括号包裹成一个完整的对象
        {"choices", { 
            {  // <-- choices[0] 对象的开头
                {"index", 0},
                {"message", {
                    {"role", "assistant"},
                    {"content", "Smart pointers are RAII..."}
                }},
                {"finish_reason", "stop"}
            }  // <-- choices[0] 对象的结尾
        }}
    };
    // clang-format on

    std::string model = response.at("model");

    // !===== debug code ===========
    // 依次打印每一层的底层类型，看看是谁变成了非预期类型
    std::cout << "choices type is: " << response.at("choices").type_name() << std::endl;
    // 预期应该是: array

    std::cout << "choices[0] type is: " << response.at("choices").at(0).type_name() << std::endl;
    // 预期应该是: object

    std::cout << "message type is: " << response.at("choices").at(0).at("message").type_name() << std::endl;
    // 预期应该是: object
    // !===== debug code ===========

    std::string role     = response.at("choices").at(0).at("message").at("role");
    std::string content  = response.at("choices").at(0).at("message").at("content");
    std::string finished = response.at("choices").at(0).at("finish_reason");

    auto message_ = std::format("The model is {} and role is {}, get message {} by finished {}\n", model, role, content,
                                finished);
    std::cout << message_ << std::endl;
}
} // namespace Ithaca