/**
 * @file safely_accessing.hpp
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

using json = nlohmann::json;

namespace Ithaca {

void SafelyExample()
{
    std::cout << "---->[Begin] The Safely Accessing Example for JSON\n\n";

    json response = {
        {"model", "gpt-6"},
        {"temp.",    0.75}
    };

    std::string model = response["model"]; // OK
    // ! json::type_error exception! when assigning!
    // ? [] 方式会自动插入一个键值对
    // std::string missing = response["missing"]; // !NG

    // ? at method - throws if key is missing
    std::string model_1 = response.at("model"); // OK
    // ! throws json::out_of_range exception! when accessing!
    // ? at 不会插入不存在的 key, 访问就会抛异常
    // std::string missing = response.at("missing"); // !NG

    // ! throws json_type_error exception!
    // ?double --> string cannot convert
    // std::string missing = response.at("temp."); // !NG

    // * 正确安全的访问逻辑：key 必须判断是否存在，同时获取的类型是否正确
    // * json 提供 contains and type check
    if (response.contains("model") && response.at("model").is_string())
    {
        std::string model = response.at("model");
    }

    // * using value() to provide default value
    // * check first and if key does NOT exist assign default value
    std::string name = response.value("name", "Frank");

    if (response.contains("temp.") && response.at("temp.").is_number())
    {
        double temp = response.at("temp.").get<double>();
    }

    try
    {
        auto model = response.at("model").get<std::string>();
        auto temp  = response.at("temp.").get<double>();
    }
    catch (const json::exception &ex)
    {
        std::cerr << "Invalid JSON configuration: " << ex.what() << '\n';
    }

    // ! bad json data
    std::string bad_json = R"(
        {
            "name": "Frank",
            "age": 35,
            "tenuded": false,
        }
    )";

    try
    {
        json j = json::parse(bad_json);

        // the line is never reached if parsing fails
        std::string name = j.at("name").get<std::string>();
    }
    catch (const json::parse_error &ex)
    {
        std::cerr << "json parse error :" << ex.what() << '\n';
    }
    catch (const json::exception &ex)
    {
        std::cerr << "json other exception :" << ex.what() << '\n';
    }
    catch (const std::exception &ex)
    {
        std::cerr << "json C++ exception :" << ex.what() << '\n';
    }

    std::cout << "---->[end] The Safely Accessing Example for JSON\n\n";
}

} // namespace Ithaca