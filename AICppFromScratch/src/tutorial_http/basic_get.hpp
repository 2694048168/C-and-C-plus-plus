/**
 * @file basic_get.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-24
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <format>
#include <iostream>
#include <string>

using json = nlohmann::json;

namespace Ithaca {

void BasicGetExample()
{
    std::cout << "\n---->[Begin] The GET method for HTTP\n\n";

    const std::string url = "http://jsonplaceholder.typicode.com";
    httplib::Client   client(url);

    const std::string path     = "/posts/1";
    const auto        response = client.Get(path);

    // ? 200 is OK, and response is std::shared_ptr<Response>, nullptr is fail
    if (response && 200 == response->status)
    {
        try
        {
            const json result = json::parse(response->body);
            const auto title  = result.at("title").get<std::string>();
            const auto body   = result.at("body").get<std::string>();

            std::cout << std::format("Title: {}\nBody: {}\n", title, body);
        }
        catch (const json::exception &ex)
        {
            std::cerr << "JSON exception: " << ex.what() << '\n';
        }
        catch (const std::exception &ex)
        {
            std::cerr << "C++ std exception: " << ex.what() << '\n';
        }
    }
    else
    {
        if (!response)
        {
            std::cout << "GET Request failed: no response received\n";
        }
        else
        {
            std::cout << std::format("GET Request failed: HTTP {}\n", response->status);
        }
    }

    std::cout << "\n---->[Een] The GET method for HTTP\n\n";
}

} // namespace Ithaca