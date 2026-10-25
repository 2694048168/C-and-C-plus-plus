/**
 * @file basic_post.hpp
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

void BasicPostExample()
{
    std::cout << "\n---->[Begin] The POST method for HTTP\n\n";

    const json post_data = {
        { "title",                        "Modern AI and C++"},
        {  "body", "This is a tutorial for AI via Modern C++"},
        {"userId",                                       2851},
    };

    const std::string url = "http://jsonplaceholder.typicode.com";
    httplib::Client   client(url);

    const std::string path      = "/posts";
    const std::string data_type = "application/json";
    const auto        response  = client.Post(path, post_data.dump(), data_type);

    // ? 201 is OK by Created, and response is std::shared_ptr<Response>, nullptr is fail
    if (response && 201 == response->status)
    {
        try
        {
            const json result = json::parse(response->body);

            const auto id     = result.at("id").get<int>();
            const auto title  = result.at("title").get<std::string>();
            const auto body   = result.at("body").get<std::string>();
            const auto userId = result.at("userId").get<int>();

            std::cout << std::format("POST ID created by Server: {}\nTitle: {}\nBody: {}\nuserID: {}", id, title, body,
                                     userId);
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
            std::cout << "POST Request failed: no response received\n";
        }
        else
        {
            std::cout << std::format("POST Request failed: HTTP {}\n", response->status);
        }
    }

    std::cout << "\n---->[End] The POST method for HTTP\n\n";
}

} // namespace Ithaca